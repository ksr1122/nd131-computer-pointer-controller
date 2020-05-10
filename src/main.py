import os
import argparse

from input_feeder import InputFeeder
from mouse_controller import MouseController

from face_detection import FaceDetect
from gaze_estimation import GazeEstimate
from head_pose_estimation import HeadPoseEstimate
from facial_landmarks_detection import FacialLandMarkDetect

class Pipeline:

    def __init__(self, args):
        self.log_level = "INFO" if os.environ.get("LOGLEVEL") == "INFO" or args.verbose_stage else "WARNING"
        log.basicConfig(level=self.log_level)

        input_type = 'cam' if args.cam else 'video'
        self.feed = InputFeeder(input_type, args.video)
        if not self.feed.load_data():
            raise Exception('Input valid image or video file')

        fps, w, h = self.feed.get_props()
        self.out_video = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*'MJPG'), fps, (w, h), True)

        args.head_pose_model = os.path.join(args.head_pose_model,
                                            args.precision, os.path.basename(args.head_pose_model))
        args.landmarks_model = os.path.join(args.landmarks_model,
                                            args.precision, os.path.basename(args.landmarks_model))
        args.gaze_model = os.path.join(args.gaze_model,
                                       args.precision, os.path.basename(args.gaze_model))

        self.fd = FaceDetect(args.face_model, args.device, args.extension, args.threshold)
        self.fd.load_model()
        self.fd.set_out_size(w, h)

        self.hp = HeadPoseEstimate(args.head_pose_model, args.device, args.extension, args.threshold)
        self.hp.load_model()

        self.fl = FacialLandMarkDetect(args.landmarks_model, args.device, args.extension, args.threshold)
        self.fl.load_model()

        self.gz = GazeEstimate(args.gaze_model, args.device, args.extension, args.threshold)
        self.gz.load_model()

        self.mc = MouseController()
        self.verbose_stage = args.verbose_stage

    def get_bounding_rect(self, x, y):
        width, height = 40, 20
        x1, y1 = x-int(width/2), y-int(height/2)
        x2, y2 = x+int(width/2), y+int(height/2)
        return x1, y1, x2, y2

    def verbose_stage_draw(self, frame, face_coord, eye_coord, head_pose_angles, mouse_coord):
        f_x1, f_y1, f_x2, f_y2 = face_coord
        self.fd.draw_rect(frame, (f_x1, f_y1), (f_x2, f_y2))

        e_x1, e_y1, e_x2, e_y2 = eye_coord
        left_x, left_y, right_x, right_y = self.get_bounding_rect(e_x1, e_y1)
        self.fl.draw_rect(frame, (f_x1+left_x, f_y1+left_y),
                                 (f_x1+right_x, f_y1+right_y))
        left_x, left_y, right_x, right_y = self.get_bounding_rect(e_x2, e_y2)
        self.fl.draw_rect(frame, (f_x1+left_x, f_y1+left_y),
                                 (f_x1+right_x, f_y1+right_y))

        text = "Yaw: {:+.0f}, Pitch: {:+.0f}, Roll: {:+.0f}".format(*head_pose_angles)
        self.hp.draw_text(frame, text, (100, 100))

        self.gz.draw_circle(frame, mouse_coord, 10)

    def run(self):
        abs_mouse_x = abs_mouse_y = 0
        for frame in self.feed.next_batch():
            f_x1, f_y1, f_x2, f_y2 = self.fd.predict(frame)

            face_frame = frame[f_y1:f_y2, f_x1:f_x2]

            if not face_frame.size: # skip if face not detected
                continue

            head_pose_angles = self.hp.predict(face_frame)

            self.fl.set_out_size(f_x2-f_x1, f_y2-f_y1)
            e_x1, e_y1, e_x2, e_y2 = self.fl.predict(face_frame)

            left_x, left_y, right_x, right_y = self.get_bounding_rect(e_x1, e_y1)
            left_eye_frame = face_frame[left_y:right_y,
                                        left_x:right_x]
            left_x, left_y, right_x, right_y = self.get_bounding_rect(e_x2, e_y2)
            right_eye_frame = face_frame[left_y:right_y,
                                         left_x:right_x]

            if not left_eye_frame.size or not right_eye_frame.size: # skip if eyes not detected
                continue

            g_x, g_y, _ = self.gz.predict(left_eye_frame, right_eye_frame, [[*head_pose_angles]])

            self.mc.move(g_x, g_y)

            if self.verbose_stage:
                _, w, h = self.feed.get_props()
                if abs_mouse_x == 0 and abs_mouse_y == 0:
                    abs_mouse_x = int(f_x1+(e_x1 + e_x2)/2)
                    abs_mouse_y = int(f_y1+(e_y1 + e_y2)/2)
                else:
                    abs_mouse_x += int(g_x * w / 250)
                    abs_mouse_y -= int(g_y * h / 250)

                self.verbose_stage_draw(frame,
                                        (f_x1, f_y1, f_x2, f_y2),
                                        (e_x1, e_y1, e_x2, e_y2),
                                        head_pose_angles,
                                        (abs_mouse_x, abs_mouse_y))

            self.out_video.write(frame)

    def close(self):
        self.feed.close()
        self.out_video.release()

def main(args):
    pipeline = Pipeline(args)
    pipeline.run()
    pipeline.close()

def build_argparser():
    """
    Parse command line arguments.
    :return: command line arguments
    """
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-i', '--video', default='../bin/demo.mp4',
                        help='Path to input image or video file')
    group.add_argument('-c', '--cam', action='store_true',
                        help='Use camera as input')
    parser.add_argument('-v', '--verbose_stage', action='store_true',
                        help='Display the outputs of intermediate models')
    parser.add_argument('-o', '--out', default='out.mp4',
                        help='Path to output image or video file')
    parser.add_argument('-d', '--device', default='CPU',
                        help='Target Device: CPU(default)/GPU/MYRIAD/FPGA')
    parser.add_argument('-p', '--precision', default='FP32',
                        help='Model precision: FP32(default)/FP16/FP16-INT8')
    parser.add_argument('-e', '--extension', default=None,
                        help='Absolute path to a shared library with kernel impl for custom layers')
    parser.add_argument('-t', '--threshold', default=0.60,
                        help='Probability threshold for detections filtering')
    parser.add_argument('--gaze_model',
                        default="../intel/gaze-estimation-adas-0002",
                        help='Path to Gaze Estimation model')
    parser.add_argument('--head_pose_model',
                        default="../intel/head-pose-estimation-adas-0001",
                        help='Path to Head Pose Detection model')
    parser.add_argument('--face_model',
                        default="../intel/face-detection-adas-binary-0001/FP32-INT1/face-detection-adas-binary-0001",
                        help='Path to Face Detection model')
    parser.add_argument('--landmarks_model',
                        default="../intel/landmarks-regression-retail-0009",
                        help='Path to Facial Lanndmarks Estimation model')
    return parser


if __name__ == '__main__':
    args = build_argparser().parse_args()
    main(args)

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

    def run(self):
        pass

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

import argparse

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


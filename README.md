# Computer Pointer Controller

`Objective`: To control the mouse pointer of a computer using gaze detection model. 

`Description`: The Gaze Estimation model is used to estimate the gaze of the user's eyes and it is applied to `pyautogui` module to change the mouse pointer position accordingly. This project demonstrates the ability to run multiple models in the same machine and coordinate the flow of data between those models.

## Project Set Up and Installation
Project source structure is as shown below.

```
pointer-controller
├── README.md
├── bin
│   └── demo.mp4
├── intel
│   ├── face-detection-adas-binary-0001
│   │   └── FP32-INT1
│   │       ├── face-detection-adas-binary-0001.bin
│   │       └── face-detection-adas-binary-0001.xml
│   ├── gaze-estimation-adas-0002
│   │   ├── FP16
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   ├── FP16-INT8
│   │   │   ├── gaze-estimation-adas-0002.bin
│   │   │   └── gaze-estimation-adas-0002.xml
│   │   └── FP32
│   │       ├── gaze-estimation-adas-0002.bin
│   │       └── gaze-estimation-adas-0002.xml
│   ├── head-pose-estimation-adas-0001
│   │   ├── FP16
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   ├── FP16-INT8
│   │   │   ├── head-pose-estimation-adas-0001.bin
│   │   │   └── head-pose-estimation-adas-0001.xml
│   │   └── FP32
│   │       ├── head-pose-estimation-adas-0001.bin
│   │       └── head-pose-estimation-adas-0001.xml
│   └── landmarks-regression-retail-0009
│       ├── FP16
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       ├── FP16-INT8
│       │   ├── landmarks-regression-retail-0009.bin
│       │   └── landmarks-regression-retail-0009.xml
│       └── FP32
│           ├── landmarks-regression-retail-0009.bin
│           └── landmarks-regression-retail-0009.xml
├── requirements.txt
├── res
│   └── pipeline.png
└── src
    ├── face_detection.py
    ├── facial_landmarks_detection.py
    ├── gaze_estimation.py
    ├── head_pose_estimation.py
    ├── input_feeder.py
    ├── main.py
    ├── model.py
    └── mouse_controller.py
```

### Python dependencies
This project depends on [openvino-toolkit](https://software.intel.com/en-us/openvino-toolkit) and some other additional libraries mentioned in `requirements.txt` file. Install the dependencies as shown below.

    $ pip install -r requirements.txt

## Demo
`How to run`:

    $ cd src/
    $ python main.py

Command Line options:
 * `cam`: Use camera as input
 * `out`: Path to output image or video file (*defatult*: `'out.mp4'`)
 * `video`: Path to input image or video file (*default*: `'../bin/demo.mp4'`)
 * `threshold`: Probability threshold for detections filtering
 * `verbose_stage`: Enable to show visualization of intermediate stages
 * `device`: option to specify the target device - *CPU/GPU/MYRIAD/FPGA (default: `CPU`)*
 * `extension`: Path to shared library extension for custom layers implementation
 * `precision`: Model precision for applicable models - *FP16/FP16-INT8/FP32 (default: `FP32`)*
 * `gaze_model`, `head_pose_model`, `face_model`, annd `landmarks_model`: options to specify the corresponding model path if the models are placed elsewhere than the project directory.

`Note`:
  * `face-detection-adas-binary-0001` model has only one (`FP32-INT1`) precision.

## Documentation
`Pipeline`: This project makes use of 4 pre-trained models provided by OpenVINO toolkit. The data flow between them is shown in the figure below.
![Pipeline](res/pipeline.png)

 * [Face Detection](https://docs.openvinotoolkit.org/latest/_models_intel_face_detection_adas_binary_0001_description_face_detection_adas_binary_0001.html)
 * [Head Position Estimation](https://docs.openvinotoolkit.org/latest/_models_intel_head_pose_estimation_adas_0001_description_head_pose_estimation_adas_0001.html)
 * [Facial Landmark Detection](https://docs.openvinotoolkit.org/latest/_models_intel_landmarks_regression_retail_0009_description_landmarks_regression_retail_0009.html)
 * [Gaze Estimation Model](https://docs.openvinotoolkit.org/latest/_models_intel_gaze_estimation_adas_0002_description_gaze_estimation_adas_0002.html)

## Benchmarks
| Model                            | Precision | Load Time    | Inference Time | Pre-process Time |
| -------------------------------- | --------- | -----------: | -------------- | ---------------- |  
| face-detection-adas-binary-0001  | FP32-INT1 | 211.75 msecs | 9.99 msecs     | 0.55 msecs       |
| gaze-estimation-adas-0002        | FP16      | 138.08 msecs | 1.52 msecs     | 0.05 msecs       |
|                                  | FP16-INT8 | 239.18 msecs | 1.41 msecs     |                  |
|                                  | FP32      | 122.49 msecs | 1.82 msecs     |                  |
| head-pose-estimation-adas-0001   | FP16      | 129.37 msecs | 1.24 msecs     | 0.06 msecs       |
|                                  | FP16-INT8 | 207.70 msecs | 1.18 msecs     |                  |
|                                  | FP32      |  96.98 msecs | 1.45 msecs     |                  |
| landmarks-regression-retail-0009 | FP16      | 108.77 msecs | 0.51 msecs     | 0.05 msecs       |
|                                  | FP16-INT8 | 132.88 msecs | 0.45 msecs     |                  |
|                                  | FP32      | 105.91 msecs | 0.51 msecs     |                  |

## Results
As we can see from the above benchmark table, there is difference in inference time for different precisions like *FP32*, *FP16* and *INT8* models. One of the main reasons could be due to quantization technic used for optimization which uses INT8 weights instead of FLOATS. This reduces the accuracy but makes the inference faster. Therefore the `FP32` takes higher inference time but more accuracy than `INT8`. So `Accuracy` & `Inference Time` is in the order `FP32 > FP16 > INT8`. But the model size is inversely proportional to the above parameters. Hence lower the precision, lighter is the model. Which might make the storing cost lesser and loading time faster during the startup.

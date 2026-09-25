# Face Detection

Real-time object and face detection with [YOLOv8](https://docs.ultralytics.com/) and OpenCV. Two models run on the same input and their results are drawn on a single image:

- **Object detection** (`yolov8n.pt`): detects what is in the picture (person, cat, etc.).
- **Face detection** (`yolov8m-face.pt`): detects and boxes the faces.

The project works on a **still image** or on a **live webcam stream**.

## Project structure

```
face_detection/
├── images/            # Sample images
├── source.py          # Detection on a single image
└── source_video.py    # Detection on the webcam (real time)
```

## Requirements

- Python 3.9+
- An NVIDIA GPU with CUDA (the scripts load the models with `.to("cuda")`)
- A webcam (for `source_video.py`)
- The `yolov8m-face.pt` weights file placed next to the scripts (`yolov8n.pt` is downloaded automatically by Ultralytics on first run)

## Installation

```bash
git clone https://github.com/rayene00/face_detection.git
cd face_detection
pip install ultralytics opencv-python
```

> To get PyTorch with CUDA support, follow the instructions on [pytorch.org](https://pytorch.org/get-started/locally/) before installing `ultralytics`.

## Usage

### On an image

Set the image path in `source.py` (default: `image2.jpg`):

```python
photo = cv2.imread('image2.jpg')
```

Then run:

```bash
python source.py
```

A window opens with the detected objects and faces drawn on the image.

### On the webcam

```bash
python source_video.py
```

The webcam stream is analysed frame by frame. Press **`x`** to quit.

## How it works

1. Both YOLO models are loaded on the GPU.
2. The input (image or webcam frame) is passed to the object model and the face model.
3. The object detections are drawn on the image with `result[0].plot()`.
4. The face detections are drawn on top of that same image with `face_result[0].plot(img=...)`.
5. The result is displayed with OpenCV (`cv2.imshow`).

## Running without a GPU

Replace `.to("cuda")` with `.to("cpu")` in both scripts. It works, but the webcam mode will be noticeably slower.

## Tech stack

- [Python](https://www.python.org/)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [OpenCV](https://opencv.org/)

## Author

Rayene Medjtoh, [@rayene00](https://github.com/rayene00)

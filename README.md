# Pose & Emotion Detector

Real-time human **pose** and **emotion detection** system using:
- **MediaPipe & custom PyTorch & Onnx models** for posture classification
- **YOLO** and **custom emotion models** for facial expression recognition

> 🚀 Built for real-time webcam input, but easily extendable to images and videos!

---

## 🛠 Features

- 📸 **Pose Detection**: Classify human body posture as `upright` or `hunchback`.
- 😄 **Emotion Recognition**: Detect facial emotions (`Happy`, `Sad`, `Angry`, `Neutral`, `Fear`, `Surprise`).
- 🔥 **Fast Inference**: Uses ONNX Runtime and quantized models for optimized performance.
- 🎯 **YOLOv11 Face Detection**: Accurate and efficient face detection.
- 🖥️ **Real-Time Webcam Feed**: Streamlined pipeline for live analysis.
- 🧩 **Modular Code**: Easily swap detectors (YOLO, Haar Cascade, ONNX) and classification models.

---

## 📂 Project Structure
**🔒 checkpoints**/ –  Saved models

**⚙️ utils/** –  Core components

    • emotion_model_api.py – API for Emotion detection via webcam
    
    • emotion_pose_model.py – Combined real-time pose + emotion detection
    
    • haar_detector.py – Haar cascade-based face detection
    
    • pose_model_api.py – API for Posture detection via webcam 
    
    • yolo_detector.py – YOLOv8-based face detection

**🏋️ train.py** –  Script to train the posture classification model

**🧠 train_emotion.py** –  Script to train the emotion classification model

**📊 evaluate.py** –  Evaluate model performance on test data

**📦 requirements.txt** –  Python dependency list

**📖 README.md** –  Project documentation (you are here!)


---

## ⚙️ Installation

1. **Clone the repository:**

```bash
git clone https://github.com/Yogesh-45/pose-emotion-detector.git
cd pose-emotion-detector
```

2. **Setup Python Environment:**

```
conda create -n pose_emotion python=3.10
conda activate pose_emotion

pip install -r requirements.txt
```

## 🚀 Usage

1. **Real-time Pose and Emotion Detection (Webcam)**
```
python utils/emotion_pose_model.py
```
This script:

Classifies pose as **Upright** or **Hunchback**.

Detects dominant Emotion.

Displays bounding boxes and labels.

Displays real-time cpu & memory usage.

2. **Run via C++ Application (Pybind11)**
You can also use C++ + Pybind11 to run Python emotion detection:
```
cd cpp
mkdir build && cd build
cmake ..
cmake --build . --config Release
.\Release\main.exe
```
✅ This shows how to embed Python models inside a native C++ application.

**📌 Requirements for C++ Integration**
To build and run the C++ integration successfully, ensure the following:

✅ CMake ≥ 3.14 is installed

✅ Visual Studio Build Tools (Windows) or g++ (Linux/macOS)

✅ Python 3.10+ installed

✅ Python development headers are available
(e.g., via conda install python-dev or choosing Python with dev headers during install)

✅ pybind11 source code is cloned inside your repo at:

```
cpp/external/pybind11
```
You can get it via:

```
git clone https://github.com/pybind/pybind11.git cpp/external/pybind11
```

✅ Your emotion_api.py is in the correct path (./utils or adjusted accordingly)

✅ You’ve added sys.path.insert(0, "<path-to-emotion_api>") in both main.cpp and bindings.cpp


## 🧠 Technologies Used

| Component           | Library / Framework                         |
|---------------------|---------------------------------------------|
| Face Detection      | YOLOv11, Haar Cascade                        |
| Pose Estimation      | MediaPipe Pose, Custom Classifier           |
| Emotion Detection   | DeepFace (initially), Custom PyTorch model  |
| Optimization         | ONNX, FP16 quantization                     |
| Real-Time Inference | OpenCV, PyTorch                             |
| C++ Integration      | pybind11, CMake                             |


## 🙌 Acknowledgments
[PyTorch](https://pytorch.org/)

[OpenCV](https://opencv.org/)

[YOLO Face](https://github.com/akanametov/yolo-face)

[ONNX Runtime](https://github.com/microsoft/onnxruntime)

[MediaPipe](https://ai.google.dev/edge/mediapipe/solutions/guide)

[Pybind11](https://github.com/pybind/pybind11)

## 📄 License
This project is licensed under the MIT License.

## 👤 Author
**Yogesh Dhyani**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/yogesh-dhyani-b0a1511ab/)

⭐ If you find this project helpful, give it a star and share it!





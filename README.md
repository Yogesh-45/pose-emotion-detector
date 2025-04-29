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
- 🎯 **YOLOv8 Face Detection**: Accurate and efficient face detection.
- 🖥️ **Real-Time Webcam Feed**: Streamlined pipeline for live analysis.
- 🧩 **Modular Code**: Easily swap detectors (YOLO, Haar Cascade, ONNX) and classification models.

---

## 📂 Project Structure


import cv2
import time
import os
from deepface import DeepFace
from ultralytics import YOLO
import pose_model_api as pma
import emotion_model_api as ema
import yolo_detector as yd
import haar_detector as hd
import argparse
import psutil
from collections import deque
import onnxruntime as ort

parser= argparse.ArgumentParser(description="emotion detection")
parser.add_argument('--detector', type= str, default='YOLO', help='detector model to be used')
parser.add_argument('--pmt', type= str, default='pt', help='posture model type')
parser.add_argument('--pp', type= str, default='fp32', help='posture precision')
parser.add_argument('--emotion_model', type= bool, default=True, help='emotion model is used or not')
parser.add_argument('--input_type', type=str, choices=['webcam', 'image'], default='webcam', help='Input type')
parser.add_argument('--image_path', type=str, default='./test_data/happy_face.jpg', help='Path to input image')
args= parser.parse_args()


# --------------------------- #
# Setup
# --------------------------- #
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

# Initialize Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# yolo detector
detector= YOLO("yolov11m-face.pt")

# --------------------------- #
# Posture Model
# --------------------------- #
posture_model= pma.my_model()

# Posture model onnx
if args.pp== 'fp16':
    onnx_model_path = "checkpoints/best_model_checkpoint_onnx_fp16.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])
elif args.pp== 'fp32':
    onnx_model_path = "checkpoints/best_model_checkpoint_onnx.onnx"
    onnx_session = ort.InferenceSession(onnx_model_path, providers=["CPUExecutionProvider"])

# --------------------------- #
# Emotion Model
# --------------------------- #
emotion_model= ema.my_model()



# --------------------------- #
# Real-Time Loop
# --------------------------- #

def main():
    prev_time = 0
    fps_history = deque(maxlen=30)  # to average over the last 30 frames
    frame_count = 0
    skip_frames = 2  # Tune this

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        emotion = "Unknown"
        posture = "Unknown"


        
        try:       

            emotion = "Unknown"
            posture = "Unknown"
            # Analyze frame for emotion
            if args.detector == 'haar':
                face= hd.detector_haar(frame, face_cascade)
            elif args.detector == 'YOLO':
                if frame_count % skip_frames == 0:
                    face= yd.detector_YOLO(frame, detector)
            else:
                face= frame
                enforce_detection= True

            if face is not None:
                if args.emotion_model == True:
                    emotion= ema.infer_emotion(face, emotion_model)
                else:
                    result = DeepFace.analyze(face, actions=['emotion'], enforce_detection= enforce_detection)
                    emotion = result[0]['dominant_emotion']

                # Posture prediction
                if args.pmt == 'onnx':
                    if args.pp == 'fp16':
                        posture= pma.predict_posture_onnx(frame, onnx_session, args.pp)
                    else:
                        posture= pma.predict_posture_onnx(frame, onnx_session)
                else:
                    posture = pma.predict_posture(frame, posture_model)

            # Calculate FPS
            curr_time = time.time()
            fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
            prev_time = curr_time

            # Update FPS history and calculate average FPS
            fps_history.append(fps)
            avg_fps = sum(fps_history) / len(fps_history)

            # Get system usage
            cpu_usage = psutil.cpu_percent(interval=None)  # non-blocking
            mem = psutil.virtual_memory()
            memory_usage = mem.percent

            # Display on frame
            cv2.putText(frame, f"FPS: {int(avg_fps)}", (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"CPU: {cpu_usage:.1f}%", (30, 680), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Memory: {memory_usage:.1f}%", (30, 710), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            
            # # 5. Display info
            # cv2.putText(frame, f"FPS: {int(fps)}", (30, 650), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion}", (30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            color = (0, 255, 0) if posture == "upright" else (0, 0, 255)
            cv2.putText(frame, f"Posture: {posture}", (30, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            
        except Exception as e:
            print("Error:", e)

        frame_count += 1
        cv2.imshow("Real-Time Emotion + Posture Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
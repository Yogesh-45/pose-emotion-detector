import cv2
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from ultralytics import YOLO

# Load Model
model = models.mobilenet_v3_small()
model.classifier[3] = nn.Linear(1024, 6)
model.load_state_dict(torch.load('./checkpoints/best_model_emotion_6cls_balanced.pth', map_location='cpu'))
model.eval()

# Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# YOLO Detector
detector = YOLO('./yolov11m-face.pt')

def detector_YOLO(image):
    resized = cv2.resize(image, (640, 640))
    h_org, w_org = image.shape[:2]

    results = detector(resized)[0]

    for box in results.boxes:
        conf = float(box.conf[0])
        if conf < 0.5:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        scale_x = w_org / 640
        scale_y = h_org / 640
        x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
        y1, y2 = int(y1 * scale_y), int(y2 * scale_y)

        face = image[y1:y2, x1:x2]
        if face.size != 0:
            return face
    return None

def detect_emotion(image_path):
    print(">> detect_emotion called with", image_path)  # 👈 Debug line
    frame = cv2.imread(image_path)
    if frame is None:
        return "Invalid Image Path"
    
    face = detector_YOLO(frame)
    if face is None:
        return "No face detected"

    face_pil = Image.fromarray(face)
    input_tensor = transform(face_pil).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).item()
        return EMOTIONS[pred]

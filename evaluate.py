import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from torchvision import transforms
from tqdm import tqdm
import utils.pose_model_api as pma  # your model loading + predict_posture method

# ---------------------------------------- #
# Configuration
# ---------------------------------------- #
DATA_DIR = './posture_dataset/val'
CLASSES = ['hunchback', 'upright']
IMG_SIZE = 224  

# Transform (adjust mean/std if different during training)
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

# ---------------------------------------- #
# Load Model
# ---------------------------------------- #
model = pma.my_model()
model.eval()

# ---------------------------------------- #
# Evaluation Loop
# ---------------------------------------- #
y_true = []
y_pred = []

for class_idx, class_name in enumerate(CLASSES):
    class_dir = os.path.join(DATA_DIR, class_name)
    if not os.path.exists(class_dir):
        continue
    for fname in tqdm(os.listdir(class_dir), desc=f"Processing {class_name}"):
        fpath = os.path.join(class_dir, fname)
        img = cv2.imread(fpath)
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(img_rgb).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()

        y_true.append(class_idx)
        y_pred.append(pred)

# ---------------------------------------- #
# Report & Confusion Matrix
# ---------------------------------------- #
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASSES))

cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)

fig, ax = plt.subplots(figsize=(6, 5))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix: Posture Classification")
plt.tight_layout()
plt.show()
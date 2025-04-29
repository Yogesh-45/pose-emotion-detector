from torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np

class my_model(nn.Module):
    def __init__(self, checkpoint_path= 'checkpoints/best_model_checkpoint.pth'):
        super(my_model, self).__init__()  # <- Call parent constructor first
        self.model= models.mobilenet_v3_small(weight=False)
        self.model.classifier[3]= nn.Linear(1024,2)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),             # Resize to a standard size
    transforms.CenterCrop(224),                # Deterministic crop (no randomness)
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)              # Same normalization as training
])


def predict_posture(frame, posture_model):
    # Crop upper body (top 60% of the frame)
    h, w, _ = frame.shape
    upper_body = frame
    pil_img = Image.fromarray(cv2.cvtColor(upper_body, cv2.COLOR_BGR2RGB))
    input_tensor = transform(pil_img).unsqueeze(0)
    with torch.no_grad():
        output = posture_model(input_tensor)
        _, pred = torch.max(output, 1)
    return "hunchback" if pred.item() == 0 else "upright"


def preprocess_image(image, precision):
    pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if precision == 'fp16':
        input_tensor = transform(pil_img).unsqueeze(0).numpy().astype(np.float16)
    elif precision == 'fp32':
        input_tensor = transform(pil_img).unsqueeze(0).numpy()
    return input_tensor

def predict_posture_onnx(frame, onnx_session, precision= 'fp32'):
    h, w, _ = frame.shape
    # upper_body = frame[:int(h * 0.6), :]
    upper_body = frame
    input_tensor = preprocess_image(upper_body, precision)

    inputs = {onnx_session.get_inputs()[0].name: input_tensor}
    outputs = onnx_session.run(None, inputs)
    pred = np.argmax(outputs[0], axis=1)[0]

    return "hunchback" if pred == 0 else "upright"
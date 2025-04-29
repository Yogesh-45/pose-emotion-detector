from torchvision import models
import torch.nn as nn
import torch
from torchvision import transforms
from PIL import Image

class my_model(nn.Module):
    def __init__(self, checkpoint_path= 'checkpoints/best_model_emotion_6cls_balanced.pth'):
        super(my_model, self).__init__()  # <- Call parent constructor first
        self.model= models.mobilenet_v3_small(weight=False)
        self.model.classifier[3]= nn.Linear(1024,6)
        self.model.load_state_dict(torch.load(checkpoint_path, map_location='cpu'))
        self.model.eval()

    def forward(self, x):
        return self.model(x)

# ------------------------- #
# Transforms and Labels
# ------------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
EMOTIONS = ['Angry', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# ------------------------- #
# Emotion Inference Function
# ------------------------- #
def infer_emotion(face, model):
    # face = detector.predict(frame)
    if face is not None:
        face_pil = Image.fromarray(face)
        input_tensor = transform(face_pil).unsqueeze(0)

        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, dim=1).item()
            emotion = EMOTIONS[pred]
        return emotion
    return "No face detected"
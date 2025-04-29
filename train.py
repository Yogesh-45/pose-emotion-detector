import torch
import torch.nn as nn
from torch import optim
from torchvision import models
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

device= 'cuda' if torch.cuda.is_available() else 'cpu'
in_features=1024
out_features= 2

model= models.mobilenet_v3_small(weights=True)
model.classifier[3]= nn.Linear(in_features, out_features) 
model= model.to(device)
# print(model)

optimizer= optim.AdamW(model.parameters(), lr= 0.001)
criterion= nn.CrossEntropyLoss()

batch_size= 32
epochs= 50
input_dim= (3, 224, 224)
best_val_acc = 0.0  # Initialize best accuracy


transform= transforms.Compose([
    transforms.Resize((256,256)),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(0.5,0.5)

])

#train dir
train_dir= './posture_dataset/train'
val_dir  = './posture_dataset/val'

#create datasets
train_dataset= datasets.ImageFolder(train_dir, transform)
val_dataset  = datasets.ImageFolder(val_dir, transform)

#create dataloaders
train_loader= DataLoader(train_dataset, batch_size, shuffle= True, drop_last= True)
val_loader=   DataLoader(val_dataset, batch_size)

for epoch in range(epochs):
    print(f"\nEpoch {epoch+1}/{epochs}")
    print("-" * 30)

    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    

    for inputs, labels in tqdm(train_loader, desc= "Training"):
        inputs= inputs.to(device)
        labels= labels.to(device)
        outputs= model(inputs)

        loss= criterion(outputs, labels)

        #backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        #track loss and accuracy
        running_loss += loss.item() * inputs.size(0)
        _, predicted  = torch.max(outputs,1)
        total   += inputs.size(0)
        correct += (predicted==labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = 100 * correct / total
    print(f"Train Loss: {epoch_loss:.4f} | Train Acc: {epoch_acc:.2f}%")

    # ======== Validation ========
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Validating"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()


    val_epoch_loss = val_loss / val_total
    val_epoch_acc = 100 * val_correct / val_total
    print(f"Val Loss: {val_epoch_loss:.4f} | Val Acc: {val_epoch_acc:.2f}%")

    # -------- Save best checkpoint --------
    if val_epoch_acc > best_val_acc:
        best_val_acc = val_epoch_acc
        torch.save(model.state_dict(), './checkpoints/best_model_checkpoint.pth')
        print(f"✅ Saved new best model with Val Acc: {best_val_acc:.2f}%")


torch.save(model.state_dict(), './checkpoints/model_1.pth')



import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
from PIL import Image
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog

# ---------------- CONFIG ----------------
DATASET_PATH = r"D:\hackathon\script_squad\data"  # <-- your dataset folder
IMG_SIZE = 128
BATCH_SIZE = 8
EPOCHS = 15
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DISASTER_THRESHOLD = 0.8  # threshold for disaster detection

# ---------------------------------------
# SAFE IMAGE LOADER
# ---------------------------------------
class SafeImageFolder(ImageFolder):
    def __getitem__(self, index):
        path, target = self.samples[index]
        try:
            img = Image.open(path).convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, target
        except:
            return self.__getitem__((index + 1) % len(self.samples))

# ---------------------------------------
# CNN MODEL
# ---------------------------------------
class DisasterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

    def forward(self, x):
        return self.net(x)

# ---------------------------------------
# DATA TRANSFORMS
# ---------------------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor()
])

# ---------------------------------------
# LOAD DATASET & SPLIT
# ---------------------------------------

dataset = SafeImageFolder(DATASET_PATH, transform=transform)
print("Classes:", dataset.classes)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# ---------------------------------------
# MODEL, LOSS, OPTIMIZER
# ---------------------------------------
model = DisasterCNN().to(DEVICE)

# Optional: class weights if dataset is imbalanced
class_counts = [sum(1 for _, t in train_dataset if t == i) for i in range(len(dataset.classes))]
weights = 1 / torch.tensor(class_counts, dtype=torch.float)
criterion = nn.CrossEntropyLoss(weight=weights.to(DEVICE))

optimizer = optim.Adam(model.parameters(), lr=0.001)

# ---------------------------------------
# TRAINING LOOP
# ---------------------------------------
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for imgs, labels in train_loader:
        imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in val_loader:
            imgs, labels = imgs.to(DEVICE), labels.to(DEVICE)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    val_acc = correct / total
    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {total_loss:.2f} | Val Acc: {val_acc:.2f}")

print("Training complete.\n")

# ---------------------------------------
# LOAD YOLO
# ---------------------------------------
yolo = YOLO("yolov8n.pt")

# ---------------------------------------
# IMAGE / FRAME ANALYSIS
# ---------------------------------------
def analyze_image(frame):
    # Convert BGR -> RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = torch.tensor(img).permute(2,0,1).float().unsqueeze(0) / 255.0
    img = img.to(DEVICE)

    with torch.no_grad():
        out = model(img)
        probs = torch.softmax(out, dim=1)[0]

    disaster_idx = dataset.class_to_idx['disaster']
    risk = probs[disaster_idx].item()
    label = "DISASTER" if risk > DISASTER_THRESHOLD else "NORMAL"

    # YOLO people count
    results = yolo(frame, conf=0.4, verbose=False)
    people = sum(int(c)==0 for r in results for c in r.boxes.cls)

    return label, risk, people

# ---------------------------------------
# CAMERA MODE
# ---------------------------------------
def open_camera(emergency=False):
    cap = cv2.VideoCapture(0)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        label, risk, people = analyze_image(frame)
        cv2.putText(frame, f"{label} | Risk: {risk:.2f} | People: {people}",
                    (20,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
        cv2.imshow("HAVEN Live Detection", frame)
        if emergency or risk > DISASTER_THRESHOLD:
            print("\n🚨 EMERGENCY ALERT 🚨")
            print("Disaster:", label)
            print("Risk Level:", round(risk,2))
            print("People Detected:", people)
            print("----------------------")
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

# ---------------------------------------
# TKINTER UI
# ---------------------------------------
def upload_image():
    path = filedialog.askopenfilename()
    if not path:
        return
    img = cv2.imread(path)
    label, risk, people = analyze_image(img)
    print("\n📷 IMAGE ANALYSIS")
    print("Disaster:", label)
    print("Risk Level:", round(risk,2))
    print("People Detected:", people)
    print("----------------------")

def emergency_button():
    open_camera(emergency=True)

root = tk.Tk()
root.title("HAVEN - Disaster Detection")
root.geometry("500x300")
tk.Button(root, text="Upload Image", width=30, command=upload_image).pack(pady=10)
tk.Button(root, text="Live Camera Detection", width=30, command=open_camera).pack(pady=10)
tk.Button(root, text="🚨 Emergency", width=30, bg="red", fg="white",
          command=emergency_button).pack(pady=10)
root.mainloop()

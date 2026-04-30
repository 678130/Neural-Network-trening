# Mel spektrum (oppløsning 128) som input til CNN
# Dataset NSynth-mel (preprosessert versjon av NSynth)
# 24.06.2026
# Trent i kaggle
# Kan ikkje kjøre lokalt enda

import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import time
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

from mel_dataset import MelDataset
from cnn import CNN

BASE = "/kaggle/input/datasets/sondreh/nsynth-mel/nsynth-train-mel"

TRAIN_MEL = os.path.join(BASE, "train", "mel_train")
VALID_MEL = os.path.join(BASE, "test", "mel_test")

TRAIN_LABELS = os.path.join(BASE, "train", "examples-train-original.json")
VALID_LABELS = os.path.join(BASE, "test", "examples-test-original.json")

# Sjekk at filene finnes og sammenlign antall labels med antall mel-filer

with open(TRAIN_LABELS) as f:
    labels = json.load(f)

print("Total labels:", len(labels))

for i, (k, v) in enumerate(labels.items()):
    print(k, "->", v)
    if i >= 4:
        break

MEL_DIR = os.path.dirname(TRAIN_LABELS)

missing = []
for k in list(labels.keys())[:1000]:
    path = os.path.join(TRAIN_MEL, f"{k}.npy")
    if not os.path.exists(path):
        missing.append(k)

print("Missing files (sample check):", len(missing))

with open(TRAIN_LABELS) as f:
    metadata = json.load(f)
    
all_pitches = sorted(set(v["pitch"] for v in metadata.values()))
pitch_to_class = {p: i for i, p in enumerate(all_pitches)}

import json

with open("pitch_to_class.json", "w") as f:
    json.dump(pitch_to_class, f)

MelDataset(TRAIN_MEL, TRAIN_LABELS, pitch_to_class, augment=False)

train_dataset = MelDataset(
    mel_dir=TRAIN_MEL,
    labels_path=TRAIN_LABELS,
    pitch_to_class=pitch_to_class,
    augment=True
)

valid_dataset = MelDataset(
    mel_dir=VALID_MEL,
    labels_path=VALID_LABELS,
    pitch_to_class=pitch_to_class,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

model = CNN(len(pitch_to_class)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

history = {
    "train_loss": [],
    "val_loss": [],
    "train_acc": [],
    "val_acc": [],
    "lr" : []

}

top3_correct = 0

EPOCHS = 20

for epoch in range(EPOCHS):
    start_time = time.time()

    # ---- TRAIN ----
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
    
        outputs = model(x)
        loss = criterion(outputs, y)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
        train_loss += loss.item() * x.size(0)
    
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    train_loss /= total
    train_acc = correct / total

    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    
    top3_correct = 0
    total_error = 0
    within_1 = 0
    within_2 = 0
    
    with torch.no_grad():
        for x, y in valid_loader:
            x, y = x.to(device), y.to(device)
    
            outputs = model(x)
            loss = criterion(outputs, y)
    
            val_loss += loss.item() * x.size(0)
    
            preds = outputs.argmax(dim=1)
    
            correct += (preds == y).sum().item()
            total += y.size(0)
    
            # Top-3
            top3 = outputs.topk(3, dim=1).indices
            top3_correct += (top3 == y.unsqueeze(1)).any(dim=1).sum().item()
    
            total_error += (preds - y).abs().sum().item()
            within_1 += ((preds - y).abs() <= 1).sum().item()
            within_2 += ((preds - y).abs() <= 2).sum().item()
    
    val_loss /= total
    val_acc = correct / total
    top3_acc = top3_correct / total
    mean_error = total_error / total
    acc_1 = within_1 / total
    acc_2 = within_2 / total

    epoch_time = time.time() - start_time

    scheduler.step(val_loss)
    current_lr = optimizer.param_groups[0]['lr']
    print(f"LR: {current_lr:.6f}")

    history["train_loss"].append(train_loss)
    history["val_loss"].append(val_loss)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)
    history["lr"].append(current_lr)

    print(f"Epoch {epoch+1:02d} | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.3f} | "
          f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.3f} | "
          f"Time: {epoch_time:.1f}s  | ")

    print(f"Top1: {val_acc:.3f} | Top3: {top3_acc:.3f} | "
          f"Err: {mean_error:.2f} | "
          f"±1: {acc_1:.3f} | ±2: {acc_2:.3f}")

torch.save(model.state_dict(), "/kaggle/working/note_model.pth")

plt.plot(history["train_loss"], label="Train Loss")
plt.plot(history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss Curve")
plt.show()

all_preds = []
all_labels = []

model.eval()

with torch.no_grad():
    for x, y in valid_loader:
        x = x.to(device)
        y = y.to(device)

        outputs = model(x)
        predicted = outputs.argmax(dim=1)

        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y.cpu().numpy())

cm = confusion_matrix(all_labels, all_preds)
class_names = [str(p) for p in sorted(pitch_to_class.keys())]

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
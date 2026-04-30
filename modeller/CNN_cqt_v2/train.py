import os
import json
import torch
import librosa
import numpy as np
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from modeller.CNN_cqt_v2.cnn import CNN
from modeller.CNN_cqt_v2.cqt_dataset import CQTDataset

BASE = "/kaggle/input/datasets/sondreh/nsynth-cqt/Nsynth_cqt"

TRAIN_CQT = os.path.join(BASE, "cqt_train")
VALID_CQT = os.path.join(BASE, "cqt_valid")

TRAIN_LABELS = os.path.join(TRAIN_CQT, "labels.json")
VALID_LABELS = os.path.join(VALID_CQT, "labels.json")

import json
import os


with open(TRAIN_LABELS) as f:
    labels = json.load(f)

print("Total labels:", len(labels))

# print first 5 entries
for i, (k, v) in enumerate(labels.items()):
    print(k, "->", v)
    if i >= 4:
        break

# check that files actually exist
DIR = os.path.dirname(TRAIN_LABELS)

missing = []
for k in list(labels.keys())[:1000]:  # check subset (fast)
    path = os.path.join(DIR, f"{k}.npy")
    if not os.path.exists(path):
        missing.append(k)

print("Missing files (sample check):", len(missing))

import os
import numpy as np

files = [f for f in os.listdir(DIR) if f.endswith(".npy")]

# load a few samples
for f in files[:5]:
    cqt = np.load(os.path.join(DIR, f))
    print(f, "-> shape:", cqt.shape)


with open(TRAIN_LABELS) as f:
    metadata = json.load(f)
    
all_pitches = sorted(set(metadata.values()))
pitch_to_class = {p: i for i, p in enumerate(all_pitches)}

with open("pitch_to_class.json", "w") as f:
    json.dump(pitch_to_class, f)

train_dataset = CQTDataset(
    cqt_dir=TRAIN_CQT,
    labels_path=TRAIN_LABELS,
    pitch_to_class=pitch_to_class,
    augment=True
)

valid_dataset = CQTDataset(
    cqt_dir=VALID_CQT,
    labels_path=VALID_LABELS,
    pitch_to_class=pitch_to_class,
    augment=False
)

train_loader = DataLoader(
    train_dataset,
    batch_size=64,          # increase now that it's faster
    shuffle=True,
    num_workers=4,          # increase for Kaggle CPU
    pin_memory=True
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

x, y = next(iter(train_loader))

print("x shape:", x.shape)   # should be [B, 1, 128, W]
print("y shape:", y.shape)
print("label example:", y[:10])

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN(len(pitch_to_class)).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

print(device)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.5,
    patience=2
)

import time

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
    
        # accumulate properly
        train_loss += loss.item() * x.size(0)
    
        preds = outputs.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    
    # normalize properly
    train_loss /= total
    train_acc = correct / total

    # ---- VALIDATION ----
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
    
            # Distance metrics
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

# Save model
torch.save(model.state_dict(), "/kaggle/working/note_model.pth")

import torch
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

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

# correct class labels
class_names = [str(p) for p in sorted(pitch_to_class.keys())]

plt.figure(figsize=(12,10))
sns.heatmap(cm, cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.show()
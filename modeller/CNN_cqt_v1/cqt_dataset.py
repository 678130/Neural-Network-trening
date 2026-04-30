import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class CQTDataset(Dataset):
    def __init__(self, cqt_dir, labels_path, pitch_to_class, augment=False):
        self.cqt_dir = cqt_dir
        self.augment = augment
        self.pitch_to_class = pitch_to_class

        with open(labels_path) as f:
            labels = json.load(f)

        # match mel structure
        self.items = []
        for k, v in labels.items():
            pitch = v  # your CQT labels are already just pitch
            self.items.append((k, pitch))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_id, pitch = self.items[idx]

        cqt = np.load(os.path.join(self.cqt_dir, f"{file_id}.npy"))

        # ---- SAME RANDOM WINDOW AS MEL ----
        target_frames = 32

        T = cqt.shape[1]

        if T > target_frames:
            start = np.random.randint(0, T - target_frames)
            cqt = cqt[:, start:start+target_frames]
        else:
            pad_width = target_frames - T
            cqt = np.pad(cqt, ((0, 0), (0, pad_width)))

        # -----------------------------------

        # ---- SAME AUGMENTATION ----
        if self.augment:
            cqt += np.random.normal(0, 0.02, cqt.shape)

        # ---- SAME NORMALIZATION ----
        cqt = (cqt - cqt.mean()) / (cqt.std() + 1e-6)

        # ---- TO TENSOR ----
        cqt = torch.tensor(cqt).unsqueeze(0).float()

        label = self.pitch_to_class[pitch]

        return cqt, label
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

class MelDataset(Dataset):
    def __init__(self, mel_dir, labels_path, pitch_to_class, augment=False):
        self.mel_dir = mel_dir
        self.augment = augment
        self.pitch_to_class = pitch_to_class

        with open(labels_path) as f:
            labels = json.load(f)

        self.items = []

        for k, v in labels.items():
            pitch = v["pitch"]
            self.items.append((k, pitch))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_id, pitch = self.items[idx]
    
        mel = np.load(os.path.join(self.mel_dir, f"{file_id}.npy"))
    
        target_frames = 32
    
        T = mel.shape[1]
    
        if T > target_frames:
            start = np.random.randint(0, T - target_frames)
            mel = mel[:, start:start+target_frames]
    
        if self.augment:
            mel += np.random.normal(0, 0.02, mel.shape)
        
        mel = (mel - mel.mean()) / (mel.std() + 1e-6)
        
        mel = torch.tensor(mel).unsqueeze(0).float()
        label = self.pitch_to_class[pitch]
    
        return mel, label
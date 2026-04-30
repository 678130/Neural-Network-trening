from scipy.ndimage import gaussian_filter1d
import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset

SR = 16000
HOP_LENGTH = 512

class CQTDataset(Dataset):
    def __init__(self, cqt_dir, labels_path, pitch_to_class, sr, hop_length, augment=False):
        self.cqt_dir = cqt_dir
        self.augment = augment
        self.pitch_to_class = pitch_to_class
        self.target_frames = 48

        self.sr = sr
        self.hop_length = hop_length

        with open(labels_path) as f:
            labels = json.load(f)

        self.items = list(labels.items())

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        file_id, pitch = self.items[idx]

        cqt = np.load(os.path.join(self.cqt_dir, f"{file_id}.npy"))
        T = cqt.shape[1]

        # ---- SLICE SELECTION ----
        if T > self.target_frames:

            energy = np.max(cqt, axis=0)
            energy_smooth = gaussian_filter1d(energy, sigma=2)

            # avoid early attack region (~0.2s)
            frames_per_sec = SR / HOP_LENGTH
            min_offset = int(0.2 * frames_per_sec)

            start_range = max(0, min_offset)
            end_range = T - self.target_frames

            if np.random.rand() < 0.7:
                # fast rolling mean
                window = np.ones(self.target_frames) / self.target_frames
                scores = np.convolve(energy_smooth, window, mode="valid")

                start = np.argmax(scores[start_range:end_range]) + start_range
            else:
                start = np.random.randint(0, T - self.target_frames)

            chunk = cqt[:, start:start+self.target_frames]

        else:
            pad_width = self.target_frames - T
            chunk = np.pad(cqt, ((0, 0), (0, pad_width)))

        # ---- AUGMENTATION (on chunk!) ----
        if self.augment:
            chunk = chunk + np.random.normal(0, 0.02, chunk.shape)

        # ---- NORMALIZATION (on chunk!) ----
        chunk = (chunk - chunk.mean()) / (chunk.std() + 1e-6)

        # ---- TO TENSOR ----
        chunk = torch.tensor(chunk).unsqueeze(0).float()

        label = self.pitch_to_class[pitch]

        return chunk, label
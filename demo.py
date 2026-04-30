import sys
import numpy as np
import librosa
import torch
import json

from PyQt5.QtWidgets import (
    QApplication, QWidget, QPushButton, QLabel, QVBoxLayout, QFileDialog
)
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtCore import QUrl, QTimer

# ===== LOAD MODEL =====
from modeller.CNN_cqt_v4.cnn import BetterCNN

MODEL_PATH = "modeller/CNN_cqt_v4/resource/note_model.pth"

with open("modeller/CNN_cqt_v4/resource/pitch_to_class.json") as f:
    pitch_to_class = json.load(f)

class_to_pitch = {v: k for k, v in pitch_to_class.items()}

model = BetterCNN(num_classes=len(pitch_to_class))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()


def midi_to_note(midi):
    midi = int(midi)
    names = ['C', 'C#', 'D', 'D#', 'E', 'F',
             'F#', 'G', 'G#', 'A', 'A#', 'B']
    return f"{names[midi % 12]}{(midi // 12)}"

# ===== AUDIO SETTINGS =====
SR = 16000
HOP_LENGTH = 512
WINDOW_SEC = 1.5

class App(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Real-time Pitch Detection")

        layout = QVBoxLayout()

        self.btn = QPushButton("Load MP3")
        self.btn.clicked.connect(self.load_file)

        self.label = QLabel("Note: -")
        self.label.setStyleSheet("font-size: 28px;")

        layout.addWidget(self.btn)
        layout.addWidget(self.label)
        self.setLayout(layout)

        self.player = QMediaPlayer()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_prediction)

        self.audio = None

        # 🔥 manual time tracking
        self.current_time = 0.0
        self.step_time = 0.25  # 250 ms

        # smoothing
        self.history = []

    def load_file(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select MP3", "", "Audio Files (*.mp3)")

        if file:
            self.audio, _ = librosa.load(file, sr=SR)

            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file)))
            self.player.play()

            self.current_time = 0.0
            self.history = []

            self.timer.start(int(self.step_time * 1000))

    def update_prediction(self):
        if self.audio is None:
            return

        pos_sample = int(self.current_time * SR)
        window_size = int(WINDOW_SEC * SR)

        # stop when audio ends
        if pos_sample >= len(self.audio):
            self.timer.stop()
            return

        # extract segment safely
        end = pos_sample + window_size
        if end >= len(self.audio):
            segment = self.audio[pos_sample:]
            pad = window_size - len(segment)
            segment = np.pad(segment, (0, pad))
        else:
            segment = self.audio[pos_sample:end]

        # advance time
        self.current_time += self.step_time

        # ===== CQT =====
        cqt = librosa.cqt(
            segment,
            sr=SR,
            hop_length=HOP_LENGTH,
            fmin=librosa.note_to_hz("C2"),
            n_bins=72,
            bins_per_octave=12
        )

        cqt = np.abs(cqt)
        cqt = librosa.amplitude_to_db(cqt, ref=np.max)

        # center crop to 48 frames (better than taking start)
        if cqt.shape[1] > 48:
            mid = cqt.shape[1] // 2
            start = max(0, mid - 24)
            cqt = cqt[:, start:start+48]
        else:
            pad = 48 - cqt.shape[1]
            cqt = np.pad(cqt, ((0, 0), (0, pad)))

        # normalize
        cqt = (cqt - cqt.mean()) / (cqt.std() + 1e-6)

        x = torch.tensor(cqt).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            out = model(x)
            pred = torch.argmax(out, dim=1).item()

        # smoothing (majority vote)
        self.history.append(pred)
        if len(self.history) > 5:
            self.history.pop(0)

        pred = max(set(self.history), key=self.history.count)

        midi = class_to_pitch[pred]
        note = midi_to_note(midi)

        self.label.setText(f"Note: {note}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = App()
    window.show()
    sys.exit(app.exec_())
import torch
import os
from CNN_cqt_v2.cnn import CNN
from torch import onnx

from CNN_cqt_v4.cnn import BetterCNN

# FOR VISALISERING AV MODELLSTRUKTUR
# KJØR SCRIPT FOR Å GENERERE "model.onnx" FILEN
# FILEN SKAL DERETTER LASTES OPP TIL https://netron.app/ FOR Å SE MODELLSTRUKTUREN   

BASE_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_PATH, "CNN_cqt_v4", "resource", "note_model.pth") #bytt CNN med den faktiske mappen der modellen ligger

model = BetterCNN(112)
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

x = torch.randn(1, 1, 72, 48)  # Bare simulere input (batch, channel, freq, time)
y = model(x)

onnx.export(model, x, "model.onnx")
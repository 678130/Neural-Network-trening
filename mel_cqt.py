import matplotlib.pyplot as plt

# Epochs
epochs_cqt = list(range(1, 21))
epochs_mel = list(range(1, 21))

# --- CQT DATA ---
cqt_train_acc = [
    0.446, 0.641, 0.676, 0.693, 0.705, 0.714, 0.720, 0.725, 0.730, 0.733,
    0.736, 0.740, 0.748, 0.750, 0.752, 0.752, 0.755, 0.756, 0.756, 0.761
]

cqt_val_acc = [
    0.636, 0.692, 0.703, 0.727, 0.731, 0.738, 0.744, 0.742, 0.754, 0.753,
    0.752, 0.740, 0.764, 0.764, 0.764, 0.760, 0.759, 0.765, 0.750, 0.766
]

cqt_val_loss = [
    2.0707, 1.8888, 1.8222, 1.7584, 1.7426, 1.7157, 1.7043, 1.7068, 1.6845, 1.6901,
    1.6849, 1.7039, 1.6449, 1.6524, 1.6459, 1.6385, 1.6507, 1.6450, 1.6655, 1.6236
]

# --- MEL DATA ---
mel_train_acc = [
    0.716, 0.722, 0.724, 0.727, 0.738, 0.741, 0.744, 0.744, 0.746, 0.747,
    0.748, 0.756, 0.756, 0.757, 0.761, 0.762, 0.763, 0.765, 0.765, 0.766
]

mel_val_acc = [
    0.723, 0.717, 0.723, 0.727, 0.738, 0.729, 0.732, 0.737, 0.731, 0.740,
    0.742, 0.737, 0.741, 0.744, 0.742, 0.736, 0.742, 0.749, 0.746, 0.746
]

mel_val_loss = [
    1.1595, 1.1830, 1.1929, 1.1924, 1.1407, 1.1701, 1.1392, 1.1344, 1.1617, 1.1675,
    1.1482, 1.1529, 1.1888, 1.1529, 1.1609, 1.1787, 1.1596, 1.1165, 1.1516, 1.1390
]

# -------------------------
# 1. VAL ACCURACY (HOVEDPLOT)
# -------------------------
plt.figure()
plt.plot(epochs_cqt, cqt_val_acc, label="CQT Val Acc")
plt.plot(epochs_mel, mel_val_acc, label="Mel Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Validation Accuracy: CQT vs Mel")
plt.legend()
plt.grid()
plt.savefig("val_accuracy.png")
plt.show()

# -------------------------
# 2. VAL LOSS
# -------------------------
plt.figure()
plt.plot(epochs_cqt, cqt_val_loss, label="CQT Val Loss")
plt.plot(epochs_mel, mel_val_loss, label="Mel Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Validation Loss: CQT vs Mel")
plt.legend()
plt.grid()
plt.savefig("val_loss.png")
plt.show()

# -------------------------
# 3. TRAIN VS VAL (CQT)
# -------------------------
plt.figure()
plt.plot(epochs_cqt, cqt_train_acc, label="Train Acc")
plt.plot(epochs_cqt, cqt_val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CQT: Train vs Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("cqt_train_vs_val.png")
plt.show()

# -------------------------
# 4. TRAIN VS VAL (MEL)
# -------------------------
plt.figure()
plt.plot(epochs_mel, mel_train_acc, label="Train Acc")
plt.plot(epochs_mel, mel_val_acc, label="Val Acc")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("Mel: Train vs Validation Accuracy")
plt.legend()
plt.grid()
plt.savefig("mel_train_vs_val.png")
plt.show()
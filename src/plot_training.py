import pickle
import matplotlib.pyplot as plt

# Load training history
with open("training_history.pkl", "rb") as f:
    history = pickle.load(f)

# Correct key names
train_losses = history['train_loss']
val_losses = history['val_loss']
train_accuracies = history['train_acc']
val_accuracies = history['val_acc']

epochs = range(1, len(train_losses) + 1)

plt.figure(figsize=(12, 5))

# Plot Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, 'b-', label='Train Loss')
plt.plot(epochs, val_losses, 'r-', label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Loss Curve')
plt.legend()

# Plot Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, 'b-', label='Train Accuracy')
plt.plot(epochs, val_accuracies, 'r-', label='Val Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Accuracy Curve')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")
print("[INFO] Training curves saved as training_curves.png")

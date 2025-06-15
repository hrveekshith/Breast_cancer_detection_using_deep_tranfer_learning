# src/plot_confusion_matrix.py

import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from src.model import DualBranchCNN
from src.dataset import get_dataloaders

# =========================
# 1️⃣ Set device
# =========================
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# =========================
# 2️⃣ Load model
# =========================
model = DualBranchCNN()   # No num_classes needed here
checkpoint_path = 'dual_branch_cnn.pth'  # Ensure you have the correct model checkpoint
state_dict = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(state_dict)
model = model.to(device)
model.eval()

# =========================
# 3️⃣ Load val_loader
# =========================
_, val_loader = get_dataloaders(batch_size=32)

# =========================
# 4️⃣ Get predictions function
# =========================
def get_predictions(model, dataloader, device):
    y_true = []
    y_pred = []
    with torch.no_grad():
        for (histo, thermal), labels in dataloader:
            histo = histo.to(device)
            thermal = thermal.to(device)
            labels = labels.to(device)

            outputs = model(histo, thermal)
            preds = torch.argmax(outputs, dim=1)

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    return np.array(y_true), np.array(y_pred)

# =========================
# 5️⃣ Get predictions
# =========================
y_true, y_pred = get_predictions(model, val_loader, device=device)

# =========================
# 6️⃣ Plot confusion matrix
# =========================
cm = confusion_matrix(y_true, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Benign', 'Malignant'])

disp.plot(cmap='Blues')
plt.title('Confusion Matrix')

# Save the plot as an image
plt.savefig('confusion_matrix.png')
print("Confusion matrix saved as 'confusion_matrix.png'")

# Close the plot to avoid unnecessary window pop-ups
plt.close()

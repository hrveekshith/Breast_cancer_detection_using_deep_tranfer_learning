# src/evaluate.py

import torch
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from src.dataloader import get_dataloader
# from src.model import DualBranchCNN                                            old
from src.model import DualBranchResNet as DualBranchCNN

# Load data (val_loader only)
_, val_loader = get_dataloader(batch_size=32, shuffle=False)

# Load model
model = DualBranchCNN()
model.load_state_dict(torch.load("dual_branch_cnn.pth"))
model.eval()

# Collect all predictions and true labels
all_preds = []
all_labels = []

with torch.no_grad():
    for orig_img, thermal_img, labels in val_loader:
        outputs = model(orig_img, thermal_img)
        preds = torch.argmax(outputs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# Confusion matrix
cm = confusion_matrix(all_labels, all_preds)
labels = ["Benign", "Malignant"]

plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
plt.savefig("confusion_matrix.png")
print("[INFO] Confusion matrix saved as confusion_matrix.png")


# Classification report
print("Classification Report:")
print(classification_report(all_labels, all_preds, target_names=labels))

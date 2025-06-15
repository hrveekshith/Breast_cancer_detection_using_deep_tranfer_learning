
# ===================== src/train.py =====================
import torch
import torch.nn as nn
import torch.optim as optim
from src.model import DualBranchResNet as DualBranchCNN
from src.dataloader import get_dataloader
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
import pickle

# -------------------- Focal Loss --------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=0.75, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        ce_loss = self.ce(inputs, targets)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# -------------------- Data Loaders --------------------
train_loader, val_loader = get_dataloader()

# -------------------- Model --------------------
model = DualBranchCNN()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# -------------------- Class Weights (optional for reference only) --------------------
all_train_labels = []
for _, _, labels in train_loader:
    all_train_labels.extend(labels.numpy())
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(all_train_labels), y=all_train_labels)
class_weights = torch.tensor(class_weights, dtype=torch.float).to(device)

# -------------------- Loss & Optimizer --------------------
# Use focal loss
criterion = FocalLoss(alpha=0.75, gamma=2)
optimizer = optim.Adam(model.parameters(), lr=0.0005)

# -------------------- Training Loop --------------------
num_epochs = 25
history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for org, thermal, labels in train_loader:
        org, thermal, labels = org.to(device), thermal.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(org, thermal)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    train_acc = correct / total
    train_loss = total_loss

    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for org, thermal, labels in val_loader:
            org, thermal, labels = org.to(device), thermal.to(device), labels.to(device)
            outputs = model(org, thermal)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            val_correct += (predicted == labels).sum().item()
            val_total += labels.size(0)

    val_acc = val_correct / val_total

    history["train_loss"].append(train_loss)
    history["train_acc"].append(train_acc)
    history["val_loss"].append(val_loss)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} - Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")

# -------------------- Save Model --------------------
torch.save(model.state_dict(), "dual_branch_cnn.pth")
print("Model saved as dual_branch_cnn.pth")

with open("training_history.pkl", "wb") as f:
    pickle.dump(history, f)
print("Training history saved as training_history.pkl")

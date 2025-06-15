
# ===================== src/dataloader.py =====================
# src/dataloader.py
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.dataset import BreakHisSingleDataset
from src.common_transforms import common_transform  # 👈 import this

def get_dataloader(batch_size=32, shuffle=True, thermal_dir="data/thermal", transform=common_transform):
    dataset = BreakHisSingleDataset(root_dir="data/raw")
    img_paths, labels = dataset.get_data()

    train_img_paths, val_img_paths, train_labels, val_labels = train_test_split(
        img_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )

    train_dataset = breakHisTorchDataset(train_img_paths, train_labels, thermal_dir, transform)
    val_dataset = breakHisTorchDataset(val_img_paths, val_labels, thermal_dir, transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader


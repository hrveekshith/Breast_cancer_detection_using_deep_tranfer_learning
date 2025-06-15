
# ===================== src/dataset.py =====================
import os
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch
from torchvision import transforms

# Transformation shared across both branches
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

class BreakHisSingleDataset(Dataset):
    def __init__(self, root_dir, transform=transform):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        self.transform = transform

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    path = os.path.join(root, file)
                    label = 0 if 'sob_b_' in file.lower() else 1
                    self.images.append(path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return image, label

class BreakHisDualDataset(Dataset):
    def __init__(self, histo_dir, thermal_dir, transform=transform):
        self.histo_dataset = BreakHisSingleDataset(histo_dir, transform)
        self.thermal_dataset = BreakHisSingleDataset(thermal_dir, transform)

        assert len(self.histo_dataset) == len(self.thermal_dataset), "Mismatch in histo and thermal dataset sizes."

    def __len__(self):
        return len(self.histo_dataset)

    def __getitem__(self, idx):
        histo_img, label1 = self.histo_dataset[idx]
        thermal_img, label2 = self.thermal_dataset[idx]
        assert label1 == label2, "Label mismatch between histo and thermal data."
        return (histo_img, thermal_img), label1

def get_dataloaders(batch_size=32, histo_dir='data/raw', thermal_dir='data/thermal'):
    dataset = BreakHisDualDataset(histo_dir, thermal_dir)

    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader






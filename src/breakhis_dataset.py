
import os
import torch
from torch.utils.data import Dataset
import cv2

class BreakHisDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []

        for root, _, files in os.walk(root_dir):
            for file in files:
                if file.lower().endswith(('.jpg', '.png')):
                    path = os.path.join(root, file)
                    label = 0 if 'sob_b_' in file.lower() else 1
                    self.images.append(path)
                    self.labels.append(label)

    def get_data(self):
        return self.images, self.labels


class breakHisTorchDataset(Dataset):  # ✅ make sure this name matches the import
    def __init__(self, img_paths, labels, thermal_dir, transform=None):
        self.img_paths = img_paths
        self.labels = labels
        self.thermal_dir = thermal_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        # Load original image
        org_img = cv2.imread(img_path)
        org_img = cv2.cvtColor(org_img, cv2.COLOR_BGR2RGB)

        # Load thermal image (match by filename)
        img_name = os.path.basename(img_path)
        thermal_path = os.path.join(self.thermal_dir, img_name)
        thermal_img = cv2.imread(thermal_path)
        thermal_img = cv2.cvtColor(thermal_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            org_img = self.transform(org_img)
            thermal_img = self.transform(thermal_img)
        else:
            org_img = torch.tensor(org_img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            thermal_img = torch.tensor(thermal_img, dtype=torch.float32).permute(2, 0, 1) / 255.0

        return org_img, thermal_img, torch.tensor(label, dtype=torch.long)


# src/common_transforms.py
from torchvision import transforms

common_transform = transforms.Compose([
    transforms.ToPILImage(),                     # Ensure input is PIL
    transforms.Resize((224, 224)),              # Resize all images to same size
    transforms.ToTensor(),                      # Convert to tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])  # Normalize for pretrained models
])

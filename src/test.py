
# src/test.py

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import os

# ✅ Correct import for ResNet-based model
from src.model import DualBranchResNet as DualBranchCNN

def load_model(model_path, device):
    model = DualBranchCNN()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

def preprocess_image(img_path):
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img = Image.open(img_path).convert('RGB')
    return preprocess(img).unsqueeze(0)  # shape [1, 3, 224, 224]

def predict(model, histo_tensor, thermal_tensor, device):
    histo_tensor = histo_tensor.to(device)
    thermal_tensor = thermal_tensor.to(device)

    with torch.no_grad():
        output = model(histo_tensor, thermal_tensor)
        probs = F.softmax(output, dim=1)
        confidence, predicted = torch.max(probs, 1)

        classes = ['Benign', 'Malignant']
        predicted_label = classes[predicted.item()]
        confidence_score = confidence.item() * 100

        return predicted_label, confidence_score

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_path = "dual_branch_cnn.pth"  # Your latest model
    model = load_model(model_path, device)

    histo_images = [
        "SOB_B_A-14-22549AB-40-001.png",
        "SOB_M_MC-14-13418DE-200-002.png",
        "SOB_B_TA-14-16184-200-002.png",
        "SOB_M_LC-14-15570-40-012.png",
        "SOB_B_PT-14-29315EF-200-011.png",
    ]

    for histo_path in histo_images:
        thermal_path = os.path.join("data", "thermal", os.path.basename(histo_path))

        if not os.path.exists(histo_path):
            print(f"[ERROR] Histopathological image not found: {histo_path}")
            continue
        if not os.path.exists(thermal_path):
            print(f"[ERROR] Thermal image not found: {thermal_path}")
            continue

        histo_tensor = preprocess_image(histo_path)
        thermal_tensor = preprocess_image(thermal_path)

        predicted_label, confidence_score = predict(model, histo_tensor, thermal_tensor, device)

        print(f"✅ Image: {histo_path}")
        print(f"   Prediction: {predicted_label}")
        print(f"   Confidence: {confidence_score:.2f}%\n")

if __name__ == "__main__":
    main()



import torch
import torch.nn as nn
from torchvision import models

class DualBranchResNet(nn.Module):
    def __init__(self):
        super(DualBranchResNet, self).__init__()

        # Pretrained ResNet18 for histopathological images
        resnet_org = models.resnet18(pretrained=True)
        self.org_branch = nn.Sequential(*list(resnet_org.children())[:-1])  # [B, 512, 1, 1]

        # Pretrained ResNet18 for thermal images
        resnet_thermal = models.resnet18(pretrained=True)
        self.thermal_branch = nn.Sequential(*list(resnet_thermal.children())[:-1])  # [B, 512, 1, 1]

        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 2)
        )

    def forward(self, org_img, thermal_img):
        org_feat = self.org_branch(org_img).squeeze()         # [B, 512]
        thermal_feat = self.thermal_branch(thermal_img).squeeze()  # [B, 512]

        if len(org_feat.shape) == 1:
            org_feat = org_feat.unsqueeze(0)
        if len(thermal_feat.shape) == 1:
            thermal_feat = thermal_feat.unsqueeze(0)

        combined = torch.cat((org_feat, thermal_feat), dim=1)  # [B, 1024]
        return self.classifier(combined)


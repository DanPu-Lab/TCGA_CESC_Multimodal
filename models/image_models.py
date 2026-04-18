import torch
import torch.nn as nn
import torchvision.models as models

class ImageModel(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super().__init__()
        self.resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        self.feature_dim = self.resnet.fc.in_features
        for param in list(self.resnet.parameters())[:-20]:
            param.requires_grad = False
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-1])
        self.attention = nn.Sequential(
            nn.Linear(self.feature_dim, 256), nn.Tanh(), nn.Linear(256, 1), nn.Softmax(dim=1)
        )
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        B, N, C, H, W = x.shape
        x = x.view(-1, C, H, W)
        features = self.resnet(x).view(B, N, -1)
        attn = self.attention(features)
        weighted = (features * attn).sum(dim=1)
        return self.classifier(weighted)
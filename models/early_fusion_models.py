import torch
import torch.nn as nn
import torchvision.models as models

class EarlyFusionModel(nn.Module):
    def __init__(self, gene_input_dim, num_classes=2, pretrained=True):
        super().__init__()
        # Image branch
        resnet = models.resnet50(weights='IMAGENET1K_V1' if pretrained else None)
        for param in list(resnet.parameters())[:-20]:
            param.requires_grad = False
        self.image_encoder = nn.Sequential(*list(resnet.children())[:-1])
        self.image_attn = nn.Sequential(
            nn.Linear(2048, 128), nn.Tanh(), nn.Linear(128, 1), nn.Softmax(dim=1)
        )
        # Gene branch
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        # Fusion classifier
        self.classifier = nn.Sequential(
            nn.Linear(2048 + 256, 512), nn.BatchNorm1d(512), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(512, num_classes)
        )

    def forward(self, images, genes):
        B, N, C, H, W = images.shape
        images = images.view(-1, C, H, W)
        img_feat = self.image_encoder(images).view(B, N, -1)
        attn = self.image_attn(img_feat)
        img_feat = (img_feat * attn).sum(dim=1)
        gene_feat = self.gene_encoder(genes)
        fused = torch.cat([img_feat, gene_feat], dim=1)
        return self.classifier(fused)
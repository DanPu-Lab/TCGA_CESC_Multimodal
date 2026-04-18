import torch
import torch.nn as nn
import torchvision.models as models

class LateFusionModel(nn.Module):
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
        self.image_classifier = nn.Sequential(
            nn.Linear(2048, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        # Gene branch
        self.gene_encoder = nn.Sequential(
            nn.Linear(gene_input_dim, 256), nn.BatchNorm1d(256), nn.ReLU(), nn.Dropout(0.3)
        )
        self.gene_classifier = nn.Linear(256, num_classes)
        # Learnable fusion weights
        self.image_weight = nn.Parameter(torch.tensor(0.5))
        self.gene_weight = nn.Parameter(torch.tensor(0.5))

    def forward(self, images, genes):
        B, N, C, H, W = images.shape
        images = images.view(-1, C, H, W)
        img_feat = self.image_encoder(images).view(B, N, -1)
        attn = self.image_attn(img_feat)
        img_feat = (img_feat * attn).sum(dim=1)
        img_logits = self.image_classifier(img_feat)
        img_probs = torch.softmax(img_logits, dim=1)

        gene_feat = self.gene_encoder(genes)
        gene_logits = self.gene_classifier(gene_feat)
        gene_probs = torch.softmax(gene_logits, dim=1)

        weights = torch.softmax(torch.stack([self.image_weight, self.gene_weight]), dim=0)
        fused_probs = weights[0] * img_probs + weights[1] * gene_probs
        return fused_probs, img_probs, gene_probs
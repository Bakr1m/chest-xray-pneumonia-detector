"""Model definitions: small from-scratch CNN (Day 33) + transfer heads (Day 34+)."""
from torch import nn
from torchvision import models


class SmallCNN(nn.Module):
    """3-conv baseline (~150k params): establishes the floor before transfer."""

    def __init__(self, num_classes=2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 112
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),  # 56
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 64 x 1 x 1
        )
        self.head = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x).flatten(1)
        return self.head(x)


def mobilenet_frozen(num_classes=2):
    """MobileNetV2, ImageNet backbone frozen, new classifier head (Day 34)."""
    net = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    for p in net.features.parameters():
        p.requires_grad = False
    net.classifier[1] = nn.Linear(net.classifier[1].in_features, num_classes)
    return net

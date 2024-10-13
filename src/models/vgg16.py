from typing import Optional

import torch
from torch import nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: Optional[float] = 0.5,
        pretrained: Optional[bool] = True,
    ):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

    def forward(self, x):
        x = self.vgg16.features(x)
        x = torch.flatten(x, 1)

        for i, layer in enumerate(self.vgg16.classifier):
            x = layer(x)

        return x

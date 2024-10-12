import random
from typing import Optional

import torch
from torch import nn
from torchvision import models


class VGG16(nn.Module):
    def __init__(
        self,
        num_classes: int,
        dropout: Optional[float] = 0.5,
        st_depth_prob: Optional[float] = 0.5,
        pretrained: Optional[bool] = True,
    ):
        super(VGG16, self).__init__()
        self.vgg16 = models.vgg16(pretrained=pretrained)
        self.st_depth_prob = st_depth_prob

        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)
        )

    def stochastic_depth(self, x, layer):
        if self.training and random.uniform(0, 1) < self.st_depth_prob:
            return x
        else:
            return layer(x)

    def forward(self, x):
        x = self.vgg16.features(x)
        x = torch.flatten(x, 1)

        for i, layer in enumerate(self.vgg16.classifier):
            if isinstance(layer, nn.Linear):
                x = self.stochastic_depth(x, layer)
            else:
                x = layer(x)

        return x

import random
from typing import Optional
#
# import torch
# from torch import nn
# from torchvision import models
#
#
# class VGG16(nn.Module):
#     def __init__(
#         self,
#         num_classes: int,
#         dropout: Optional[float] = 0.5,
#         stochastic_depth: Optional[float] = 0.5,
#         pretrained: Optional[bool] = True,
#     ):
#         super(VGG16, self).__init__()
#         self.vgg16 = models.vgg16(pretrained=pretrained)
#         self.stochastic_depth = stochastic_depth
#
#         self.vgg16.classifier = nn.Sequential(
#             nn.Linear(25088, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, 4096),
#             nn.ReLU(True),
#             nn.Dropout(p=dropout),
#             nn.Linear(4096, num_classes)
#         )
#
#     def forward(self, x):
#         # pass through feature extractor (vgg16 conv layers)
#         x = self.vgg16.features(x)
#
#         # unfold tensor into vector to enter fc layer
#         x = torch.flatten(x, 1)
#
#         # apply stochastic depth to fc layers
#         for i, layer in enumerate(self.vgg16.classifier):
#             if isinstance(layer, nn.Linear):
#                 if self.training and random.uniform(0, 1) < self.stochastic_depth:
#                     continue
#
#             x = layer(x)
#
#         return x


import torch
from torch import nn
from torchvision import models

class VGG16(nn.Module):
    def __init__(
        self,
        num_classes: int = 37,  # 37 pet breeds
        dropout: Optional[float] = 0.5,
        stochastic_depth: Optional[float] = 0.5,
        pretrained: Optional[bool] = True,
    ):
        super(VGG16, self).__init__()
        # Load pre-trained VGG16 model
        self.vgg16 = models.vgg16(pretrained=pretrained)
        self.stochastic_depth = stochastic_depth

        # Modify the classifier to match the number of classes for Oxford Pets (37 classes)
        self.vgg16.classifier = nn.Sequential(
            nn.Linear(25088, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, num_classes)  # Output layer for 37 classes
        )

    def forward(self, x):
        # Pass through VGG16 feature extractor
        x = self.vgg16.features(x)

        # Flatten the tensor into a vector for the fully connected layers
        x = torch.flatten(x, 1)

        # Apply stochastic depth to fully connected layers
        for layer in self.vgg16.classifier:
            x = layer(x)

        return x

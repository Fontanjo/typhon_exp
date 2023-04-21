import torch
import torch.nn as nn
import copy
from torchvision import models
import torch.nn.functional as F
from functools import partial


# Adapted model from: https://github.com/mniwk/RF-Net/tree/main/models
def get_block(dropout, in_channels=1):
    return Res_net_container(in_channels=in_channels)


class Res_net_container(nn.Module):
    def __init__(self, in_channels):
        super(Res_net_container, self).__init__()

        # Original paper they used pretrained weights
        self.resnet = models.resnet34()
        # Disable bias for convolutions direclty followed by a batch norm
        # https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html
        self.firstconv = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.firstbn = self.resnet.bn1
        self.firstrelu = self.resnet.relu
        self.firstmaxpool = self.resnet.maxpool
        self.encoder1 = self.resnet.layer1
        self.encoder2 = self.resnet.layer2
        self.encoder3 = self.resnet.layer3
        self.encoder4 = self.resnet.layer4


    def forward(self, inputs):
        # Block 1
        x = self.firstconv(inputs)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x = self.firstmaxpool(x)
        # Block 2
        s2 = self.encoder1(x)
        # Block 3
        s3 = self.encoder2(s2)
        # Block 4
        s4 = self.encoder3(s3)
        # Block 5
        x = self.encoder4(s4)

        return [x, s4, s3, s2]


    def __iter__(self):
        return iter([
            self.firstconv,
            self.firstbn,
            self.firstrelu,
            self.firstmaxpool,
            self.encoder1,
            self.encoder2,
            self.encoder3,
            self.encoder4,
        ])

import torch
import torch.nn as nn
import copy
from torchvision import models
import torch.nn.functional as F
from functools import partial


# Adapted model from: https://github.com/mniwk/RF-Net/tree/main/models
def get_block(dropout):
    return Unet_container()


# nonlinearity = partial(F.relu, inplace=True)
nonlinearity = F.relu


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.relu1 = nonlinearity

        self.deconv2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(in_channels//4, in_channels//4, 3, padding=1),
            nn.BatchNorm2d(in_channels // 4)
        )
        self.relu2 = nonlinearity

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nonlinearity

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu1(x)

        x = self.deconv2(x)
        x = self.relu2(x)

        x = self.conv3(x)
        x = self.norm3(x)
        x = self.relu3(x)

        return x


class Unet_container(nn.Module):
    def __init__(self):
        super(Unet_container, self).__init__()

        self.decoder4 = DecoderBlock(512, 256)
        self.decoder3 = DecoderBlock(256, 128)
        self.decoder2 = DecoderBlock(128, 64)
        self.decoder1 = DecoderBlock(64, 64)

        self.finalconv1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(64, 32, 3, padding=1)
        )
        self.finalrelu1 = nonlinearity
        self.finalconv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.finalrelu2 = nonlinearity
        self.finalconv3 = nn.Conv2d(32, 1, 3, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, inputs):
        # Extract values
        x, s4, s3, s2 = inputs

        x = self.decoder4(x) + s4
        x = self.decoder3(x) + s3
        x = self.decoder2(x) + s2
        x = self.decoder1(x)

        x = self.finalconv1(x)
        x = self.finalrelu1(x)
        x = self.finalconv2(x)
        x = self.finalrelu2(x)
        x = self.finalconv3(x)
        x = self.sigmoid(x)

        return x


    def __iter__(self):
        return iter([
            self.decoder4,
            self.decoder3,
            self.decoder2 ,
            self.decoder1,
            self.finalconv1,
            self.finalrelu1,
            self.finalconv2,
            self.finalrelu2,
            self.finalconv3,
            self.sigmoid,
        ])

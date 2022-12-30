import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.ConvTranspose2d(1, 64, kernel_size=3, stride=2, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
        nn.Sigmoid()
    )

import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=16, out_channels=3, kernel_size=31, stride=1, padding=15),
        nn.Sigmoid()
    )

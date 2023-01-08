import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(3),
        nn.ReLU(True),

        nn.Conv2d(in_channels=3, out_channels=3, kernel_size=5, stride=1, padding=2),
        # nn.BatchNorm2d(3),
        # nn.ReLU(True),
        nn.Sigmoid()
    )

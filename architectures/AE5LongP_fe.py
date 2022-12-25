import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(True),

        nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),

        nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(True),

        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(True),

        nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        nn.BatchNorm2d(512),
        nn.ReLU(True),
    )

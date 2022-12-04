import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.ConvTranspose2d(128, 64, kernel_size=3,
        stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.ConvTranspose2d(64, 32, kernel_size=3,
        stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(32),
        nn.ReLU(True),

        nn.ConvTranspose2d(32, 16, kernel_size=3,
        stride=2, padding=1, output_padding=1),
        nn.BatchNorm2d(16),
        nn.ReLU(True),

        nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
        padding=1, output_padding=1),
        nn.BatchNorm2d(8),
        nn.ReLU(True),

        nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2,
        padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(True),

        nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(3),
        nn.ReLU(True),
        
        nn.ConvTranspose2d(3, 3, kernel_size=2, stride=1),
        nn.Sigmoid()
    )

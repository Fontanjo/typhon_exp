import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Linear(128, 52*40),
        nn.ELU(),
        nn.Dropout(p=dropout),

        nn.Unflatten(1, (1, 52, 40)),

        nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
        nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=(0, 1), output_padding=1),
        nn.BatchNorm2d(1),
        nn.ReLU(True),

        nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
        nn.Sigmoid()
    )

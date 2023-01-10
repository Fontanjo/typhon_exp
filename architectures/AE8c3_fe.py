import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(64),
        nn.ReLU(True),

        nn.Conv2d(in_channels=64, out_channels=1, kernel_size=8, stride=4),
        
        nn.Flatten(), # shape = (batch, 8400)

        nn.Linear(52*40, 128),
        nn.ELU(),
        nn.Dropout(p=dropout),

        nn.Linear(128, 128),
        nn.ELU(),
        nn.Dropout(p=dropout)
    )

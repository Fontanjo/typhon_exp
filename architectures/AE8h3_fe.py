import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=256, kernel_size=5, stride=1, padding=2),
        nn.BatchNorm2d(256),
        nn.ELU(),

        # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
        # nn.BatchNorm2d(64),
        # nn.ELU(),

        # nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1),
        # nn.BatchNorm2d(1),
        # nn.ELU(),
        #
        # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),    # shape = (batch, 1, 105, 80)
        # nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),    # shape = (batch, 1, 52, 40)
        # nn.Flatten(), # shape = (batch, 8400)

        nn.Conv2d(in_channels=256, out_channels=1, kernel_size=8, stride=4, padding=(2,0)),

        nn.Flatten(),

        nn.Linear(52*39, 128),
        nn.ELU(),
        nn.Dropout(p=dropout),

        nn.Linear(128, 128),
        nn.ELU(),
        nn.Dropout(p=dropout)
    )

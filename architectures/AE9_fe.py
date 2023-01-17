import torch.nn as nn

def get_block(dropout, in_channels=1):
    return nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),

        # TODO: Check dilation
        # torch.nn.MaxPool3d(kernel_size=(64, 5, 5), stride=4, padding=(0, 2, 0), dilation=1, return_indices=True, ceil_mode=False)
        # 210x160
        torch.nn.MaxPool3d(kernel_size=(64, 3, 3), stride=3, dilation=1, return_indices=True, ceil_mode=True), # 1x70x54


        nn.BatchNorm2d(3),  # Rescales image-wise coordinate [0,1]

        # nn.Flatten(),
        # nn.Linear(52*39, 128),
        # nn.ELU(),
        # nn.Dropout(p=dropout),
        #
        # nn.Linear(128, 128),
        # nn.ELU(),
        # nn.Dropout(p=dropout)
    )

import torch.nn as nn

def get_block(dropout, num_classes=2):
    return nn.Sequential(
        #
        # nn.Conv2d(64, 16, kernel_size=1, padding=0),

        nn.Flatten(),

        # nn.Linear(64, 16),
        # nn.ELU(),
        #
        nn.Linear(262144, num_classes)
    )

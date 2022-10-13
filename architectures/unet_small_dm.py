import torch.nn as nn

def get_block(dropout, num_classes=2):
    return nn.Sequential(
        nn.Conv2d(64, 1, kernel_size=1, padding=0), # Original unet

        nn.Sigmoid()  # Not (or yes?) in the original model. Added since loss requires values in [0, 1]
    )

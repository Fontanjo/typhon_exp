import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return AE11()




# Creating a PyTorch class
class AE11(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ELU()
        )


    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)

        return x

    def __iter__(self):
        return iter([
            self.encoder_cnn,
        ])

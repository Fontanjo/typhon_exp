import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return Autoencoder_fe()

class Autoencoder_fe(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),
        )
        # out: (Bx64x210x160)

        self.mp = torch.nn.MaxPool3d(kernel_size=(64, 1, 1), stride=1, dilation=1, return_indices=False, ceil_mode=True)
        # out: (Bx1x210x160)



    def forward(self, inputs):
        # Encode
        x = self.encoder_cnn(inputs)
        # Find max values
        x = self.mp(x)
        # Flatten last 2 dims
        x = torch.flatten(x, start_dim=2)
        # Get only 128 max values
        top_k, idx = torch.topk(x, 128)

        # Pass a single object
        return [x, idx]


    def __iter__(self):
        return iter([
            self.encoder_cnn,
            self.mp
        ])

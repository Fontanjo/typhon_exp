import torch
import torch.nn as nn
import sys
from pathlib import Path

def get_block(dropout, in_channels=1):
    return AE9c(dropout)



# Creating a PyTorch class
class AE9c(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()

        # Reconstruct background
        self.decoder_conv1 = nn.Sequential(
            nn.Linear(128, 52*40),
            nn.ELU(),
            nn.Dropout(p=dropout),

            nn.Unflatten(1, (1, 52, 40)),

            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=(0, 1), output_padding=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
        )

        self.linear = nn.Linear(128, 128)


        self.decoder_conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

        self.uf = nn.Unflatten(2, (210, 160))


    def forward(self, x):
        # Split back
        top_k, idx = x

        top_k = self.linear(top_k)

        if torch.cuda.is_available():
            cuda_device = f'cuda:{top_k.get_device()}'
        else:
            cuda_device = 'cpu'

        # Reconstruct background
        bg = torch.ones(top_k.shape[0], 128).to(cuda_device)
        bg = self.decoder_conv1(bg)

        # Insert values
        bg = torch.flatten(bg, start_dim=2)
        # bg.register_hook(self.my_hook)
        rec = bg.scatter(-1, idx, top_k)

        # Unflatten
        rec = self.uf(rec)
        # Some more convs
        x = self.decoder_conv2(rec)
        return x


    def __iter__(self):
        return iter([
            self.linear,
            self.decoder_conv1,
            self.decoder_conv2,
            self.uf,
        ])

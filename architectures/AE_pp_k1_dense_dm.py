import torch
import torch.nn as nn
import sys
from pathlib import Path

def get_block(dropout, in_channels=1):
    return AE10(dropout)

K_VALS = 1

# Creating a PyTorch class
class AE10(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()

        # Reconstruct background
        self.decoder_dense = nn.Sequential(
            nn.Linear(1, 3*210*160)
        )

        # Merge bg and sprites
        self.decoder_conv2 = nn.Sequential(
            # nn.ConvTranspose2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            # nn.BatchNorm2d(64),
            # nn.ReLU(),
            #
            # nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            # nn.Sigmoid()

            nn.ConvTranspose2d(in_channels=6, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        # Reconstruct sprites only
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=K_VALS*2, out_channels=64, kernel_size=5, stride=1, padding=2),
            # nn.ConvTranspose2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

        self.uf = nn.Unflatten(2, (210, 160))


    def forward(self, x):
        # Split back
        reduced_idx3, reduced_top3, idx = x

        # Get batch size (for readability)
        batch_size = idx.shape[0]

        if torch.cuda.is_available() and idx.get_device() != -1:
            cuda_device = f'cuda:{idx.get_device()}'
        else:
            cuda_device = 'cpu'

        # Reconstruct background
        bg = torch.ones(batch_size, 1).to(cuda_device)
        bg = self.decoder_dense(bg)
        bg = bg.reshape(batch_size, 3, 210, 160)

        # bg = torch.flatten(bg, start_dim=2)


        ##################
        ## Insert coordinates of top3 in new tensors 210x160 ##
        ##################

        # Insert values
        zeros_z0 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)

        # Insert values, one channel for each one (select with [:,k])
        rec_z0 = zeros_z0.scatter(-1, idx, reduced_idx3[:,0].reshape(batch_size, 1, reduced_idx3.shape[-1])) # Make sure it has the correct shape

        # Unflatten
        rec_z0 = self.uf(rec_z0)


        ##################
        ## Insert values of top3 in new tensors 210x160 ##
        ##################

        # Insert values
        zeros_v0 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)

        # Insert values, one channel for each one
        rec_v0 = zeros_v0.scatter(-1, idx, reduced_top3[:,0].reshape(batch_size, 1, reduced_top3.shape[-1]))

        # Unflatten
        rec_v0 = self.uf(rec_v0)

        sprites_infos = torch.cat([rec_z0, rec_v0], dim=1)
        sprites = self.decoder_conv3(sprites_infos)

        # Merge bg and sprites
        merged = torch.cat([bg, sprites], dim=1)

        # Some more convs
        x = self.decoder_conv2(merged)
        return x


    def __iter__(self):
        return iter([
            self.decoder_dense,
            self.decoder_conv2,
            self.uf,
        ])

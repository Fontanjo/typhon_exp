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

            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )


        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=9, out_channels=64, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=1, stride=1, padding=0),
            nn.Sigmoid()
        )

        self.uf = nn.Unflatten(2, (210, 160))


    def forward(self, x):
        # Split back
        reduced_idx3, reduced_top3, idx = x

        # Get batch size (for readability)
        batch_size = idx.shape[0]

        if torch.cuda.is_available():
            cuda_device = f'cuda:{idx.get_device()}'
        else:
            cuda_device = 'cpu'

        # Reconstruct background
        bg = torch.zeros(batch_size, 128).to(cuda_device)
        bg = self.decoder_conv1(bg)

        # bg = torch.flatten(bg, start_dim=2)


        ##################
        ## Insert coordinates of top3 in new tensors 210x160 ##
        ##################

        # Insert values
        zeros_z0 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)
        zeros_z1 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)
        zeros_z2 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)

        # print('idx', idx.shape)
        # print('reduced_idx3[0]', reduced_idx3[0].shape)
        # print('zeros_z0', zeros_z0.shape)
        #
        # exit(1)

        # Insert values, one channel for each one (select with [:,k])
        rec_z0 = zeros_z0.scatter(-1, idx, reduced_idx3[:,0].reshape(batch_size, 1, reduced_idx3.shape[-1])) # Make sure it has the correct shape
        rec_z1 = zeros_z1.scatter(-1, idx, reduced_idx3[:,1].reshape(batch_size, 1, reduced_idx3.shape[-1])) #  by adding batch and channel dim
        rec_z2 = zeros_z2.scatter(-1, idx, reduced_idx3[:,2].reshape(batch_size, 1, reduced_idx3.shape[-1]))

        # Unflatten
        rec_z0 = self.uf(rec_z0)
        rec_z1 = self.uf(rec_z1)
        rec_z2 = self.uf(rec_z2)


        ##################
        ## Insert values of top3 in new tensors 210x160 ##
        ##################

        # Insert values
        zeros_v0 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)
        zeros_v1 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)
        zeros_v2 = torch.zeros(batch_size, 1, 210*160).to(cuda_device)

        # Insert values, one channel for each one
        rec_v0 = zeros_v0.scatter(-1, idx, reduced_top3[:,0].reshape(batch_size, 1, reduced_top3.shape[-1]))
        rec_v1 = zeros_v1.scatter(-1, idx, reduced_top3[:,1].reshape(batch_size, 1, reduced_top3.shape[-1]))
        rec_v2 = zeros_v2.scatter(-1, idx, reduced_top3[:,2].reshape(batch_size, 1, reduced_top3.shape[-1]))

        # Unflatten
        rec_v0 = self.uf(rec_v0)
        rec_v1 = self.uf(rec_v1)
        rec_v2 = self.uf(rec_v2)


        # Merge channels
        merged = torch.cat([bg, rec_z0, rec_z1, rec_z2, rec_v0, rec_v1, rec_v2], dim=1)

        # Some more convs
        x = self.decoder_conv2(merged)
        return x


    def __iter__(self):
        return iter([
            self.decoder_conv1,
            self.decoder_conv2,
            self.uf,
        ])

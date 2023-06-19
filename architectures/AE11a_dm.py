import torch
import torch.nn as nn
import sys
from pathlib import Path

def get_block(dropout, in_channels=1):
    return AE11(dropout)



# Creating a PyTorch class
class AE11(torch.nn.Module):
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

        # Merge bg and sprites
        self.decoder_conv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

        # Reconstruct sprites only
        self.decoder_conv3 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=6, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )

        self.uf = nn.Unflatten(2, (210, 160))

        self.mp0 = torch.nn.MaxPool3d(kernel_size=(3, 1, 1), stride=1, dilation=1, return_indices=True, ceil_mode=True) # ~64x64x3


    def forward(self, x):
        if torch.cuda.is_available():
            cuda_device = f'cuda:{x.get_device()}'
        else:
            cuda_device = 'cpu'


        # Put channel dimension as last
        channel_last_dim = x.transpose(1, 2).transpose(2, 3)

        # Get top 3 values for each pixel
        top3, idx3 = torch.topk(channel_last_dim, 3)


        # Move back channel dimension
        top3 = top3.transpose(2, 3).transpose(1, 2)
        idx3 = idx3.transpose(2, 3).transpose(1, 2)

        # And top 1 (to apply next topk without risk of taking multiple time the same coordinate. Maybe this is not necessary)
        max_vals, idxmp = self.mp0(top3)


        # Flatten last 2 dims
        max_vals = torch.flatten(max_vals, start_dim=2)
        # Get only 128 max values
        top_k, idx = torch.topk(max_vals, 128)


        # Keep only selected coords
        flatten_top3 = torch.flatten(top3, start_dim=2)
        flatten_idx3 = torch.flatten(idx3, start_dim=2)

        # Add 1 channel (otherwise .cat will merge the 3 channels with the batch channel)
        flatten_top3 = flatten_top3.reshape(flatten_top3.shape[0], 1, *flatten_top3.shape[1:])
        flatten_idx3 = flatten_idx3.reshape(flatten_idx3.shape[0], 1, *flatten_idx3.shape[1:])

        # Iterate over batch --> There should be a better way
        reduced_top3 = torch.cat([torch.index_select(flatten_top3[b], 2, idx[b].squeeze()) for b in range(idx.shape[0])]).to(cuda_device)
        reduced_idx3 = torch.cat([torch.index_select(flatten_idx3[b,:], 2, idx[b].squeeze()) for b in range(idx.shape[0])]).to(cuda_device)


        # Transform to same type
        reduced_top3 = reduced_top3.float()
        reduced_idx3 = reduced_idx3.float()

        ################################################
        ############## Center of AE ####################
        ################################################
        # # Return a single object
        # x = [reduced_idx3, reduced_top3, idx]
        #
        # # Split back
        # reduced_idx3, reduced_top3, idx = x

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

        sprites_infos = torch.cat([rec_z0, rec_z1, rec_z2, rec_v0, rec_v1, rec_v2], dim=1)
        sprites = self.decoder_conv3(sprites_infos)

        # Merge bg and sprites
        merged = torch.cat([bg, sprites], dim=1)

        # Some more convs
        x = self.decoder_conv2(merged)
        return x


    def __iter__(self):
        return iter([
            self.decoder_conv1,
            self.decoder_conv2,
            self.uf,
            self.mp0,
        ])




def index_select_multiple_dimensions(tensor, dim, index):
    return tensor.gather(dim, index.unsqueeze(dim)).squeeze(dim)


def unflatten_coords(idx, shape):
    rest = shape
    coords = []
    divisor = 1
    while rest:
        *rest, curr = rest
        # if not rest: break
        coords.append(idx // divisor % curr) # or modify idx (idx // divisor) each loop
        divisor *= curr
    return coords

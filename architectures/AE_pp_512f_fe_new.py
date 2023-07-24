import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return AE()


NB_FEATURES = 512

"""Applies multiple kernels on the input and returns only those giving max value


# TODO add some assertions for the sizes at the various steps


Parameters
----------
x : torch tensor (?), 4-dimensional (batch, channel, x, y)
    The input tensor, classical nn module input
n_features : int, >= 1
    Number of features to identify (number of kernel match)
k_vals : int, >= 1
    For each feature identified (x,y coord), how many kernels (z) to consider (k from top_k, used to get it)

Returns
-------
x : torch tensor
    Output of the module, in form of [reduced_idx3, reduced_top3, idx] (?)
"""

class PixelPerfect(torch.nn.Module):
    def __init__(self, n_features=128, k_vals=3):
        super().__init__()
        self.n_features = n_features
        self.k_vals = k_vals

        # TODO move directly in forward, no reasons to define here
        self.mp0 = torch.nn.MaxPool3d(kernel_size=(3, 1, 1), stride=1, dilation=1, return_indices=True, ceil_mode=True) # ~64x64x3

    def forward(self, x):
        # Get cuda device of input
        if torch.cuda.is_available() and x.get_device() != -1:
            cuda_device = f'cuda:{x.get_device()}'
        else:
            cuda_device = 'cpu'


        # Put channel dimension as last
        channel_last_dim = x.transpose(1, 2).transpose(2, 3)

        # Get top k values for each pixel
        topk, idxk = torch.topk(channel_last_dim, self.k_vals)


        # Move back channel dimension
        topk = topk.transpose(2, 3).transpose(1, 2)
        idxk = idxk.transpose(2, 3).transpose(1, 2)

        # And top 1 (to apply next topk without risk of taking multiple time the same coordinate. Maybe this is not necessary)
        max_vals, idxmp = self.mp0(topk)


        # Flatten last 2 dims
        max_vals = torch.flatten(max_vals, start_dim=2)
        # Get the coordinates of the n max values
        _, idxn = torch.topk(max_vals, self.n_features)


        # Keep only selected coords
        flatten_topk = torch.flatten(topk, start_dim=2)
        flatten_idxk = torch.flatten(idxk, start_dim=2)

        # Add 1 channel (otherwise .cat will merge the 3 channels with the batch channel)
        flatten_topk = flatten_topk.reshape(flatten_topk.shape[0], 1, *flatten_topk.shape[1:])
        flatten_idxk = flatten_idxk.reshape(flatten_idxk.shape[0], 1, *flatten_idxk.shape[1:])

        # Iterate over batch --> There should be a better way
        reduced_topk = torch.cat([torch.index_select(flatten_topk[b], 2, idxn[b].squeeze()) for b in range(idxn.shape[0])]).to(cuda_device)
        reduced_idxk = torch.cat([torch.index_select(flatten_idxk[b,:], 2, idxn[b].squeeze()) for b in range(idxn.shape[0])]).to(cuda_device)


        # Transform to same type
        reduced_topn = reduced_topk.float()
        reduced_idxk = reduced_idxk.float()



        # Return a single object
        x = [reduced_idxk, reduced_topk, idxn]

        return x


# Creating a PyTorch class
class AE(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ELU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(128),
            nn.ELU()
        )


    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)

        x = PixelPerfect(n_features=NB_FEATURES, k_vals=3)(x)

        return x


    def __iter__(self):
        return iter([
            self.encoder_cnn
        ])

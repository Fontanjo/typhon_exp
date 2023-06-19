import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return AE10()




# Creating a PyTorch class
class AE10(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=128, kernel_size=7, stride=1, padding=3),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.mp0 = torch.nn.MaxPool3d(kernel_size=(3, 1, 1), stride=1, dilation=1, return_indices=True, ceil_mode=True) # ~64x64x3


    def forward(self, x):
        if torch.cuda.is_available():
            cuda_device = f'cuda:{x.get_device()}'
        else:
            cuda_device = 'cpu'

        # Encode
        x = self.encoder_cnn(x)

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


        # Return a single object
        x = [reduced_idx3, reduced_top3, idx]

        return x


    def __iter__(self):
        return iter([
            self.encoder_cnn,
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

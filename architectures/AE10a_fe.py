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
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ELU()
        )

        self.mp0 = torch.nn.MaxPool3d(kernel_size=(64, 1, 1), stride=1, dilation=1, return_indices=True, ceil_mode=True) # ~64x64x3


    def forward(self, x):
        if torch.cuda.is_available():
            cuda_device = f'cuda:{x.get_device()}'
        else:
            cuda_device = 'cpu'

        # Encode
        x = self.encoder_cnn(x)

        # Find max values
        max_vals, idxmp = self.mp0(x)

        # For each [w,h], get the z-coordinate at which the max was found ("which mask")
        idx_correct = unflatten_coords(idxmp, x.shape) # returns indices in (h, w, z, batch)
        z_coord = idx_correct[2]
        # x_coord = idx_correct[1]
        # y_coord = idx_correct[0]

        # Flatten last 2 dims
        max_vals = torch.flatten(max_vals, start_dim=2)
        # Get only 128 max values
        top_k, idx = torch.topk(max_vals, 128)

        # Keep only selected coords
        flatten_z_coord = torch.flatten(z_coord, start_dim=2)

        reduced_z_coord = torch.cat([torch.index_select(flatten_z_coord[i], 1, idx[i].squeeze()) for i in range(idx.shape[0])]).to(cuda_device)

        # Add one dimension
        reduced_z_coord = reduced_z_coord.reshape(reduced_z_coord.shape[0], 1, reduced_z_coord.shape[1])

        # Transform to same type
        reduced_z_coord = reduced_z_coord.float()

        # Return a single object
        x = [reduced_z_coord, top_k, idx]

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

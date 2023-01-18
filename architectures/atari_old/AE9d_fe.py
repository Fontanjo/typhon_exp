import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return AE9c()




# Creating a PyTorch class
class AE9c(torch.nn.Module):
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

        self.mp0 = torch.nn.MaxPool3d(kernel_size=(64, 1, 1), stride=1, dilation=1, return_indices=False, ceil_mode=True) # ~64x64x3

    def my_hook(self, grad):
      grad = grad.clone()
      grad[0][0] = 50
      print('#############################')
      print('#############################')
      print("new grad: ", grad)
      print('#############################')
      return grad

    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)
        # Find max values
        x = self.mp0(x)
        # Flatten last 2 dims
        x = torch.flatten(x, start_dim=2)
        # x.register_hook(self.my_hook)
        # Get only 64 max values
        top_k, idx = torch.topk(x, 64)

        # top_k.register_hook(self.my_hook)
        # Return a single object
        x = [top_k, idx]

        return x


    def __iter__(self):
        return iter([
            self.encoder_cnn,
            self.mp0,
        ])

import torch
import torch.nn as nn
import sys
from pathlib import Path

def get_block(dropout, in_channels=1):
    return AE9c()



# Creating a PyTorch class
class AE9c(torch.nn.Module):
    def __init__(self):
        super().__init__()

        # Reconstruct background
        self.decoder_conv1 = nn.Sequential(
            nn.Linear(64, 52*40),
            nn.ELU(),
            nn.Dropout(p=0.1),

            nn.Unflatten(1, (1, 52, 40)),

            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=(0, 1), output_padding=1),
            nn.BatchNorm2d(1),
            nn.ELU(),
        )


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


    def my_hook(self, grad):
      grad = grad.clone()
      print('#############################')
      print('#############################')
      print("new grad: ", grad.shape)
      print('#############################')
      return grad


    def forward(self, x):
        # cuda_device = sys.argv[-1] if not (sys.argv[-1].endswith('.py') or sys.argv[-1].startswith('-')) else Path(__file__).stem.split('_')[-1]
        # print(cuda_device)
        # exit(1)
        # Split back
        top_k, idx = x

        if torch.cuda.is_available():
            cuda_device = f'cuda:{top_k.get_device()}'
        else:
            cuda_device = 'cpu'

        # Reconstruct background
        bg = torch.ones(top_k.shape[0], 64).to(cuda_device)
        bg = self.decoder_conv1(bg)


        # Insert values
        bg = torch.flatten(bg, start_dim=2)
        # bg.register_hook(self.my_hook)
        # rec = bg.scatter(2, idx, top_k)
                                                                            # CAUSE: fe passed the wrong tensor, that't causing the problem
                                                                            ###############################
        rec = bg.scatter_add(2, idx.repeat([1, 1, int(33600/64)]), top_k)   ####### Here is the main problem, the gradient does not backpropagate
                                                                            ##### behind here due to the shape difference.
                                                                            ##### (128 while it should be 33600)
                                                                            #####   find a better solution than 'repeat' (which in addition
                                                                            ##### forces to have a central size as a divisor of 210x160)
                                                                            ###############################
        # rec.register_hook(self.my_hook)
        rec = self.uf(rec)
        # rec = torch.unflatten(rec, 2, (210, 160)) # For some reasons this throws an error
        # rec.register_hook(self.my_hook)
        # Some more convs
        x = self.decoder_conv2(rec)
        # x.register_hook(self.my_hook)
        return x


    def __iter__(self):
        return iter([
            self.decoder_conv1,
            self.decoder_conv2,
            self.uf,
        ])

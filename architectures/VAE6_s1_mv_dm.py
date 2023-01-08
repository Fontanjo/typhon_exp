import torch
import torch.nn as nn



def get_block(dropout, in_channels=1):
    return VAE6()


class VAE6(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=5, stride=1, padding=2),
            nn.Sigmoid()
        )


    def forward(self, x):
        # Extract mu and logvar
        #  In theory it is not necessary for the dm to receive and forward mu and logvar,
        #  we could simply keep them from the fe output. But this way we avoid changing the base structure
        #  with useless and difficult-to-read conditional statements
        x, mu, logvar = x
        # Decode
        x = self.decoder_cnn(x)
        return x, mu, logvar


    def __iter__(self):
        return iter([
            self.decoder_cnn
        ])

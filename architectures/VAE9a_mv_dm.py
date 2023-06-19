import torch
import torch.nn as nn



def get_block(dropout, in_channels=1):
    return VAE8(dropout)


class VAE8(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

        ### Convolutional section
        self.decoder = nn.Sequential(
            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(128, 52*40),
            nn.ELU(),
            nn.Dropout(p=self.dropout),

            nn.Unflatten(1, (1, 52, 40)),

            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ConvTranspose2d(1, 1, kernel_size=3, stride=2, padding=(0, 1), output_padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),

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
        x = self.decoder(x)
        return x, mu, logvar


    def __iter__(self):
        return iter([
            self.decoder
        ])

import torch
import torch.nn as nn



def get_block(dropout, in_channels=1):
    return VAE3()


class VAE3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.ConvTranspose2d(512, 256, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.ConvTranspose2d(256, 128, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.ConvTranspose2d(128, 64, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 32, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.ConvTranspose2d(32, 16, kernel_size=3,
            stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2,
            padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.ConvTranspose2d(8, 3, kernel_size=3, stride=2,
            padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.ConvTranspose2d(3, 3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(True),

            nn.ConvTranspose2d(3, 3, kernel_size=2, stride=1),
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

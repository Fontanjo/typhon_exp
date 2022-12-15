import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return VAE2()


class VAE2(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(8), # Was missing
            nn.ReLU(True),
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True)
        )

        self.conv_mu = nn.Conv2d(128, 128, kernel_size=2, stride=2)
        self.conv_var = nn.Conv2d(128, 128, kernel_size=2, stride=2)


    def sample(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        s = mu + eps*std
        return s

    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)
        # Get mu and sigma
        mu = self.conv_mu(x)
        var = self.conv_var(x)
        # Sample
        x = self.sample(mu, var)
        if x.isnan().any():
            print('Nan generated, stop')
            exit(-1)
        return x, mu, var


    def __iter__(self):
        return iter([
            self.encoder_cnn,
            self.conv_mu,
            self.conv_var
        ])

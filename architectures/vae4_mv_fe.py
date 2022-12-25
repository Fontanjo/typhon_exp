import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return VAE3()


class VAE3(torch.nn.Module):
    def __init__(self):
        super().__init__()

        ### Convolutional section
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=8, kernel_size=9, stride=1, padding=4),
            nn.MaxPool2d(kernel_size=9, stride=2, padding=4),
            nn.BatchNorm2d(8),
            nn.ReLU(True),

            nn.Conv2d(8, 16, kernel_size=5, stride=1, padding=2),
            nn.MaxPool2d(kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(True),

            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(True),

            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(True),

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            nn.Flatten(),
        )

        self.linear_mu = nn.Linear(512 * 2 * 2, 512)
        self.linear_var = nn.Linear(512 * 2 * 2, 512)


    def sample(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        s = mu + eps*std
        return s

    def forward(self, x):
        # Encode
        x = self.encoder_cnn(x)
        # Get mu and sigma
        mu = self.linear_mu(x)
        var = self.linear_var(x)
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

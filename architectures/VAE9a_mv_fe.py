import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return VAE8(dropout)


class VAE8(torch.nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

        ### Convolutional section
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1, stride=1),
            nn.BatchNorm2d(1),
            nn.ReLU(True),

            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),    # shape = (batch, 1, 105, 80)
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=2),    # shape = (batch, 1, 52, 40)
            nn.Flatten(), # shape = (batch, 8400)

            nn.Linear(52*40, 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout),

            nn.Linear(128, 128),
            nn.ELU(),
            nn.Dropout(p=self.dropout)
        )

        self.mu = nn.Linear(128, 128)
        self.var = nn.Linear(128, 128)


    def sample(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        s = mu + eps*std
        return s

    def forward(self, x):
        # Encode
        x = self.encoder(x)
        # Get mu and sigma
        mu = self.mu(x)
        var = self.var(x)
        # Sample
        x = self.sample(mu, var)
        if x.isnan().any():
            print('Nan generated, stop')
            exit(-1)
        return x, mu, var


    def __iter__(self):
        return iter([
            self.encoder,
            self.mu,
            self.var
        ])

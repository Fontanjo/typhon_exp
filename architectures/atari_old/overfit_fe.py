import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return VAE3()


class VAE3(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def sample(self, mu, var):
        std = torch.exp(0.5*var)
        eps = torch.randn_like(std)
        s = mu + eps*std
        return s

    def forward(self, x):
        if torch.cuda.is_available():
            device = f'cuda:{torch.cuda.device_of(x).idx}'
        else:
            device = 'cpu'
        batch_size = x.shape[0]
        # Keep mu and var fixed
        mu = torch.rand([batch_size, 512, 1, 1]).to(device) # Send to same cuda device of input
        var = torch.rand([batch_size, 512, 1, 1]).to(device)
        # Sample
        x = self.sample(mu, var)
        return x, mu, var


    def __iter__(self):
        return iter([
            self.encoder_cnn,
            self.conv_mu,
            self.conv_var
        ])

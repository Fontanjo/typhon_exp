import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return Autoencoder_dm()

class Autoencoder_dm(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.decoder_cnn = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1),
            nn.Sigmoid(),
        )
        # out: (Bx64x210x160)

        self.unf = torch.nn.Unflatten(2, (210, 160))
        # self.sig = nn.Sigmoid()



    def forward(self, inputs):
        # Split back in dm
        top_k, idx = inputs
        # Create a 0-tensor
        batch_size = top_k.shape[0]
        res = torch.zeros(batch_size, 1, 210 * 160).to('cuda:6')
        # Add values at the right place
        x = res.scatter(2, idx, top_k)

        # Back to original shape
        # x = torch.unflatten(x, 2, (210, 160))
        x = self.unf(x)
        # Repaint sprites
        x = self.decoder_cnn(x)
        # Bound in [0,1]
        # x = self.sig(x)
        return x


    def __iter__(self):
        return iter([
            self.decoder_cnn,
            self.unf,
            # self.sig,
        ])

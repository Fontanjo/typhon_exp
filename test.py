import torch
import torch.nn as nn



class LocalityConvertor(torch.nn.Module):
    def __init__(nfeatures, dilation=1,):
        super().__init__()
        self.nfeatures = nfeatures
        self.dilation = dilation

        self.mp = torch.nn.MaxPool3d(kernel_size=(self.nfeatures, 1, 1),
                                      stride=1,
                                      padding=0,
                                      dilation=self.dilation,
                                      return_indices=True,
                                      ceil_mode=False)


        # Flatten uses row first, then columns, then depth

    def forward(self, x):
        out, idx = self.mp(x) # (Bx1x210x160) x2

        b, f, h, w = out.shape

        tens = torch.Tensor((b, h*w, 4))

        for nb in range(b):
            for nf in range(f):  # Only 1 feature
                for nh in range(h):
                    for nw in range(w):
                        tens[nb, nh*nw, 0] = out[nb, nf, nh, nw]
                        tens[nb, nh*nw, 1] = idx[nb, nf, nh, nw]
                        tens[nb, nh*nw, 2] = nh
                        tens[nb, nh*nw, 3] = nw

        # Sort
        toch.sort(tens, lambda x: x[:,:,0]) # check if exist
        # out, idx = torch.topk(tens[:,:,0], 64)




    def backward(self):
        pass


    def unflatten_coords(self, idx, shape):
        rest = shape
        coords = []
        divisor = 1
        for *rest, curr in rest:
            # if not rest: break
            coors.append(idx // divisor % curr) # or modify idx (idx // divisor) each loop
            divisor *= curr
        return coords


a = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),

        nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ELU(),

        # TODO: Check dilation
        # out: 1x210x160
        # 210x160
        # LocalityConvertor(kernel_size=(64, 3, 3), stride=3, dilation=1, ceil_mode=True),
        # torch.nn.MaxPool3d(kernel_size=(64, 3, 3), stride=3, dilation=1, return_indices=True, ceil_mode=True), # 1x70x54


#         nn.BatchNorm2d(3),  # Rescales image-wise coordinate [0,1]
    )

inp = torch.ones((1, 3, 210, 160))

out = a(inp)

out2 = torch.flatten(out[0], start_dim=2)

out3 = torch.topk(out2, 128)

import IPython

print(out.shape)

import numpy as np

r = np.arange(24).reshape((2, 3, 4))

rest = r.shape

def unflatten_coords(idx, shape=r.shape):
    rest = shape
    coords = []
    divisor = 1
    for *rest, curr in rest:
        # if not rest: break
        coors.append(idx // divisor % curr) # or modify idx (idx // divisor) each loop
        divisor *= curr
    return coords

# np.array([1,2,3],
        # )

# print(r.flatten())

IPython.embed()

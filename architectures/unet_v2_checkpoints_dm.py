# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
# Slightly modified to always output 1 value for encoder blocks and take 1 input value for deconder blocks

import torch
import torch.nn as nn
from architectures.unet_blocks import conv_block, decoder_block

def get_block(dropout, in_chanels=1):
    return Unet_container(in_chanels=in_chanels)


class Unet_container(nn.Module):
    def __init__(self, in_chanels):
        super().__init__()
        """ Decoder """
        self.d4 = decoder_block(128, 64)
        """ Classifier """
        self.output = nn.Conv2d(64, 1, kernel_size=1, padding=0)
        """ Sigmoid """
        self.sig = nn.Sigmoid()



    def forward(self, inputs):
        d3, s1 = inputs
        d4 = self.d4([d3, s1])

        out = self.output(d4)
        sig = self.sig(out)

        return sig

    def __iter__(self):
        return iter([self.d4])

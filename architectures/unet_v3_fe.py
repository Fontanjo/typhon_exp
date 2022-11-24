# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
# Slightly modified to always output 1 value for encoder blocks and take 1 input value for deconder blocks

import torch
import torch.nn as nn
from architectures.unet_blocks import conv_block, encoder_block, decoder_block

def get_block(dropout, in_chanels=1):
    return Unet_container(in_chanels=in_chanels)


class Unet_container(nn.Module):
    def __init__(self, in_chanels):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_chanels, 64)
        self.e2 = encoder_block(64, 128)
        self.e3 = encoder_block(128, 256)
        self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        self.b = conv_block(512, 1024)

        """ Decoder """
        self.d1 = decoder_block(1024, 512)
        self.d2 = decoder_block(512, 256)



    def forward(self, inputs):
        inputs = inputs[:,:,:256,:256] # Reshape, since in deconding phase will be a power of 2 and must be equal
        """ Encoder """
        s1, p1 = self.e1(inputs)
        s2, p2 = self.e2(p1)
        s3, p3 = self.e3(p2)
        s4, p4 = self.e4(p3)

        """ Bottleneck """
        b = self.b(p4)

        """ Decoder """
        d1 = self.d1([b, s4])
        d2 = self.d2([d1, s3])

        return [d2, s2, s1]

    def __iter__(self):
        return iter([self.e1, self.e2, self.e3, self.e4, self.b, self.d2, self.d1])

# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
import torch
import torch.nn as nn
from architectures.unet_blocks_old import conv_block, encoder_block, decoder_block

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
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)


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
        d1 = self.d1(b, s4)
        d2 = self.d2(d1, s3)
        d3 = self.d3(d2, s2)
        d4 = self.d4(d3, s1)

        # print("####################################")
        # print("####################################")
        # print(f'{s1.shape}  ===  {p1.shape}')
        # print(f'{s2.shape}  ===  {p4.shape}')
        # print(f'{s3.shape}  ===  {p4.shape}')
        # print(f'{s4.shape}  ===  {p4.shape}')
        # print("####################################")
        # print("####################################")
        # print(f'{d1.shape}')
        # print(f'{d2.shape}')
        # print(f'{d3.shape}')
        # print(f'{d4.shape}')
        # print("####################################")
        # print("####################################")

        return d4

    def __iter__(self):
        return iter([self.e1, self.e2, self.e3, self.e4, self.b, self.d4, self.d3, self.d2, self.d1])

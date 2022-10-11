# https://medium.com/analytics-vidhya/unet-implementation-in-pytorch-idiot-developer-da40d955f201
import torch
import torch.nn as nn

def get_block(dropout, in_chanels=1):
    return Unet_container(in_chanels=in_chanels)


class conv_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_c)

        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_c)

        self.relu = nn.ReLU()


    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        return x


class encoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = conv_block(in_c, out_c)
        self.pool = nn.MaxPool2d((2, 2))

    def forward(self, inputs):
        x = self.conv(inputs)
        p = self.pool(x)
        return x, p


class decoder_block(nn.Module):
    def __init__(self, in_c, out_c):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0)
        self.conv = conv_block(out_c+out_c, out_c)

    def forward(self, inputs, skip):
        x = self.up(inputs)
        x = torch.cat([x, skip], axis=1)
        x = self.conv(x)
        return x


class Unet_container(nn.Module):
    def __init__(self, in_chanels):
        super().__init__()
        """ Encoder """
        self.e1 = encoder_block(in_chanels, 64)
        self.e2 = encoder_block(64, 128)
        # self.e3 = encoder_block(128, 256)
        # self.e4 = encoder_block(256, 512)

        """ Bottleneck """
        # self.b = conv_block(512, 1024)
        self.b = conv_block(128, 256)

        """ Decoder """
        # self.d1 = decoder_block(1024, 512)
        # self.d2 = decoder_block(512, 256)
        self.d3 = decoder_block(256, 128)
        self.d4 = decoder_block(128, 64)


    # TODO: check if can make bigger (or adapt size to input!)
    # TODO: remove unused parameters
    def forward(self, inputs):
        inputs = inputs[:,:,:64,:64] # Reshape, since in deconding phase will be a power of 2 and must be equal
        """ Encoder """
        # print(f"in: ##### {inputs.size()} #####")
        s1, p1 = self.e1(inputs)
        # print(f"e1: ##### {s1.size()} , {p1.size()} #####")
        s2, p2 = self.e2(p1)
        # print(f"e2: ##### {s2.size()} , {p2.size()} #####")
        # s3, p3 = self.e3(p2)
        # s4, p4 = self.e4(p3)

        """ Bottleneck """
        # b = self.b(p4)
        b = self.b(p2)
        # print(f"b:  ##### {b.size()} #####")

        """ Decoder """
        # d1 = self.d1(b, s4)
        # d2 = self.d2(d1, s3)
        # d3 = self.d3(d2, s2)
        d3 = self.d3(b, s2)
        # print(f"d3: ##### {d3.size()} #####")
        d4 = self.d4(d3, s1)
        # print(f"d4: ##### {d4.size()} #####")

        return d4

    def __iter__(self):
        # return iter([self.e1, self.e2, self.e3, self.e4, self.b, self.d4, self.d3, self.d2, self.d1])
        return iter([self.e1, self.e2, self.b, self.d3, self.d4])

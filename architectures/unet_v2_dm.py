import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return Unet_container(in_channels=in_channels)


class Unet_container(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        """ Ascending ('decoder') part """
        """ Block 3 """
        self.upconv_3 =     nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2, padding=0)

        self.conv_d_3_1 =   nn.Conv2d(128 + 128, 128, kernel_size=3, padding=1)
        self.bn_d_3_1 =     nn.BatchNorm2d(128)
        self.relu_d_3_1 =   nn.ReLU()

        self.conv_d_3_2 =   nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_d_3_2 =     nn.BatchNorm2d(128)
        self.relu_d_3_2 =   nn.ReLU()


        """ Block 4 """
        self.upconv_4 =     nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2, padding=0)

        self.conv_d_4_1 =   nn.Conv2d(64 + 64, 64, kernel_size=3, padding=1)
        self.bn_d_4_1 =     nn.BatchNorm2d(64)
        self.relu_d_4_1 =   nn.ReLU()

        self.conv_d_4_2 =   nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_d_4_2 =     nn.BatchNorm2d(64)
        self.relu_d_4_2 =   nn.ReLU()


        """ Final part """
        self.conv_f =       nn.Conv2d(64, 1, kernel_size=1, padding=0),
        self.sigmoid =      nn.Sigmoid()




    def forward(self, inputs):
        # Extract values
        x, s2, s1 = inputs

        """ Ascending ('decoder') part """
        """ Block 3 """
        x = self.upconv_3(x)

        x = torch.cat([x, s2], axis=1)

        x = self.conv_d_3_1(x)
        x = self.bn_d_3_1(x)
        x = self.relu_d_3_1(x)

        x = self.conv_d_3_2(x)
        x = self.bn_d_3_2(x)
        x = self.relu_d_3_2(x)


        """ Block 4 """
        x = self.upconv_4(x)

        x = torch.cat([x, s1], axis=1)

        x = self.conv_d_4_1(x)
        x = self.bn_d_4_1(x)
        x = self.relu_d_4_1(x)

        x = self.conv_d_4_2(x)
        x = self.bn_d_4_2(x)
        x = self.relu_d_4_2(x)

        """ Final part """
        x = self.conv_f(x)
        x = self.sigmoid(x)

        return x


    def __iter__(self):
        return iter([
            self.upconv_3,
            self.conv_d_3_1,
            self.bn_d_3_1,
            self.relu_d_3_1,
            self.conv_d_3_2,
            self.bn_d_3_2,
            self.relu_d_3_2,

            self.upconv_4,
            self.conv_d_4_1,
            self.bn_d_4_1,
            self.relu_d_4_1,
            self.conv_d_4_2,
            self.bn_d_4_2,
            self.relu_d_4_2,

            self.conv_f,
            self.sigmoid
        ])

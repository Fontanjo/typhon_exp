import torch
import torch.nn as nn

def get_block(dropout, in_channels=1):
    return Unet_container(in_channels=in_channels)


class Unet_container(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        """ Descending ('encoder') part """
        """ Block 1 """
        self.conv_e_1_1 =   nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn_e_1_1 =     nn.BatchNorm2d(64)
        self.relu_e_1_1 =   nn.ReLU()

        self.conv_e_1_2 =   nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn_e_1_2 =     nn.BatchNorm2d(64)
        self.relu_e_1_2 =   nn.ReLU()

        self.pool_e_1 =     nn.MaxPool2d((2, 2))

        """ Block 2 """
        self.conv_e_2_1 =   nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn_e_2_1 =     nn.BatchNorm2d(128)
        self.relu_e_2_1 =   nn.ReLU()

        self.conv_e_2_2 =   nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn_e_2_2 =     nn.BatchNorm2d(128)
        self.relu_e_2_2 =   nn.ReLU()

        self.pool_e_2 =     nn.MaxPool2d((2, 2))

        """ Block 3 """
        self.conv_e_3_1 =   nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn_e_3_1 =     nn.BatchNorm2d(256)
        self.relu_e_3_1 =   nn.ReLU()

        self.conv_e_3_2 =   nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_e_3_2 =     nn.BatchNorm2d(256)
        self.relu_e_3_2 =   nn.ReLU()

        self.pool_e_3 =     nn.MaxPool2d((2, 2))


        """ Block 4 """
        self.conv_e_4_1 =   nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn_e_4_1 =     nn.BatchNorm2d(512)
        self.relu_e_4_1 =   nn.ReLU()

        self.conv_e_4_2 =   nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_e_4_2 =     nn.BatchNorm2d(512)
        self.relu_e_4_2 =   nn.ReLU()

        self.pool_e_4 =     nn.MaxPool2d((2, 2))


        """ Central ('bottleneck') part """
        self.conv_b_1 =     nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.bn_b_1 =       nn.BatchNorm2d(1024)
        self.relu_b_1 =     nn.ReLU()

        self.conv_b_2 =     nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.bn_b_2 =       nn.BatchNorm2d(1024)
        self.relu_b_2 =     nn.ReLU()


        """ Ascending ('decoder') part """
        """ Block 1 """
        self.upconv_1 =     nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2, padding=0)

        self.conv_d_1_1 =   nn.Conv2d(512 + 512, 512, kernel_size=3, padding=1)
        self.bn_d_1_1 =     nn.BatchNorm2d(512)
        self.relu_d_1_1 =   nn.ReLU()

        self.conv_d_1_2 =   nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn_d_1_2 =     nn.BatchNorm2d(512)
        self.relu_d_1_2 =   nn.ReLU()


        """ Block 2 """
        self.upconv_2 =     nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2, padding=0)

        self.conv_d_2_1 =   nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        self.bn_d_2_1 =     nn.BatchNorm2d(256)
        self.relu_d_2_1 =   nn.ReLU()

        self.conv_d_2_2 =   nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn_d_2_2 =     nn.BatchNorm2d(256)
        self.relu_d_2_2 =   nn.ReLU()


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




    def forward(self, inputs):
        """ Descending ('encoder') part """
        """ Block 1 """
        x = self.conv_e_1_1(inputs)
        x = self.bn_e_1_1(x)
        x = self.relu_e_1_1(x)

        x = self.conv_e_1_2(x)
        x = self.bn_e_1_2(x)
        x = self.relu_e_1_2(x)

        s1 = x # Store first skip

        x = self.pool_e_1(x)


        """ Block 2 """
        x = self.conv_e_2_1(x)
        x = self.bn_e_2_1(x)
        x = self.relu_e_2_1(x)

        x = self.conv_e_2_2(x)
        x = self.bn_e_2_2(x)
        x = self.relu_e_2_2(x)

        s2 = x # Store second skip

        x = self.pool_e_2(x)


        """ Block 3 """
        x = self.conv_e_3_1(x)
        x = self.bn_e_3_1(x)
        x = self.relu_e_3_1(x)

        x = self.conv_e_3_2(x)
        x = self.bn_e_3_2(x)
        x = self.relu_e_3_2(x)

        s3 = x # Store third skip

        x = self.pool_e_3(x)


        """ Block 4 """
        x = self.conv_e_4_1(x)
        x = self.bn_e_4_1(x)
        x = self.relu_e_4_1(x)

        x = self.conv_e_4_2(x)
        x = self.bn_e_4_2(x)
        x = self.relu_e_4_2(x)

        s4 = x # Store fourth skip

        x = self.pool_e_4(x)


        """ Central ('bottleneck') part """
        x = self.conv_b_1(x)
        x = self.bn_b_1(x)
        x = self.relu_b_1(x)

        x = self.conv_b_2(x)
        x = self.bn_b_2(x)
        x = self.relu_b_2(x)


        """ Ascending ('decoder') part """
        """ Block 1 """
        x = self.upconv_1(x)

        x = torch.cat([x, s4], axis=1)

        x = self.conv_d_1_1(x)
        x = self.bn_d_1_1(x)
        x = self.relu_d_1_1(x)

        x = self.conv_d_1_2(x)
        x = self.bn_d_1_2(x)
        x = self.relu_d_1_2(x)


        """ Block 2 """
        x = self.upconv_2(x)

        x = torch.cat([x, s3], axis=1)

        x = self.conv_d_2_1(x)
        x = self.bn_d_2_1(x)
        x = self.relu_d_2_1(x)

        x = self.conv_d_2_2(x)
        x = self.bn_d_2_2(x)
        x = self.relu_d_2_2(x)


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

        return x


    def __iter__(self):
        return iter([
            self.conv_e_1_1,
            self.bn_e_1_1,
            self.relu_e_1_1,
            self.conv_e_1_2,
            self.bn_e_1_2,
            self.relu_e_1_2,
            self.pool_e_1,

            self.conv_e_2_1,
            self.bn_e_2_1,
            self.relu_e_2_1,
            self.conv_e_2_2,
            self.bn_e_2_2,
            self.relu_e_2_2,
            self.pool_e_2,

            self.conv_e_3_1,
            self.bn_e_3_1,
            self.relu_e_3_1,
            self.conv_e_3_2,
            self.bn_e_3_2,
            self.relu_e_3_2,
            self.pool_e_3,

            self.conv_e_4_1,
            self.bn_e_4_1,
            self.relu_e_4_1,
            self.conv_e_4_2,
            self.bn_e_4_2,
            self.relu_e_4_2,
            self.pool_e_4,

            self.conv_b_1,
            self.bn_b_1,
            self.relu_b_1,
            self.conv_b_2,
            self.bn_b_2,
            self.relu_b_2,

            self.upconv_1,
            self.conv_d_1_1,
            self.bn_d_1_1,
            self.relu_d_1_1,
            self.conv_d_1_2,
            self.bn_d_1_2,
            self.relu_d_1_2,

            self.upconv_2,
            self.conv_d_2_1,
            self.bn_d_2_1,
            self.relu_d_2_1,
            self.conv_d_2_2,
            self.bn_d_2_2,
            self.relu_d_2_2,

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
            self.relu_d_4_2
        ])

import torch
from torch import nn


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding="same",
                bias=False,
                groups=in_channels,
            ),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.seq(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            DepthwiseConv(in_channels, out_channels),
            DepthwiseConv(out_channels, out_channels),
        )

    def forward(self, x):
        return self.seq(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seq = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x):
        return self.seq(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2)
        self.contract = DoubleConv(in_channels, out_channels)

    def forward(self, x_down, x):
        x_up = self.up(x_down)

        pad_r = x.shape[3] - x_up.shape[3]
        pad_b = x.shape[2] - x_up.shape[2]
        x_up = nn.functional.pad(x_up, (0, pad_r, 0, pad_b))
        x_up = torch.cat((x_up, x), dim=1)

        return self.contract(x_up)


class DetectionModel(nn.Module):
    """
    Text detection model.

    This uses a U-Net-like architecture. See https://arxiv.org/abs/1505.04597.

    It expects a greyscale image as input and outputs a text/not-text
    segmentation mask.
    """

    def __init__(self):
        super().__init__()

        # Number of feature channels at each size level in the network.
        #
        # The U-Net paper uses 64 for the first level. This model uses a
        # reduced scale to cut down the parameter count.
        #
        # depth_scale = [64, 128, 256, 512, 1024]
        depth_scale = [16, 32, 64, 128, 256]

        self.in_conv = DoubleConv(1, depth_scale[0])

        self.down1 = Down(depth_scale[0], depth_scale[1])
        self.down2 = Down(depth_scale[1], depth_scale[2])
        self.down3 = Down(depth_scale[2], depth_scale[3])
        self.down4 = Down(depth_scale[3], depth_scale[4])

        self.up4 = Up(depth_scale[4], depth_scale[3])
        self.up3 = Up(depth_scale[3], depth_scale[2])
        self.up2 = Up(depth_scale[2], depth_scale[1])
        self.up1 = Up(depth_scale[1], depth_scale[0])

        self.final = nn.Sequential(
            nn.Conv2d(depth_scale[0], 2, kernel_size=1, padding="same"), nn.Sigmoid()
        )

    def forward(self, x):
        x = self.in_conv(x)

        x_down_1 = self.down1(x)
        x_down_2 = self.down2(x_down_1)
        x_down_3 = self.down3(x_down_2)
        x_down_4 = self.down4(x_down_3)

        x_up_4 = self.up4(x_down_4, x_down_3)
        x_up_3 = self.up3(x_up_4, x_down_2)
        x_up_2 = self.up2(x_up_3, x_down_1)
        x_up_1 = self.up1(x_up_2, x)

        return self.final(x_up_1)

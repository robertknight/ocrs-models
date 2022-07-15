import torch
from torch import nn


class DepthwiseConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
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

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            DepthwiseConv(in_channels, out_channels),
            DepthwiseConv(out_channels, out_channels),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class Down(nn.Module):
    """
    Downscaling module in U-Net model.
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.seq = nn.Sequential(
            DoubleConv(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2),
        )

    def forward(self, x: torch.Tensor):
        return self.seq(x)


class Up(nn.Module):
    """
    Upscaling module in U-Net model.

    This upscales a feature map from the previous "up" stage of the network
    and combines it with a feature map coming across from a "down" stage.
    """

    def __init__(self, in_up_channels: int, in_cross_channels: int, out_channels: int):
        """
        :param in_up_channels: Channels in inputs to be upscaled
        :param in_cross_channels: Channels in inputs to be concatenated with upscaled input
        """
        super().__init__()

        self.up = nn.ConvTranspose2d(
            in_up_channels, out_channels, kernel_size=3, stride=2
        )
        self.contract = DoubleConv(out_channels + in_cross_channels, out_channels)

    def forward(self, x_to_upscale: torch.Tensor, x: torch.Tensor):
        upscaled = self.up(x_to_upscale)
        pad_r = x.shape[3] - upscaled.shape[3]
        pad_b = x.shape[2] - upscaled.shape[2]
        upscaled = nn.functional.pad(upscaled, (0, pad_r, 0, pad_b))

        combined = torch.cat((upscaled, x), dim=1)

        return self.contract(combined)


def binarize(pred_mask: torch.Tensor, thresh_mask: torch.Tensor, k: float = 50):
    """
    Differentiable Binarization function, from https://arxiv.org/abs/1911.08947.

    :param pred_mask: B1HW tensor of binary class probabilities
    :param thresh_mask: B1HW tensor of thresholds
    :param k: Amplifying factor (see eqn 2 of paper). Defaults to the value of
      50 given in the paper.
    :return: B1HW tensor of binary class probabilities
    """
    return (k * (pred_mask - thresh_mask)).sigmoid()


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
        depth_scale = [8, 16, 32, 32, 64, 128, 256]
        self.depth_scale = depth_scale

        self.in_conv = DoubleConv(1, depth_scale[0])

        self.down = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.down.append(Down(depth_scale[i], depth_scale[i + 1]))

        self.up = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.up.append(Up(depth_scale[i + 1], depth_scale[i], depth_scale[i]))

        self.prob_head = nn.Sequential(
            nn.Conv2d(depth_scale[0], depth_scale[0], kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(depth_scale[0], depth_scale[0], kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(depth_scale[0], 1, kernel_size=1, padding="same"),
            nn.Sigmoid(),
        )
        self.thresh_head = nn.Sequential(
            nn.Conv2d(depth_scale[0], depth_scale[0], kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(depth_scale[0], depth_scale[0], kernel_size=1, padding="same"),
            nn.ReLU(),
            nn.Conv2d(depth_scale[0], 1, kernel_size=1, padding="same"),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(x)

        x_down = []
        for i, down_op in enumerate(self.down):
            prev_down = x if i == 0 else x_down[-1]
            x_down.append(down_op(prev_down))

        x_up = x_down[-1]
        for i, up_op in reversed(list(enumerate(self.up))):
            x_up = up_op(x_up, x if i == 0 else x_down[i - 1])

        # TODO - Skip prediction of threshold mask and binarization step at
        # inference time.

        prob_mask = self.prob_head(x_up)
        thresh_mask = self.thresh_head(x_up)

        k_factor = 10
        # k_factor = 50.
        bin_mask = binarize(prob_mask, thresh_mask, k=k_factor)
        return torch.cat([prob_mask, bin_mask, thresh_mask], dim=1)

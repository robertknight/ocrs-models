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
                # Equivalent to "same" padding for a kernel size of 3.
                # PyTorch's ONNX export doesn't support the "same" keyword.
                padding=(1, 1),
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

        # `x_to_upscale` is assumed to be half the resolution of `x`. When
        # it is conv-transposed the result can be 1 pixel taller/wider than `x`.
        # Trim the right/bottom edges to make the images the same size.
        upscaled = upscaled[:, :, 0 : x.shape[2], 0 : x.shape[3]]

        combined = torch.cat((upscaled, x), dim=1)
        return self.contract(combined)


class DetectionModel(nn.Module):
    """
    Text detection model.

    This uses a U-Net-like architecture. See https://arxiv.org/abs/1505.04597.

    It expects a greyscale image as input and outputs a text/not-text
    segmentation mask of the same size.

    The input is first downsampled by `stem_factor` using a strided convolution,
    and the U-Net operates at that reduced resolution. The output is upsampled
    back to the input resolution using a transposed convolution, which
    predicts a `stem_factor x stem_factor` patch of the mask for each position.
    Avoiding full-resolution feature maps makes the model much faster to train
    and run than a U-Net which operates at the input resolution.
    """

    def __init__(self):
        super().__init__()

        stem_factor = 4

        # Number of feature channels at each size level in the network,
        # starting at 1/stem_factor of the input resolution.
        #
        # The U-Net paper uses 64 for the first level. This model uses a
        # reduced scale to cut down the parameter count.
        #
        # depth_scale = [64, 128, 256, 512, 1024]
        depth_scale = [64, 64, 128, 128, 256]
        self.depth_scale = depth_scale

        self.stem = nn.Sequential(
            nn.Conv2d(
                1,
                depth_scale[0],
                kernel_size=stem_factor,
                stride=stem_factor,
                bias=False,
            ),
            nn.BatchNorm2d(depth_scale[0]),
            nn.ReLU(),
        )
        self.in_conv = DoubleConv(depth_scale[0], depth_scale[0])

        self.down = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.down.append(Down(depth_scale[i], depth_scale[i + 1]))

        self.up = nn.ModuleList()
        for i in range(len(depth_scale) - 1):
            self.up.append(Up(depth_scale[i + 1], depth_scale[i], depth_scale[i]))

        n_masks = 1  # Output masks to generate
        out_upscale = nn.ConvTranspose2d(
            depth_scale[0], n_masks, kernel_size=stem_factor, stride=stem_factor
        )
        # Zero-initialize the output layer so that the model initially predicts
        # 0.5 everywhere. With PyTorch's default initialization, the initial
        # predictions are noisy and training can get stuck predicting a
        # constant value, depending on the random seed.
        nn.init.zeros_(out_upscale.weight)
        nn.init.zeros_(out_upscale.bias)
        self.out_conv = nn.Sequential(out_upscale, nn.Sigmoid())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_conv(self.stem(x))

        x_down: list[torch.Tensor] = []
        for i, down_op in enumerate(self.down):
            prev_down = x if i == 0 else x_down[-1]
            x_down.append(down_op(prev_down))

        x_up = x_down[-1]
        for i, up_op in reversed(list(enumerate(self.up))):
            x_up = up_op(x_up, x if i == 0 else x_down[i - 1])

        return self.out_conv(x_up)


class RecognitionModel(nn.Module):
    """
    Text recognition model.

    This takes NCHW images of text lines as input and outputs a sequence of
    character predictions as a (W/4)xNxC tensor.

    The input images must be greyscale and have a fixed height of 64.

    The result is a sequence 1/4 the length of the input, where the `C` dim is
    the 1-based index of the character in the alphabet used to train the model.
    The value 0 is reserved for the blank character. The result sequence needs
    to be postprocessed with CTC decoding (eg. greedy or beam search) to recover
    the recognized character sequence.

    The model follows the general structure of CRNN [1], consisting of
    convolutional layers to extract features, followed by a bidirectional RNN to
    predict the character sequence.

    [1] https://arxiv.org/abs/1507.05717
    """

    def __init__(self, alphabet: str):
        """
        Construct the model.

        :param alphabet: Alphabet of characters that the model will recognize
        """

        super().__init__()

        n_classes = len(alphabet) + 1

        self.conv = nn.Sequential(
            nn.Conv2d(
                1,
                32,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                # Don't use biases for Conv2d when followed directly by batch norm,
                # per https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm.
                bias=False,
            ),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
            ),
            nn.ReLU(),
            nn.Conv2d(
                128,
                128,
                kernel_size=3,
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 1)),
            nn.Conv2d(
                128,
                128,
                kernel_size=(2, 2),
                padding=(1, 1),  # "same" padding
                bias=False,
            ),
            nn.BatchNorm2d(128),
            nn.AvgPool2d(kernel_size=(4, 1)),
        )

        self.gru = nn.GRU(128, 256, bidirectional=True, num_layers=2)

        self.output = nn.Sequential(
            nn.Linear(512, n_classes),
            # nb. We use `LogSoftmax` here because `torch.nn.CTCLoss` expects log probs
            nn.LogSoftmax(dim=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)

        # Reshape from NCHW to WNCH
        x = torch.permute(x, (3, 0, 1, 2))

        # Combine last two dims to get WNx(CH)
        x = torch.reshape(x, (x.shape[0], x.shape[1], -1))

        # Disable autocast here as PyTorch doesn't support GRU with bfloat16.
        with torch.autocast(x.device.type, enabled=False):
            x, _ = self.gru(x.float())

        return self.output(x)

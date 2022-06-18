from torch import nn


class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(1, 3, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(3, 6, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(6, 6, kernel_size=3, padding="same"),
            nn.ReLU(),
            nn.Conv2d(6, 1, kernel_size=3, padding="same"),
        )

    def forward(self, x):
        return self.layers(x)

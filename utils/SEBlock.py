from torch import nn

class SEBlock(nn.Module):
    def __init__(self, in_channels) -> None:
        super(SEBlock, self).__init__()
        self.in_channels = in_channels
        self.layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, in_channels*3),
            nn.ReLU(),
            nn.Linear(in_channels*3, in_channels),
            nn.Sigmoid()
                )

    def forward(self, x):
        # x: (B, C, H, W)
        b, c, _, _ = x.shape
        result = self.layers(x).view(b,c,1,1)
        return x*result


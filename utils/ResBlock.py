from torch import nn
from utils.ConvBlock import ConvBlock


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, downsample=False) -> None:
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_layers = 4
        self.features_ratio = (out_channels - in_channels) // self.num_layers
        self.layers = nn.Sequential(
                ConvBlock(in_channels=in_channels, out_channels=self.in_channels + (self.features_ratio), padding=0, stride=2 if downsample else 1, kernel_size=1),
                ConvBlock(in_channels=self.in_channels + (self.features_ratio), out_channels= self.in_channels + (self.features_ratio*2), padding=1, stride=1, kernel_size=3),
                ConvBlock(in_channels=self.in_channels + (self.features_ratio*2), out_channels=self.in_channels + (self.features_ratio*3), padding=0, stride=1, kernel_size=1),
                ConvBlock(in_channels= self.in_channels + (self.features_ratio*3), out_channels=out_channels, padding=1, stride=1, kernel_size=3, activate=False)
                )
        self.identity = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0,stride=2 if downsample else 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        output = self.layers(x)
        x = self.identity(x)
        return self.relu(x+output)


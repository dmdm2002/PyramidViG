import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class Stem(nn.Module):
    def __init__(self, in_channel=3, out_channel=768, act='relu'):
        super(Stem, self).__init__()
        if act == 'relu':
            activation = nn.ReLU
        else:
            activation = nn.GELU

        self.convs = nn.Sequential(
            nn.Conv2d(in_channel, out_channel//2, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel//2),
            activation(),

            nn.Conv2d(out_channel//2, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
            activation(),

            nn.Conv2d(out_channel, out_channel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x = self.convs(x)

        return x


class Downsample(nn.Module):
    def __init__(self, in_channel=3, out_channel=768):
        super(Downsample, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        x = self.conv(x)

        return x
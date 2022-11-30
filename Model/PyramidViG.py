import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Model.PyramidModules import *
from Model.ViGModules import ViG_Block

import torchsummary


class Pyramid_ViG(nn.Module):
    def __init__(self):
        super(Pyramid_ViG, self).__init__()
        repeat = [2, 2, 16, 2]
        dimensions = [96, 192, 384, 768]
        self.stem = Stem(in_channel=3, out_channel=dimensions[0])

        vig_1 = [ViG_Block(in_channel=dimensions[0], k=9, dilation=1) for _ in range(repeat[0])]

        self.vig_1 = nn.Sequential(*vig_1)
        self.down_1 = Downsample(in_channel=dimensions[0], out_channel=dimensions[1])

        vig_2 = [ViG_Block(in_channel=dimensions[1], k=9, dilation=1) for _ in range(repeat[1])]

        self.vig_2 = nn.Sequential(*vig_2)
        self.down_2 = Downsample(in_channel=dimensions[1], out_channel=dimensions[2])

        vig_3 = [ViG_Block(in_channel=dimensions[2], k=9, dilation=1) for _ in range(repeat[2])]

        self.vig_3 = nn.Sequential(*vig_3)
        self.down_3 = Downsample(in_channel=dimensions[2], out_channel=dimensions[3])

        vig_4 = [ViG_Block(in_channel=dimensions[3], k=9, dilation=1) for _ in range(repeat[3])]

        self.vig_4 = nn.Sequential(*vig_4)

        # self.head = nn.Sequential(
        #     nn.AdaptiveAvgPool2d((1, 1)),
        #     nn.Linear(dimensions[3] * 1, 1)
        # )
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(dimensions[3]*1, 2)

    def forward(self, x):
        x = self.stem(x)

        x = self.vig_1(x)
        x = self.down_1(x)

        x = self.vig_2(x)
        x = self.down_2(x)

        x = self.vig_3(x)
        x = self.down_3(x)

        x = self.vig_4(x)
        # x = self.head(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# model = Pyramid_ViG().cuda()
# torchsummary.summary(model, (3, 224, 224), device='cuda')
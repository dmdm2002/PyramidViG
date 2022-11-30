"""gcn_lib 출처 : https://github.com/lightaime/deep_gcns_torch"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from Model.gcn_lib.dense.torch_vertex import DynConv2d


class FFNModule(nn.Module):
    def __init__(self, in_channel, hidden_channel=None, drop_path=0.0):
        super(FFNModule, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channel, hidden_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_channel),
            nn.GELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        x = x + shortcut

        return x


class GrapherModule(nn.Module):
    def __init__(self, in_channel, hidden_channel, k=9, dilation=1, drop_path=0.0):
        super(GrapherModule, self).__init__()
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
        )

        """
        Graph Convolution 부분 --> 외부 코드 가져왔음
        gcn_lib 출처 : https://github.com/lightaime/deep_gcns_torch
        """
        self.gcn = nn.Sequential(
            DynConv2d(in_channel, hidden_channel, k, dilation, act=None),
            nn.BatchNorm2d(hidden_channel),
            nn.GELU(),
        )

        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_channel, in_channel, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(in_channel),
        )

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        """
        배치와 채널만 놔두고 나머지를 1d 형태로 변환
        --> graph 연산을 수행하기 위해서
        --> ex) [2, 96, 28, 28] -> [2, 96, 784, 1]
        """
        x = x.reshape(B, C, -1, 1).contiguous()
        shortcut = x
        x = self.fc1(x)
        x = self.gcn(x)
        x = self.fc2(x)
        x = self.drop_path(x)
        x = x + shortcut

        return x.reshape(B, C ,H, W)


class ViG_Block(nn.Module):
    def __init__(self, in_channel, k, dilation, drop_path=0.0):
        super(ViG_Block, self).__init__()
        self.grapher = GrapherModule(in_channel, in_channel*2, k, dilation, drop_path)
        self.ffn = FFNModule(in_channel, in_channel*4, drop_path)

    def forward(self, x):
        x = self.grapher(x)
        x = self.ffn(x)

        return x
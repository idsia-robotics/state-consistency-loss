
import torch
import numpy as np
import torch.nn as nn
from ConvBlock import ConvBlock

np.set_printoptions(threshold=np.inf)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class PenguiNet(nn.Module):
    def __init__(self, block, layers, isGray=False):
        super(PenguiNet, self).__init__()

        if isGray == True:
            self.name = "PenguiNet"
        else:
            self.name = "PenguiNetRGB"
        self.inplanes = 32
        self.dilation = 1
        self._norm_layer = nn.BatchNorm2d

        self.groups = 1
        self.base_width = 64
        if isGray == True:
            self.conv = nn.Conv2d(
                1, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)
        else:
            self.conv = nn.Conv2d(
                3, self.inplanes, kernel_size=5, stride=2, padding=2, bias=False)

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.bn = nn.BatchNorm2d(self.inplanes)
        self.relu1 = nn.ReLU()

        self.layer1 = ConvBlock(32, 32, stride=2)
        self.layer2 = ConvBlock(32, 64, stride=2)
        self.layer3 = ConvBlock(64, 128, stride=2)

        self.dropout = nn.Dropout()

        fcSize = 1920
        self.fc = nn.Linear(fcSize, 4)

    def forward(self, x):

        conv5x5 = self.conv(x)
        btn = self.bn(conv5x5)
        relu1 = self.relu1(btn)
        max_pool = self.maxpool(relu1)

        l1 = self.layer1(max_pool)
        l2 = self.layer2(l1)
        l3 = self.layer3(l2)
        out = l3.flatten(1)

        out = self.dropout(out)
        out = self.fc(out)
        x = out[:, 0]
        y = out[:, 1]
        z = out[:, 2]
        phi = out[:, 3]
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        z = z.unsqueeze(1)
        phi = phi.unsqueeze(1)

        return out

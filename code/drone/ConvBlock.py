import logging
import numpy as np
import torch.nn as nn
np.set_printoptions(threshold=np.inf)
np.set_printoptions(suppress=True)


class ConvBlock(nn.Module):
    '''Pre-activation version of the BasicBlock.'''
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.relu2 = nn.ReLU()

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)

        return out


def printRes(layer, name):

    logger = logging.getLogger('')
    enable = logger.isEnabledFor(logging.INFO)
    yes = layer.requires_grad
    if (enable == True):
        n, c, h, w = layer.shape
        tmp = layer.reshape(-1)
        logging.info("{}={}".format(name, list(tmp.detach().numpy())))

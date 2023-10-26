import torch
from torch import nn


class ConvBlock(nn.Module):
    def __init__(self,size,ch_in,ch_out,stride=1,padding=0):
        super(ConvBlock,self).__init__()
        self.conv = nn.Conv2d(ch_in,ch_out,size,stride,padding,bias=False) #batch norm so bias = false , half precision
        self.batchNorm = nn.BatchNorm2d(ch_out)
        self.lrelu = nn.LeakyReLU(0.1)

    def forward(self,x):
        return self.lrelu(self.batchNorm(self.conv(x)))





class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super(GlobalAvgPool2d, self).__init__()

    @staticmethod
    def forward(x):
        return torch.mean(torch.flatten(x, start_dim=2), dim=2)


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)

        return img


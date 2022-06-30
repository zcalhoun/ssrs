"""
Copied from https://github.com/bohaohuang/mrs/blob/master/network/unet.py

"""


# Built-in

# Libs

# Pytorch
import torch
from torch import nn
from torch.nn import functional as F


class ConvDownSample(nn.Module):
    """
    This module defines conv-downsample block in the Unet
    conv->act->bn -> conv->act->bn -> (pool)
    """

    def __init__(self, in_chan, out_chan, pool=True):
        super(ConvDownSample, self).__init__()
        self.pool = pool
        self.conv_1 = nn.Conv2d(in_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=(0, 0))
        self.bn_1 = nn.BatchNorm2d(out_chan)
        self.bn_2 = nn.BatchNorm2d(out_chan)
        self.act = nn.PReLU()
        self.pool2d = nn.MaxPool2d((2, 2), (2, 2))

    def forward(self, x):
        x = self.bn_1(self.act(self.conv_1(x)))
        x = self.bn_2(self.act(self.conv_2(x)))
        if self.pool:
            return x, self.pool2d(x)
        else:
            return x


class UpSampleConv(nn.Module):
    """
    This module defines upsample-concat-conv block in the Unet
    interp->conv
             |
    crop->concat -> conv->act->bn -> conv->act->bn
    """

    def __init__(self, in_chan, out_chan, margin, conv_chan=None, pad=0):
        super(UpSampleConv, self).__init__()
        self.margin = margin
        if not conv_chan:
            conv_chan = out_chan
        self.up_conv = nn.Conv2d(
            in_chan, in_chan // 2, kernel_size=(3, 3), padding=(1, 1)
        )
        self.conv_1 = nn.Conv2d(conv_chan, out_chan, kernel_size=(3, 3), padding=pad)
        self.conv_2 = nn.Conv2d(out_chan, out_chan, kernel_size=(3, 3), padding=pad)
        self.bn_1 = nn.BatchNorm2d(out_chan)
        self.bn_2 = nn.BatchNorm2d(out_chan)
        self.act = nn.PReLU()

    def forward(self, x_1, x_2):
        x = F.interpolate(x_2, scale_factor=2)
        x = self.up_conv(x)
        if self.margin != 0:
            x_1 = x_1[:, :, self.margin : -self.margin, self.margin : -self.margin]
        x = torch.cat((x_1, x), 1)
        x = self.bn_1(self.act(self.conv_1(x)))
        x = self.bn_2(self.act(self.conv_2(x)))
        return x


class UnetBaseEncoder(nn.Module):
    """
    This module is the original encoder of the Unet
    """

    def __init__(self, sfn):
        super(UnetBaseEncoder, self).__init__()
        self.sfn = sfn
        self.cd_1 = ConvDownSample(3, self.sfn)
        self.cd_2 = ConvDownSample(self.sfn, self.sfn * 2)
        self.cd_3 = ConvDownSample(self.sfn * 2, self.sfn * 4)
        self.cd_4 = ConvDownSample(self.sfn * 4, self.sfn * 8)
        self.cd_5 = ConvDownSample(self.sfn * 8, self.sfn * 16, pool=False)

    def forward(self, x):
        layer0, x = self.cd_1(x)
        layer1, x = self.cd_2(x)
        layer2, x = self.cd_3(x)
        layer3, x = self.cd_4(x)
        layer4 = self.cd_5(x)
        return layer4, layer3, layer2, layer1, layer0


class UnetDecoder(nn.Module):
    """
    This module is the original decoder in the Unet
    """

    def __init__(
        self, in_chans, out_chans, margins, n_class, conv_chan=None, pad=0, up_sample=0
    ):
        super(UnetDecoder, self).__init__()
        assert len(in_chans) == len(out_chans) == len(margins)
        self.uc = []
        if not conv_chan:
            conv_chan = in_chans
        for i, o, c, m in zip(in_chans, out_chans, conv_chan, margins):
            self.uc.append(UpSampleConv(i, o, m, c, pad))
        self.uc = nn.ModuleList(self.uc)
        self.classify = nn.Conv2d(
            out_chans[-1], n_class, kernel_size=(3, 3), padding=(1, 1)
        )
        self.up_sample = up_sample

    def forward(self, x):
        # This line of code is changed to handle
        # breaking up the input into the encoder
        # and middle layers
        ftr, layers = x[0], x[1:]
        for l, uc in zip(layers, self.uc):
            ftr = uc(l, ftr)
        if self.up_sample > 0:
            ftr = F.interpolate(ftr, scale_factor=self.up_sample, mode="bilinear")
        return self.classify(ftr)


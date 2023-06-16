import torch
from torch import nn as nn
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch

class SpectralLoss(nn.Module):
    def __init__(self, ratio=2):
        super(SpectralLoss, self).__init__()
        self.ratio = ratio
        nbands = 6
        h_bl = gen_mtf(ratio, sensor='S2-20', kernel_size=9)
        h_bl = mtf_kernel_to_torch(h_bl)

        self.depthconv_20 = nn.Conv2d(in_channels=nbands,
                                 out_channels=nbands,
                                 groups=nbands,
                                 padding='same',
                                 padding_mode='replicate',
                                 kernel_size=h_bl.shape[-1],
                                 bias=False)

        self.depthconv_20.weight.data = h_bl
        self.depthconv_20.weight.requires_grad = False

        self.avgpool = nn.AvgPool2d(kernel_size=self.ratio)

        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):

        outputs_lp = self.depthconv_20(outputs)
        outputs_lr = self.avgpool(outputs_lp)

        L = self.loss(outputs_lr[:, :, 9:-9, 9:-9], labels[:, :, 9:-9, 9:-9])

        return L


class StructLoss(nn.Module):
    def __init__(self, ratio=2):
        super(StructLoss, self).__init__()
        self.ratio = ratio
        nbands = 6
        h_bl = gen_mtf(ratio, sensor='S2-20', kernel_size=9)
        h_bl = mtf_kernel_to_torch(h_bl)

        self.depthconv_20 = nn.Conv2d(in_channels=nbands,
                                 out_channels=nbands,
                                 groups=nbands,
                                 padding='same',
                                 padding_mode='replicate',
                                 kernel_size=h_bl.shape[-1],
                                 bias=False)

        self.depthconv_20.weight.data = h_bl
        self.depthconv_20.weight.requires_grad = False

        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):

        outputs_lp = self.depthconv_20(outputs)
        outputs_hr = outputs - outputs_lp

        L = self.loss(outputs_hr[:, :, 9:-9, 9:-9], labels[:, :, 9:-9, 9:-9])

        return L
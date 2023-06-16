import torch
import torch.nn.functional as F
from torch import nn

from Metrics.coregistration import stacked_fineshift
import Metrics.metrics_rr as mt
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
from Metrics.cross_correlation import xcorr_torch


def downgrade(img, kernel, ratio):
    img = F.conv2d(img, kernel.type(img.dtype).to(img.device), padding='same', groups=img.shape[1])
    ratio = int(ratio)
    if ratio == 2:
        r = torch.ones(img.shape[1], device=img.device, dtype=torch.uint8)
        c = torch.clone(r)
        img = stacked_fineshift(img, r, c, img.device)
        img = img[:, :, 1::ratio, 1::ratio]
    elif ratio == 3:
        img = img[:, :, 0::ratio, 0::ratio]
    else:
        img = img[:, :, 3::ratio, 3::ratio]
    return img


class ReproErgas(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproErgas, self).__init__()
        self.ratio = ratio
        self.ERGAS = mt.ERGAS(self.ratio)
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'

        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.ERGAS(downgraded_outputs, labels)


class ReproSAM(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproSAM, self).__init__()
        self.ratio = ratio
        self.SAM = mt.SAM()
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.SAM(downgraded_outputs, labels)


class ReproQ2n(nn.Module):
    def __init__(self, ratio, lana=False):
        super(ReproQ2n, self).__init__()
        self.ratio = ratio
        self.Q2n = mt.Q2n()
        if ratio == 2:
            sensor = 'S2-20'
        elif ratio == 6:
            sensor = 'S2-60'
        else:
            sensor = 'S2-60'
        if lana and ratio == 6:
            sensor = 'S2-60_bis'
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.Q2n(downgraded_outputs, labels)


class ReproQ(nn.Module):
    def __init__(self, ratio, lana=False, device='cuda'):
        super(ReproQ, self).__init__()
        self.ratio = ratio

        if ratio == 2:
            sensor = 'S2-20'
            nbands = 6
        elif ratio == 6:
            sensor = 'S2-60'
            nbands = 3
        else:
            sensor = 'S2-60'
            nbands = 3
        if lana and ratio == 6:
            sensor = 'S2-60_bis'

        self.Q = mt.Q(nbands, device)
        self.kernel = mtf_kernel_to_torch(gen_mtf(self.ratio, sensor))

    def forward(self, outputs, labels):

        downgraded_outputs = downgrade(outputs, self.kernel, self.ratio).float()
        return self.Q(downgraded_outputs, labels)


class D_rho(nn.Module):

    def __init__(self, sigma):
        # Class initialization
        super(D_rho, self).__init__()

        # Parameters definition:
        self.scale = sigma // 2

    def forward(self, outputs, labels):
        Y = torch.ones((outputs.shape[0], outputs.shape[1], outputs.shape[2], outputs.shape[3], labels.shape[1]),
                       device=outputs.device)

        for i in range(labels.shape[1]):
            Y[:, :, :, :, i] = torch.clamp(xcorr_torch(outputs,
                                                       torch.unsqueeze(labels[:, i, :, :],
                                                                       1),
                                                       self.scale, outputs.device), min=-1.0)

        Y = torch.amax(Y, -1)
        Y = torch.clip(Y, -1, 1)
        X = 1.0 - Y
        Lxcorr = torch.mean(X)

        return Lxcorr, torch.mean(X, dim=(2, 3))

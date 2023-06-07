import torch

from Utils.spectral_tools import LPfilterGauss
from Utils.pansharpening_aux_tools import estimation_alpha


def BT_H(ordered_dict):
    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)
    ratio = ordered_dict.ratio

    min_bands_low = torch.amin(bands_low, dim=(2, 3), keepdim=True)

    bands_high_lp = LPfilterGauss(bands_high, ratio)
    alphas = estimation_alpha(bands_low, bands_high_lp)

    img = torch.sum((bands_low - min_bands_low) * alphas, dim=1, keepdim=True)

    img_hr = (bands_high - torch.mean(bands_high_lp, dim=(2, 3), keepdim=True)) / (
                torch.std(img, dim=(2, 3), keepdim=True) / torch.std(bands_high_lp, dim=(2, 3),
                                                                     keepdim=True)) + torch.mean(img, dim=(2, 3),
                                                                                                 keepdim=True)

    bands_low_minus_min = bands_low - min_bands_low
    bands_low_minus_min = torch.clip(bands_low_minus_min, 0, bands_low_minus_min.max())

    fused = bands_low_minus_min * (img_hr / (img + torch.finfo(torch.float32).eps)).repeat(1, bands_low.shape[1], 1,
                                                                                           1) + min_bands_low

    return fused

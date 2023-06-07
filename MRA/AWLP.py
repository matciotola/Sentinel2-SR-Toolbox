import torch
from torchvision.transforms import InterpolationMode as Inter
from torchvision.transforms.functional import resize

from Utils.spectral_tools import LPFilter


def AWLP(ordered_dict):
    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)
    ratio = ordered_dict.ratio

    bs, c, h, w = bands_low.shape

    mean_low = torch.mean(bands_low, dim=1, keepdim=True)

    img_intensity = bands_low / (mean_low + torch.finfo(bands_low.dtype).eps)

    bands_high = bands_high.repeat(1, c, 1, 1)

    bands_high_lp = resize(
        resize(bands_high,
               [bands_high.shape[2] // ratio, bands_high.shape[3] // ratio],
               interpolation=Inter.BICUBIC,
               antialias=True),
        [bands_high.shape[2], bands_high.shape[3]],
        interpolation=Inter.BICUBIC,
        antialias=True)

    bands_high = (bands_high - torch.mean(bands_high, dim=(2, 3), keepdim=True)) * (
                torch.std(bands_low, dim=(2, 3), keepdim=True) / torch.std(bands_high_lp, dim=(2, 3),
                                                                           keepdim=True)) + torch.mean(bands_low,
                                                                                                       dim=(2, 3),
                                                                                                       keepdim=True)

    bands_high_lpp = []
    for i in range(bands_high.shape[1]):
        bands_high_lpp.append(LPFilter(bands_high[:, i, None, :, :].float(), ratio))

    bands_high_lpp = torch.cat(bands_high_lpp, dim=1)

    details = bands_high - bands_high_lpp

    fused = details * img_intensity + bands_low

    return fused

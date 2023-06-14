import torch
from torch.nn.functional import interpolate


def normalize(bands, scale=2048):
    normalized = bands / scale
    return normalized


def denormalize(img, scale=2048):
    denormalized = img * scale
    return denormalized


def downsample_protocol(img, ratio):
    img_lr = interpolate(img, scale_factor=1 / ratio, mode='bicubic', align_corners=False, antialias=True)
    return img_lr


def upsample_protocol(img, ratio):
    img_hr = interpolate(img, scale_factor=ratio, mode='bicubic', align_corners=False, antialias=True)
    return img_hr


def get_patches(bands, patches_size=33):
    print('list_bands for patches: ' + str(len(bands)))
    patches = []

    _, b, c, r = bands.shape
    cont = 0
    for i in range(0, r - patches_size, patches_size):
        for j in range(0, c - patches_size, patches_size):
            p = bands[:, :, i:i + patches_size, j:j + patches_size]

            patches.append(p)

            cont += 1
    patches = torch.cat(patches, dim=0)
    return patches


def input_prepro_20(bands_high, bands_low, ratio):
    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = upsample_protocol(bands_low_lr_lr, ratio)

    return bands_high_lr, bands_low_lr, bands_low

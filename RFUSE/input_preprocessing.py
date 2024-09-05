import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.functional import conv2d
from osgeo import gdal


from Utils.imresize_bicubic import imresize
from Utils.spectral_tools import mtf, gen_mtf, mtf_kernel_to_torch
from Metrics.cross_correlation import xcorr_torch
from Utils.interpolator_tools import ideal_interpolator

def normalize(bands, ratio=2, shift=2):

    if bands.shape[1] != 4:
        denormalized = bands
    else:
        denormalized = low_pass_filter(bands, ratio)

    mean = torch.mean(denormalized, dim=(2, 3), keepdim=True)
    std = torch.std(denormalized, dim=(2, 3), keepdim=True)
    normalized = shift + ((bands - mean) / std)

    return normalized


def denormalize(bands, mean, std, shift=2):

    denormalized = (bands - shift) * std.to(bands.device) + mean.to(bands.device)

    return denormalized


def downsample_protocol(img, ratio):
    if img.shape[1] == 4:
        sensor = 'S2-10'
    else:
        sensor = 'S2-20'

    img_lp = mtf(img, sensor, ratio)
    img_lr = F.avg_pool2d(img_lp, kernel_size=ratio)

    return img_lr


def upsample_protocol(img, ratio):
    img_hr = ideal_interpolator(img, ratio).float()

    return img_hr


def input_prepro_rr(bands_high, bands_low, ratio):
    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    #bands_low_lr = upsample_protocol(bands_low_lr_lr, ratio)
    bands_low_lr = F.interpolate(bands_low_lr_lr, scale_factor=ratio, mode='bicubic')
    return bands_high_lr, bands_low_lr, bands_low


def input_prepro_fr(bands_high, bands_low_lr, ratio):
    #bands_low = F.interpolate(bands_low_lr, scale_factor=ratio, mode='bicubic')
    bands_low = imresize(bands_low_lr, scale=ratio)
    struct_reference = fuseUpGenDetailRef(bands_high, bands_low_lr, ratio, ratio + 1)
    return bands_high, bands_low, bands_low_lr, struct_reference


def fuseUpGenDetailRef(bands_high, bands_low_lr, ratio=2, w_size=3, shrink=5):

    h = gen_mtf(ratio, sensor='None', kernel_size=41, nbands=bands_high.shape[1])
    h = mtf_kernel_to_torch(h).to(bands_high.device)

    bands_high_lp = conv2d(bands_high, h, padding='same', groups=bands_high.shape[1])
    bands_high_lp_lr = imresize(bands_high_lp, scale=1/ratio, antialiasing=True)

    X = []
    for i in range(bands_low_lr.shape[1]):
        temp = xcorr_torch(bands_low_lr[:, i:i + 1, :, :], bands_high_lp_lr, w_size, bands_low_lr.device)
        X.append(F.interpolate(temp, scale_factor=ratio, mode='bicubic', antialias=True)[:, :, :, :, None])
    X = torch.cat(X, dim=-1)
    eX = torch.exp(shrink * X)
    eXsum = torch.sum(eX, 1, keepdim=True)
    eX = eX / eXsum
    hp = []
    if ratio == 2:
        sensor = 'S2-20'
    else:
        sensor = 'S2-60'
    for b in range(bands_high.shape[1]):
        temp = bands_high[:, b, None, :, :].repeat(1, bands_low_lr.shape[1], 1, 1)
        temp1 = temp - mtf(temp, 'S2-20', ratio, mode='replicate')
        hp.append(temp1[:, :, :, :, None])
    hp = torch.cat(hp, dim=-1)
    eX = eX.transpose(1, -1)
    temp = hp * eX
    detailRef = torch.sum(temp, -1)

    return detailRef


def protocol(b_h_path, b_l_lr_path):
    ratio = 2
    list_file_high = os.listdir(b_h_path)
    list_file_low = os.listdir(b_l_lr_path)

    list_bands_low_lr = []
    for file_low in list_file_low:
        print(os.path.join(b_l_lr_path, file_low))
        bands_low_lr = gdal.Open(os.path.join(b_l_lr_path, file_low))
        bands_low_lr = np.asarray(bands_low_lr.ReadAsArray())
        bands_low_lr = bands_low_lr.astype('float32')
        bands_low_lr = torch.Tensor(bands_low_lr)[None, :, :, :]
        list_bands_low_lr.append(bands_low_lr)
        print('band_low: ' + str(bands_low_lr.shape))

    list_bands_high = []
    for file_high in list_file_high:
        print(os.path.join(b_h_path, file_high))
        bands_high = gdal.Open(os.path.join(b_h_path, file_high))
        bands_high = np.asarray(bands_high.ReadAsArray())
        bands_high = bands_high.astype('float32')
        bands_high = torch.Tensor(bands_high)[None, :, :, :]
        list_bands_high.append(bands_high)
        print('band_high_lr: ' + str(bands_high.shape))

    print('list_bands_high_lr: ' + str(len(list_bands_high)))
    print('list_bands_low_lr: ' + str(len(list_bands_low_lr)))
    list_bands_high_lr = torch.cat(list_bands_high, dim=0)
    list_bands_low_lr = torch.cat(list_bands_low_lr, dim=0)

    return list_bands_high_lr, list_bands_low_lr


def low_pass_filter(bands, ratio, kernel=9):

    h = gen_mtf(ratio, sensor='None', kernel_size=kernel, nbands=bands.shape[1])
    h = mtf_kernel_to_torch(h).to(bands.device)

    bands_lp = conv2d(bands, h, padding='same', groups=bands.shape[1])

    return bands_lp




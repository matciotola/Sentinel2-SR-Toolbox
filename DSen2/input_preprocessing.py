import numpy as np
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, gaussian_blur, pad
from torch.nn.functional import avg_pool2d
from typing import Tuple

def normalize(img, scale=2000):
    normalized = img / scale

    return normalized


def denormalize(img, scale=2000):
    denormalized = img * scale

    return denormalized


def input_prepro_20(bands_high, bands_low, ratio):
    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = upsample_protocol(bands_low_lr_lr, bands_high_lr.shape)

    return bands_high_lr, bands_low_lr, bands_low


def input_prepro_60(bands_high, bands_intermediate, bands_low, ratio):
    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_intermediate_lr = downsample_protocol(bands_intermediate, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = upsample_protocol(bands_low_lr_lr, bands_high_lr.shape)

    return bands_high_lr, bands_intermediate_lr, bands_low_lr, bands_low


def downsample_protocol(img, ratio):
    sigma = 1 / ratio
    radius = round(4.0 * sigma)
    kernel_size = 2 * radius + 1
    img_lp = gaussian_blur(img, [kernel_size, kernel_size], sigma)
    img_lr = avg_pool2d(img_lp, [ratio, ratio])

    return img_lr


def upsample_protocol(img, shape):
    img_hr = resize(img / 30000, (shape[2:4]),
                    interpolation=InterpolationMode.BILINEAR, antialias=True) * 30000  # bilinear
    return img_hr


def get_test_patches_20(dset_10, dset_20, patch_size=128, border=4, interp=True):
    patch_size_hr = (patch_size, patch_size)
    patch_size_lr = [p // 2 for p in patch_size_hr]
    border_hr = border
    border_lr = border_hr // 2

    # Mirror the data at the borders to have the same dimensions as the input
    # dset_10 = pad(dset_10, (border_hr, border_hr, border_hr, border_hr), padding_mode='symmetric')
    # dset_20 = pad(dset_20, [border_lr, border_lr, border_lr, border_lr], padding_mode='symmetric')

    dset_10 = symm_pad(dset_10, (border_hr, border_hr, border_hr, border_hr))
    dset_20 = symm_pad(dset_20, (border_lr, border_lr, border_lr, border_lr))

    bands10 = dset_10.shape[1]
    bands20 = dset_20.shape[1]
    patches_along_i = (dset_20.shape[2] - 2 * border_lr) // (patch_size_lr[0] - 2 * border_lr)
    patches_along_j = (dset_20.shape[3] - 2 * border_lr) // (patch_size_lr[1] - 2 * border_lr)

    nr_patches = (patches_along_i + 1) * (patches_along_j + 1)

    image_20 = torch.zeros(((nr_patches, bands20) + tuple(patch_size_lr)), dtype=torch.float32)
    image_10 = torch.zeros(((nr_patches, bands10) + patch_size_hr), dtype=torch.float32)

    range_i = np.arange(0, (dset_20.shape[2] - 2 * border_lr) // (patch_size_lr[0] - 2 * border_lr)) * (
            patch_size_lr[0] - 2 * border_lr)
    range_j = np.arange(0, (dset_20.shape[3] - 2 * border_lr) // (patch_size_lr[1] - 2 * border_lr)) * (
            patch_size_lr[1] - 2 * border_lr)

    if not ((dset_20.shape[0] - 2 * border_lr) % (patch_size_lr[0] - 2 * border_lr)) == 0:
        range_i = np.append(range_i, (dset_20.shape[2] - patch_size_lr[0]))
    if not ((dset_20.shape[1] - 2 * border_lr) % (patch_size_lr[1] - 2 * border_lr)) == 0:
        range_j = np.append(range_j, (dset_20.shape[3] - patch_size_lr[1]))

    p_count = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            crop_point_lr = [ii, jj, ii + patch_size_lr[0], jj + patch_size_lr[1]]
            crop_point_hr = [p * 2 for p in crop_point_lr]
            image_20[p_count] = dset_20[:, :, crop_point_lr[0]:crop_point_lr[2], crop_point_lr[1]:crop_point_lr[3]]
            image_10[p_count] = dset_10[:, :, crop_point_hr[0]:crop_point_hr[2], crop_point_hr[1]:crop_point_hr[3]]
            p_count += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = upsample_protocol(image_20, image_10_shape)
    else:
        data20_interp = image_20

    return image_10, data20_interp


def get_test_patches_60(dset_10, dset_20, dset_60, patch_size=128, border=8, interp=True):
    patch_size_10 = (patch_size, patch_size)
    patch_size_20 = [p // 2 for p in patch_size_10]
    patch_size_60 = [p // 6 for p in patch_size_10]
    border_10 = border
    border_20 = border_10 // 2
    border_60 = border_10 // 6

    # Mirror the data at the borders to have the same dimensions as the input
    # dset_10 = pad(dset_10, [border_10, border_10, border_10, border_10], padding_mode='symmetric')
    # dset_20 = pad(dset_20, [border_20, border_20, border_20, border_20], padding_mode='symmetric')
    # dset_60 = pad(dset_60, [border_60, border_60, border_60, border_60], padding_mode='symmetric')

    dset_10 = symm_pad(dset_10, (border_10, border_10, border_10, border_10))
    dset_20 = symm_pad(dset_20, (border_20, border_20, border_20, border_20))
    dset_60 = symm_pad(dset_60, (border_60, border_60, border_60, border_60))

    bands10 = dset_10.shape[1]
    bands20 = dset_20.shape[1]
    bands60 = dset_60.shape[1]
    patches_along_i = (dset_60.shape[2] - 2 * border_60) // (patch_size_60[0] - 2 * border_60)
    patches_along_j = (dset_60.shape[3] - 2 * border_60) // (patch_size_60[1] - 2 * border_60)

    nr_patches = (patches_along_i + 1) * (patches_along_j + 1)

    image_10 = torch.zeros((nr_patches, bands10) + patch_size_10, dtype=torch.float32)
    image_20 = torch.zeros((nr_patches, bands20) + tuple(patch_size_20), dtype=torch.float32)
    image_60 = torch.zeros((nr_patches, bands60) + tuple(patch_size_60), dtype=torch.float32)

    range_i = np.arange(0, (dset_60.shape[2] - 2 * border_60) // (patch_size_60[0] - 2 * border_60)) * (
            patch_size_60[0] - 2 * border_60)
    range_j = np.arange(0, (dset_60.shape[3] - 2 * border_60) // (patch_size_60[1] - 2 * border_60)) * (
            patch_size_60[1] - 2 * border_60)

    if not ((dset_60.shape[0] - 2 * border_60 % patch_size_60[0] - 2 * border_60) == 0):
        range_i = np.append(range_i, (dset_60.shape[2] - patch_size_60[0]))
    if not ((dset_60.shape[1] - 2 * border_60 % patch_size_60[1] - 2 * border_60) == 0):
        range_j = np.append(range_j, (dset_60.shape[3] - patch_size_60[1]))

    p_count = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):
            crop_point_60 = [ii, jj, ii + patch_size_60[0], jj + patch_size_60[1]]
            crop_point_10 = [p * 6 for p in crop_point_60]
            crop_point_20 = [p * 3 for p in crop_point_60]
            image_10[p_count] = dset_10[:, :, crop_point_10[0]:crop_point_10[2], crop_point_10[1]:crop_point_10[3]]
            image_20[p_count] = dset_20[:, :, crop_point_20[0]:crop_point_20[2], crop_point_20[1]:crop_point_20[3]]
            image_60[p_count] = dset_60[:, :, crop_point_60[0]:crop_point_60[2], crop_point_60[1]:crop_point_60[3]]
            p_count += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = upsample_protocol(image_20, image_10_shape)
        data60_interp = upsample_protocol(image_60, image_10_shape)

    else:
        data20_interp = image_20
        data60_interp = image_60

    return image_10, data20_interp, data60_interp


def recompose_images(a, border, size=None):
    if a.shape[0] == 1:
        images = a[0]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2] - border * 2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = np.ceil(size[2] / patch_size)
        y_tiles = np.ceil(size[3] / patch_size)
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        images = torch.zeros((size[0], a.shape[1], size[2], size[3]), dtype=torch.float32)

        current_patch = 0
        for y in range(int(y_tiles)):
            ypoint = y * patch_size
            if ypoint > size[3] - patch_size:
                ypoint = size[3] - patch_size
            for x in range(int(x_tiles)):
                xpoint = x * patch_size
                if xpoint > size[2] - patch_size:
                    xpoint = size[2] - patch_size
                images[:, :, ypoint:ypoint + patch_size, xpoint:xpoint + patch_size] = a[current_patch, :,
                                                                                         border:a.shape[2] - border,
                                                                                         border:a.shape[3] - border]
                current_patch += 1

    return images


def symm_pad(im: torch.Tensor, padding: Tuple[int, int, int, int]):
    h, w = im.shape[-2:]
    left, right, top, bottom = padding

    x_idx = np.arange(-left, w + right)
    y_idx = np.arange(-top, h + bottom)

    def reflect(x, minx, maxx):
        """ Reflects an array around two points making a triangular waveform that ramps up
        and down,  allowing for pad lengths greater than the input length """
        rng = maxx - minx
        double_rng = 2 * rng
        mod = np.fmod(x - minx, double_rng)
        normed_mod = np.where(mod < 0, mod + double_rng, mod)
        out = np.where(normed_mod >= rng, double_rng - normed_mod, normed_mod) + minx
        return np.array(out, dtype=x.dtype)

    x_pad = reflect(x_idx, -0.5, w - 0.5)
    y_pad = reflect(y_idx, -0.5, h - 0.5)
    xx, yy = np.meshgrid(x_pad, y_pad)
    return im[..., yy, xx]
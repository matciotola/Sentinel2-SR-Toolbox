import numpy as np
from skimage.transform import resize
import torch
from typing import Tuple
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, gaussian_blur
from torch.nn.functional import avg_pool2d

def normalize(img, scale=2000):
    normalized = img / scale
    return normalized

def denormalize(img, scale=2000):
    denormalized = img * scale
    return denormalized

def input_prepro(bands_high, bands_low, ratio):

    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = interp_patches(bands_low_lr_lr)

    return bands_high_lr, bands_low_lr, bands_low

def input_prepro60(bands_high, bands_intermediate, bands_low, ratio):

    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_intermediate_lr = downsample_protocol(bands_intermediate, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = interp_patches(bands_low_lr_lr)

    return bands_high_lr, bands_intermediate_lr, bands_low_lr, bands_low

def downsample_protocol(img, ratio):
    sigma = 1 / ratio
    radius = round(4.0 * sigma)

    kernel_size = 2 * radius + 1

    img_lp = gaussian_blur(img, [kernel_size, kernel_size], sigma)

    img_lr = avg_pool2d(img_lp, [ratio, ratio])


    return img_lr


def interp_patches(bands_low, bands_high_shape):

    data20_interp = resize(bands_low / 30000, (bands_high_shape[2:4]), interpolation=InterpolationMode.BILINEAR) * 30000  # bilinear
    return data20_interp


def get_test_patches(dset_10, dset_20, patchSize=128, border=4, interp=True):

    PATCH_SIZE_HR = (patchSize, patchSize)
    PATCH_SIZE_LR = [p//2 for p in PATCH_SIZE_HR]
    BORDER_HR = border
    BORDER_LR = BORDER_HR//2

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = symm_pad(dset_10, (BORDER_HR, BORDER_HR, BORDER_HR, BORDER_HR))
    dset_20 = symm_pad(dset_20, (BORDER_LR, BORDER_LR, BORDER_LR, BORDER_LR))


    BANDS10 = dset_10.shape[1]
    BANDS20 = dset_20.shape[1]
    patchesAlongi = (dset_20.shape[2] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    patchesAlongj = (dset_20.shape[3] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    image_20 = torch.zeros(((nr_patches, BANDS20) + tuple(PATCH_SIZE_LR)), dtype=torch.float32)
    image_10 = torch.zeros(((nr_patches, BANDS10) + PATCH_SIZE_HR), dtype=torch.float32)

    range_i = np.arange(0, (dset_20.shape[2] - 2 * BORDER_LR) // (PATCH_SIZE_LR[0] - 2 * BORDER_LR)) * (PATCH_SIZE_LR[0] - 2 * BORDER_LR)
    range_j = np.arange(0, (dset_20.shape[3] - 2 * BORDER_LR) // (PATCH_SIZE_LR[1] - 2 * BORDER_LR)) * (PATCH_SIZE_LR[1] - 2 * BORDER_LR)

    if not ((dset_20.shape[0] - 2 * BORDER_LR) % (PATCH_SIZE_LR[0] - 2 * BORDER_LR)) == 0:
        range_i = np.append(range_i, (dset_20.shape[2] - PATCH_SIZE_LR[0]))
    if not ((dset_20.shape[1] - 2 * BORDER_LR) % (PATCH_SIZE_LR[1] - 2 * BORDER_LR)) == 0:
        range_j = np.append(range_j, (dset_20.shape[3] - PATCH_SIZE_LR[1]))

    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):

            crop_point_lr = [ii, jj, ii + PATCH_SIZE_LR[0], jj + PATCH_SIZE_LR[1]]
            crop_point_hr = [p*2 for p in crop_point_lr]
            image_20[pCount] = dset_20[:, :, crop_point_lr[0]:crop_point_lr[2], crop_point_lr[1]:crop_point_lr[3]]
            image_10[pCount] = dset_10[:, :, crop_point_hr[0]:crop_point_hr[2], crop_point_hr[1]:crop_point_hr[3]]
            pCount += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
    else:
        data20_interp = image_20
    return image_10, data20_interp


def get_test_patches60(dset_10, dset_20, dset_60, patchSize=128, border=8, interp=True):

    PATCH_SIZE_10 = (patchSize, patchSize)
    PATCH_SIZE_20 = [p//2 for p in PATCH_SIZE_10]
    PATCH_SIZE_60 = [p//6 for p in PATCH_SIZE_10]
    BORDER_10 = border
    BORDER_20 = BORDER_10//2
    BORDER_60 = BORDER_10//6

    # Mirror the data at the borders to have the same dimensions as the input
    dset_10 = symm_pad(dset_10, (BORDER_10, BORDER_10, BORDER_10, BORDER_10))
    dset_20 = symm_pad(dset_20, (BORDER_20, BORDER_20, BORDER_20, BORDER_20))
    dset_60 = symm_pad(dset_60, (BORDER_60, BORDER_60, BORDER_60, BORDER_60))



    BANDS10 = dset_10.shape[1]
    BANDS20 = dset_20.shape[1]
    BANDS60 = dset_60.shape[1]
    patchesAlongi = (dset_60.shape[2] - 2 * BORDER_60) // (PATCH_SIZE_60[0] - 2 * BORDER_60)
    patchesAlongj = (dset_60.shape[3] - 2 * BORDER_60) // (PATCH_SIZE_60[1] - 2 * BORDER_60)

    nr_patches = (patchesAlongi + 1) * (patchesAlongj + 1)

    image_10 = torch.zeros((nr_patches, BANDS10) + PATCH_SIZE_10, dtype=torch.float32)
    image_20 = torch.zeros((nr_patches, BANDS20) + tuple(PATCH_SIZE_20), dtype=torch.float32)
    image_60 = torch.zeros((nr_patches, BANDS60) + tuple(PATCH_SIZE_60), dtype=torch.float32)


    range_i = np.arange(0, (dset_60.shape[2] - 2 * BORDER_60) // (PATCH_SIZE_60[0] - 2 * BORDER_60)) * (PATCH_SIZE_60[0] - 2 * BORDER_60)
    range_j = np.arange(0, (dset_60.shape[3] - 2 * BORDER_60) // (PATCH_SIZE_60[1] - 2 * BORDER_60)) * (PATCH_SIZE_60[1] - 2 * BORDER_60)

    if not ((dset_60.shape[0] - 2 * BORDER_60 % PATCH_SIZE_60[0] - 2 * BORDER_60) == 0):
        range_i = np.append(range_i, (dset_60.shape[2] - PATCH_SIZE_60[0]))
    if not ((dset_60.shape[1] - 2 * BORDER_60 % PATCH_SIZE_60[1] - 2 * BORDER_60) == 0):
        range_j = np.append(range_j, (dset_60.shape[3] - PATCH_SIZE_60[1]))


    pCount = 0
    for ii in range_i.astype(int):
        for jj in range_j.astype(int):

            crop_point_60 = [ii, jj, ii + PATCH_SIZE_60[0], jj + PATCH_SIZE_60[1]]
            crop_point_10 = [p*6 for p in crop_point_60]
            crop_point_20 = [p*3 for p in crop_point_60]
            image_10[pCount] = dset_10[:,:, crop_point_10[0]:crop_point_10[2], crop_point_10[1]:crop_point_10[3]]
            image_20[pCount] = dset_20[:,:, crop_point_20[0]:crop_point_20[2], crop_point_20[1]:crop_point_20[3]]
            image_60[pCount] = dset_60[:,:, crop_point_60[0]:crop_point_60[2], crop_point_60[1]:crop_point_60[3]]
            pCount += 1

    image_10_shape = image_10.shape

    if interp:
        data20_interp = interp_patches(image_20, image_10_shape)
        data60_interp = interp_patches(image_60, image_10_shape)

    else:
        data20_interp = image_20
        data60_interp = image_60

    return image_10, data20_interp, data60_interp

def recompose_images(a, border, size=None):
    if a.shape[1] == 1:
        images = a[1]
    else:
        # # This is done because we do not mirror the data at the image border
        # size = [s - border * 2 for s in size]
        patch_size = a.shape[2]-border*2

        # print('Patch has dimension {}'.format(patch_size))
        # print('Prediction has shape {}'.format(a.shape))
        x_tiles = np.ceil(size[2]/patch_size)
        y_tiles = np.ceil(size[3]/patch_size)
        # print('Tiles per image {} {}'.format(x_tiles, y_tiles))

        # Initialize image
        # print('Image size is: {}'.format(size))
        images = torch.zeros((size[0], a.shape[1], size[2], size[3]), dtype=torch.float32)

        print(images.shape)
        current_patch = 0
        for y in range(int(y_tiles)):
            ypoint = y * patch_size
            if ypoint > size[3] - patch_size:
                ypoint = size[3] - patch_size
            for x in range(int(x_tiles)):
                xpoint = x * patch_size
                if xpoint > size[2] - patch_size:
                    xpoint = size[2] - patch_size
                images[:, :, ypoint:ypoint+patch_size, xpoint:xpoint+patch_size] = a[current_patch, :, border:a.shape[2]-border, border:a.shape[3]-border]
                current_patch += 1

    # return images.transpose((1, 2, 0))
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

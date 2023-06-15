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


def input_prepro_20(bands_high, bands_low, ratio):
    bands_high_lr = downsample_protocol(bands_high, ratio)
    bands_low_lr_lr = downsample_protocol(bands_low, ratio)
    bands_low_lr = upsample_protocol(bands_low_lr_lr, ratio)

    return bands_high_lr, bands_low_lr, bands_low

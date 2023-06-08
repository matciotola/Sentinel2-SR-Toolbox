from torch.nn import functional as F
from Utils.spectral_tools import mtf

def downsample_protocol(bands_high, bands_intermediate, bands_low, ratio):

    if ratio == 2:

        bands_high_lp = mtf(bands_high, 'S2_10', ratio)
        bands_low_lp = mtf(bands_low, 'S2_20', ratio)

        bands_high_lr = F.interpolate(bands_high_lp, scale_factor= 1/ratio, mode='nearest-exact')
        bands_low_lr = F.interpolate(bands_low_lp, scale_factor= 1/ratio, mode='nearest-exact')

        bands_intermediate_lr = None

    elif ratio == 6:

        bands_high_lp = mtf(bands_high, 'S2_10', ratio)
        bands_intermediate_lp = mtf(bands_intermediate, 'S2_20', ratio)
        bands_low_lp = mtf(bands_low, 'S2_60', ratio)

        bands_high_lr = F.interpolate(bands_high_lp, scale_factor= 1/ratio, mode='nearest-exact')
        bands_intermediate_lr = F.interpolate(bands_intermediate_lp, scale_factor= 1/ratio, mode='nearest-exact')
        bands_low_lr = F.interpolate(bands_low_lp, scale_factor= 1/ratio, mode='nearest-exact')

    return bands_high_lr, bands_intermediate_lr, bands_low_lr
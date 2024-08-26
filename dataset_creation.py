import os
import numpy as np
from osgeo import gdal
import torch
from torch.nn.functional import interpolate
from Utils.spectral_tools import mtf
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy import io
from tqdm import tqdm

def find_null_patches(patches, null_value=0):

    # Remove patches with at least one pixel of null value (zero-value)
    zero_mask = (patches == null_value).all(dim=1)
    indexes = zero_mask.any(dim=1).any(dim=1)
    return ~indexes


def patch_extraction(bands, patch_size):
    """
    Extract patches from the input bands
    :param bands: list of bands
    :param patch_size: size of the patch
    :param null_value: value to fill the patches
    :return: list of patches
    """

    kc, kh, kw = bands.shape[1], patch_size, patch_size  # kernel size
    dc, dh, dw = bands.shape[1], patch_size, patch_size  # stride
    patches = bands.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
    patches = patches.contiguous().view(-1, kc, kh, kw)

    return patches


if __name__ == '__main__':
    rootbase = '/media/matteo/T7/Datasets/Sentinel2/'
    savebase = '/media/matteo/T7/Datasets/Sentinel2/Patches'

    n_patches = 64

    B10 = ['B02', 'B03', 'B04', 'B08']
    B20 = ['B05', 'B06', 'B07', 'B8A', 'B11', 'B12']
    B60 = ['B01', 'B09', 'B10']

    # continents = ['Training', 'Validation', 'Test']
    continents = ['Test']


    for continent in continents:
        print('Processing', continent)
        countries = sorted(next(os.walk(os.path.join(rootbase, continent)))[1])

        for country in countries:
            root = os.path.join(rootbase, continent, country)

            # Load the images
            times_acquired = sorted(list(next(os.walk(root))[1]))

            for time in times_acquired:
                print('Processing', continent, country, time)
                path_to_granule = os.path.join(root, time, 'GRANULE')
                granules = next(os.walk(path_to_granule))[1][0]
                path_to_bands = os.path.join(path_to_granule, granules, 'IMG_DATA')

                # Load the images
                bands_path = next(os.walk(path_to_bands))[2]
                bands_10_names = sorted([band for band in bands_path if any(band_wl in band for band_wl in B10)])
                bands_20_names = sorted([band for band in bands_path if any(band_wl in band for band_wl in B20)])
                bands_20_names = [bands_20_names[0], bands_20_names[1],bands_20_names[2], bands_20_names[-1], bands_20_names[3], bands_20_names[4]] # Reorder the bands
                bands_60_names = sorted([band for band in bands_path if any(band_wl in band for band_wl in B60)])

                bands_10 = [torch.from_numpy(gdal.Open(os.path.join(path_to_bands, band)).ReadAsArray().astype(np.float32))[None, :, :] for band in bands_10_names]
                bands_20 = [torch.from_numpy(gdal.Open(os.path.join(path_to_bands, band)).ReadAsArray().astype(np.float32))[None, :, :] for band in bands_20_names]
                bands_60 = [torch.from_numpy(gdal.Open(os.path.join(path_to_bands, band)).ReadAsArray().astype(np.float32))[None, :, :] for band in bands_60_names]

                bands_10 = torch.cat(bands_10, dim=0).unsqueeze(0)
                bands_20 = torch.cat(bands_20, dim=0).unsqueeze(0)
                bands_60 = torch.cat(bands_60, dim=0).unsqueeze(0)
                # Full-Resolution

                if continent == 'Training' or continent == 'Validation':


                    # Patch extraction
                    patch_size = 60
                    patches_10 = patch_extraction(bands_10, patch_size * 6)
                    patches_20 = patch_extraction(bands_20, patch_size * 3)
                    patches_60 = patch_extraction(bands_60, patch_size)
                    # Select patches randomly
                    indexes = find_null_patches(patches_60)
                    patches_10 = patches_10[indexes]
                    patches_20 = patches_20[indexes]
                    patches_60 = patches_60[indexes]

                    # Take n patches in the middle
                    if patches_10.shape[0] > n_patches:
                        m = patches_10.shape[0] // 2
                        patches_10 = patches_10[m - n_patches//2:m + n_patches//2]
                        patches_20 = patches_20[m - n_patches//2:m + n_patches//2]
                        patches_60 = patches_60[m - n_patches//2:m + n_patches//2]

                else:
                    patch_size = 400
                    if country == 'Alexandria':
                        ini_h = 1300
                        fin_h = ini_h + patch_size
                        ini_w = 1000
                        fin_w = ini_w + patch_size
                    elif country == 'Beijing':
                        ini_h = 1100
                        fin_h = ini_h + patch_size
                        ini_w = 600
                        fin_w = ini_w + patch_size
                    elif country == 'Berlin':
                        ini_h = 500
                        fin_h = ini_h + patch_size
                        ini_w = 900
                        fin_w = ini_w + patch_size
                    elif country == 'New_York':
                        ini_h = 1430
                        fin_h = ini_h + patch_size
                        ini_w = 1250
                        fin_w = ini_w + patch_size
                    elif country == 'Paris':
                        ini_h = 1000
                        fin_h = ini_h + patch_size
                        ini_w = 900
                        fin_w = ini_w + patch_size
                    elif country == 'Reynosa':
                        ini_h = 800
                        fin_h = ini_h + patch_size
                        ini_w = 600
                        fin_w = ini_w + patch_size
                    elif country == 'Rome':
                        ini_h = 400
                        fin_h = ini_h + patch_size
                        ini_w = 1100
                        fin_w = ini_w + patch_size
                    elif country == 'Tazoskij':
                        ini_h = 300
                        fin_h = ini_h + patch_size
                        ini_w = 300
                        fin_w = ini_w + patch_size
                    elif country == 'Tokyo':
                        ini_h = 900
                        fin_h = ini_h + patch_size
                        ini_w = 200
                        fin_w = ini_w + patch_size
                    elif country == 'Ulaanbaator':
                        ini_h = 700
                        fin_h = ini_h + patch_size
                        ini_w = 200
                        fin_w = ini_w + patch_size
                    else:
                        ini_h, fin_h = 0, 400
                        ini_w, fin_w = 0, 400

                    patches_10 = bands_10[:, :, ini_h * 6:fin_h * 6, ini_w * 6:fin_w * 6]
                    patches_20 = bands_20[:, :, ini_h * 3:fin_h * 3, ini_w * 3:fin_w * 3]
                    patches_60 = bands_60[:, :, ini_h:fin_h, ini_w:fin_w]

                # Save the patches

                save_path = os.path.join(savebase, 'Full_Resolution', continent)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('Saving patches in: ', save_path)
                for i in tqdm(range(patches_10.shape[0])):
                    dic = {'S2_10': np.moveaxis(patches_10[i].numpy().astype(np.uint16), 0, -1),
                           'S2_20': np.moveaxis(patches_20[i].numpy().astype(np.uint16), 0, -1),
                           'S2_60': np.moveaxis(patches_60[i].numpy().astype(np.uint16), 0, -1)}
                    io.savemat(os.path.join(save_path, time + '_' + country.upper() + '_' + str(i+1).zfill(4) + '.mat'), dic)


                # Downgrade protocol 20m
                bands_10_lp_20 = mtf(bands_10, 'S2-10', 2)
                bands_20_lp_20 = mtf(bands_20, 'S2-20', 2)
                bands_60_lp_20 = mtf(bands_60, 'S2-60', 2)

                bands_10_lr_20 = interpolate(bands_10_lp_20, scale_factor=1/2, mode='nearest-exact')
                bands_20_lr_20 = interpolate(bands_20_lp_20, scale_factor=1/2, mode='nearest-exact')
                bands_60_lr_20 = interpolate(bands_60_lp_20, scale_factor=1/2, mode='nearest-exact')

                if continent == 'Training' or continent == 'Validation':
                    # Patch extraction
                    patch_size = 60
                    patches_gt = patch_extraction(bands_20, patch_size * 6)
                    patches_10 = patch_extraction(bands_10_lr_20, patch_size * 6)
                    patches_20 = patch_extraction(bands_20_lr_20, patch_size * 3)
                    patches_60 = patch_extraction(bands_60_lr_20, patch_size)
                    # Select patches randomly
                    indexes = find_null_patches(patches_60)
                    patches_10 = patches_10[indexes]
                    patches_20 = patches_20[indexes]
                    patches_60 = patches_60[indexes]
                    patches_gt = patches_gt[indexes]

                    if patches_10.shape[0] > n_patches:
                        m = patches_10.shape[0] // 2
                        patches_10 = patches_10[m - n_patches//2:m + n_patches//2]
                        patches_20 = patches_20[m - n_patches//2:m + n_patches//2]
                        patches_60 = patches_60[m - n_patches//2:m + n_patches//2]
                        patches_gt = patches_gt[m - n_patches//2:m + n_patches//2]

                else:
                    patch_size = 400 // 2
                    if country == 'Alexandria':
                        ini_h = 1300 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 1000 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Beijing':
                        ini_h = 1100 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 600 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Berlin':
                        ini_h = 500 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 900 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'New_York':
                        ini_h = 1430 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 1250 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Paris':
                        ini_h = 1000 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 900 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Reynosa':
                        ini_h = 800 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 600 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Rome':
                        ini_h = 400 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 1100 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Tazoskij':
                        ini_h = 300 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 300 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Tokyo':
                        ini_h = 900 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 200 // 2
                        fin_w = ini_w + patch_size
                    elif country == 'Ulaanbaator':
                        ini_h = 700 // 2
                        fin_h = ini_h + patch_size
                        ini_w = 200 // 2
                        fin_w = ini_w + patch_size
                    else:
                        ini_h, fin_h = 0, 60
                        ini_w, fin_w = 0, 60
                    patches_gt = bands_20[:, :, ini_h * 6:fin_h * 6, ini_w * 6:fin_w * 6]
                    patches_10 = bands_10_lr_20[:, :, ini_h * 6:fin_h * 6, ini_w * 6:fin_w * 6]
                    patches_20 = bands_10_lr_20[:, :, ini_h * 3:fin_h * 3, ini_w * 3:fin_w * 3]
                    patches_60 = bands_60_lr_20[:, :, ini_h:fin_h, ini_w:fin_w]

                # Save the patches

                save_path = os.path.join(savebase, 'Reduced_Resolution', '20', continent)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('Saving patches in: ', save_path)
                for i in tqdm(range(patches_10.shape[0])):
                    dic = {'S2_10': np.moveaxis(patches_10[i].numpy(), 0, -1),
                           'S2_20': np.moveaxis(patches_20[i].numpy(), 0, -1),
                           'S2_60': np.moveaxis(patches_60[i].numpy(), 0, -1),
                           'S2_GT': np.moveaxis(patches_gt[i].numpy(), 0, -1)}
                    io.savemat(os.path.join(save_path, time + '_' + country.upper() + '_' + str(i+1).zfill(4) + '.mat'), dic)

                # Downgrade protocol 60m
                bands_10_lp_60 = mtf(bands_10, 'S2-10', 6)
                bands_20_lp_60 = mtf(bands_20, 'S2-20', 6)
                bands_60_lp_60 = mtf(bands_60, 'S2-60', 6)

                bands_10_lr_60 = interpolate(bands_10_lp_60, scale_factor=1/6, mode='nearest-exact')
                bands_20_lr_60 = interpolate(bands_20_lp_60, scale_factor=1/6, mode='nearest-exact')
                bands_60_lr_60 = interpolate(bands_60_lp_60, scale_factor=1/6, mode='nearest-exact')

                if continent == 'Training' or continent == 'Validation':
                    # Patch extraction
                    patch_size = 60
                    patches_gt = patch_extraction(bands_60, patch_size * 6)
                    patches_10 = patch_extraction(bands_10_lr_60, patch_size * 6)
                    patches_20 = patch_extraction(bands_20_lr_60, patch_size * 3)
                    patches_60 = patch_extraction(bands_60_lr_60, patch_size)
                    # Select patches randomly
                    indexes = find_null_patches(patches_60)
                    patches_10 = patches_10[indexes]
                    patches_20 = patches_20[indexes]
                    patches_60 = patches_60[indexes]
                    patches_gt = patches_gt[indexes]

                    if patches_10.shape[0] > n_patches:
                        m = patches_10.shape[0] // 2
                        patches_10 = patches_10[m - n_patches//2:m + n_patches//2]
                        patches_20 = patches_20[m - n_patches//2:m + n_patches//2]
                        patches_60 = patches_60[m - n_patches//2:m + n_patches//2]
                        patches_gt = patches_gt[m - n_patches//2:m + n_patches//2]

                else:
                    patch_size = 60

                if country == 'Alexandria':
                    ini_h = 1300 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 1000 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Beijing':
                    ini_h = 1100 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 600 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Berlin':
                    ini_h = 500 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 900 // 6
                    fin_w = ini_w + patch_size
                elif country == 'New_York':
                    ini_h = 1430 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 1250 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Paris':
                    ini_h = 1000 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 900 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Reynosa':
                    ini_h = 800 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 600 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Rome':
                    ini_h = 400 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 1100 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Tazoskij':
                    ini_h = 300 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 300 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Tokyo':
                    ini_h = 900 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 200 // 6
                    fin_w = ini_w + patch_size
                elif country == 'Ulaanbaator':
                    ini_h = 700 // 6
                    fin_h = ini_h + patch_size
                    ini_w = 200 // 6
                    fin_w = ini_w + patch_size
                else:
                    ini_h, fin_h = 0, 60
                    ini_w, fin_w = 0, 60
                patches_gt = bands_20[:, :, ini_h * 6:fin_h * 6, ini_w * 6:fin_w * 6]
                patches_10 = bands_10_lr_20[:, :, ini_h * 6:fin_h * 6, ini_w * 6:fin_w * 6]
                patches_20 = bands_10_lr_20[:, :, ini_h * 3:fin_h * 3, ini_w * 3:fin_w * 3]
                patches_60 = bands_60_lr_20[:, :, ini_h:fin_h, ini_w:fin_w]

                # Save the patches

                save_path = os.path.join(savebase, 'Reduced_Resolution', '60', continent)
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                print('Saving patches in: ', save_path)
                for i in tqdm(range(patches_10.shape[0])):
                    dic = {'S2_10': np.moveaxis(patches_10[i].numpy(), 0, -1),
                           'S2_20': np.moveaxis(patches_20[i].numpy(), 0, -1),
                           'S2_60': np.moveaxis(patches_60[i].numpy(), 0, -1),
                           'S2_GT': np.moveaxis(patches_gt[i].numpy(), 0, -1)}
                    io.savemat(os.path.join(save_path, time + '_' + country.upper() + '_' + str(i+1).zfill(4) + '.mat'), dic)




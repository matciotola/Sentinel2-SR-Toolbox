import os.path

import torch
from scipy import io
import numpy as np
from Utils.imresize_bicubic import imresize
# from Utils.interpolator_tools import interp23tap

def open_mat(path):
    # Open .mat file
    dic_file = io.loadmat(path)

    # Extract fields and convert them in float32 numpy arrays
    bands_10_np = dic_file['S2_10'].astype(np.float64)
    bands_20_np = dic_file['S2_20'].astype(np.float64)
    bands_60_np = dic_file['S2_60'].astype(np.float64)

    if 'S2_GT' in dic_file.keys():
        gt_np = dic_file['S2_GT'].astype(np.float64)
        gt = torch.from_numpy(np.moveaxis(gt_np, -1, 0)[None, :, :, :])
    else:
        gt = None

    # Convert numpy arrays to torch tensors
    bands_10 = torch.from_numpy(np.moveaxis(bands_10_np, -1, 0)[None, :, :, :])
    bands_20 = torch.from_numpy(np.moveaxis(bands_20_np, -1, 0)[None, :, :, :])
    bands_60 = torch.from_numpy(np.moveaxis(bands_60_np, -1, 0)[None, :, :, :])

    return bands_10, bands_20, bands_60, gt


def save_mat(image, path, key):
    if os.path.exists(path):
        dic = io.loadmat(path)
    else:
        dic = {}
    dic[key] = image
    io.savemat(path, dic)
    return

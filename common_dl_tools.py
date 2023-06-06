import os.path
from osgeo import gdal
import torch
from torch.utils.data import Dataset
import yaml
from recordclass import recordclass


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def open_config(file_path):
    yaml_file = read_yaml(file_path)
    return recordclass('config', yaml_file.keys())(*yaml_file.values())

def open_tiff(path):
    bands = gdal.Open(path)
    bands = bands.ReadAsArray().astype('float32')
    bands = torch.Tensor(bands)[None, :, :, :]
    return bands

def generate_paths (root, names):
    paths_10 = []
    paths_20 = []
    paths_60 = []

    for name in names:
        paths_10.append(os.path.join(root, '10', name + '.tif'))
        paths_20.append(os.path.join(root, '20', name + '.tif'))
        paths_60.append(os.path.join(root, '60', name + '.tif'))

    return paths_10, paths_20, paths_60

class TrainingDataset20m(Dataset):
    def __init__(self, bands_high_paths, bands_low_lr_paths, norm, input_prepro, get_patches, ratio=2, patches_size=33):
        super(TrainingDataset20m, self).__init__()

        bands_low_lr = []
        bands_high = []

        for i in range(len(bands_high_paths)):
            bands_high.append(open_tiff(bands_high_paths[i]))
            bands_low_lr.append(open_tiff(bands_low_lr_paths[i]))

        bands_high = torch.cat(bands_high, 0)
        bands_low_lr = torch.cat(bands_low_lr, 0)

        bands_high_downsampled, bands_low_downsampled, bands_low_lr = input_prepro(bands_high, bands_low_lr, ratio)

        bands_high_downsampled = norm(bands_high_downsampled)
        bands_low_downsampled = norm(bands_low_downsampled)
        bands_low_lr = norm(bands_low_lr)

        self.patches_high_lr = get_patches(bands_high_downsampled, patches_size)
        self.patches_low_lr = get_patches(bands_low_downsampled, patches_size)
        self.patches_low = get_patches(bands_low_lr, patches_size)

    def __len__(self):
        return self.patches_high_lr.shape[0]

    def __getitem__(self, index):
        return self.patches_high_lr[index], self.patches_low_lr[index], self.patches_low[index]


class TrainingDataset60m(Dataset):
    def __init__(self, bands_high_paths, bands_intermediate_lr_paths, bands_low_lr_paths, norm, input_prepro, get_patches, ratio=2, patches_size=33):
        super(TrainingDataset60m, self).__init__()

        bands_low_lr = []
        bands_intermediate_lr = []
        bands_high = []

        for i in range(len(bands_high_paths)):
            bands_high.append(open_tiff(bands_high_paths[i]))
            bands_intermediate_lr.append(open_tiff(bands_intermediate_lr_paths[i]))
            bands_low_lr.append(open_tiff(bands_low_lr_paths[i]))

        bands_high = torch.cat(bands_high, 0)
        bands_intermediate_lr = torch.cat(bands_intermediate_lr, 0)
        bands_low_lr = torch.cat(bands_low_lr, 0)

        bands_high_downsampled, bands_intermediate_downsampled, bands_low_downsampled, bands_low_lr = input_prepro(bands_high, bands_intermediate_lr, bands_low_lr, ratio)

        bands_high_downsampled = norm(bands_high_downsampled)
        bands_intermediate_downsampled = norm(bands_intermediate_downsampled)
        bands_low_downsampled = norm(bands_low_downsampled)
        bands_low_lr = norm(bands_low_lr)

        self.patches_high_lr = get_patches(bands_high_downsampled, patches_size)
        self.patches_intermediate_lr = get_patches(bands_intermediate_downsampled, patches_size)
        self.patches_low_lr = get_patches(bands_low_downsampled, patches_size)
        self.patches_low = get_patches(bands_low_lr, patches_size)

    def __len__(self):
        return self.patches_high_lr.shape[0]

    def __getitem__(self, index):
        return self.patches_high_lr[index], self.patches_intermediate_lr, self.patches_low_lr[index], self.patches_low[index]

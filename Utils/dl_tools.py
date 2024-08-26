import os.path
import torch
from torch.utils.data import Dataset
import yaml
from recordclass import recordclass

from Utils.load_save_tools import open_mat

def normalize(tensor):
    return tensor / (2 ** 16)


def denormalize(tensor):
    return tensor * (2 ** 16)


def read_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)


def open_config(file_path):
    yaml_file = read_yaml(file_path)
    return recordclass('config', yaml_file.keys())(*yaml_file.values())


def generate_paths(root, names):
    paths_10 = []
    paths_20 = []
    paths_60 = []

    for name in names:
        paths_10.append(os.path.join(root, '10', name + '.tif'))
        paths_20.append(os.path.join(root, '20', name + '.tif'))
        paths_60.append(os.path.join(root, '60', name + '.tif'))

    return paths_10, paths_20, paths_60


class TrainingDataset20m(Dataset):
    def __init__(self, bands_high_paths, bands_low_lr_paths, norm, input_prepro, get_patches, ratio=2, patches_size_lr=33, patch_size_hr=33):
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

class TrainingDataset60mRR(Dataset):
    def __init__(self, paths, norm):
        super(TrainingDataset60mRR, self).__init__()
        images_10_d60 = []
        images_20_d120 = []
        images_60_d360 = []
        images_60 = []

        for i in range(len(paths)):
            bands_10_d60, bands_20_d120, bands_60_d360, bands_60 = open_mat(paths[i])
            images_10_d60.append(bands_10_d60)
            images_20_d120.append(bands_20_d120)
            images_60_d360.append(bands_60_d360)
            images_60.append(bands_60)

        images_10_d60 = torch.cat(images_10_d60, 0)
        images_20_d120 = torch.cat(images_20_d120, 0)
        images_60_d360 = torch.cat(images_60_d360, 0)
        images_60 = torch.cat(images_60, 0)

        images_10_d60 = norm(images_10_d60)
        images_20_d120 = norm(images_20_d120)
        images_60_d360 = norm(images_60_d360)
        images_60 = norm(images_60)

        self.patches_10_d60 = images_10_d60
        self.patches_20_d120 = images_20_d120
        self.patches_60_d360 = images_60_d360
        self.patches_60 = images_60

    def __len__(self):
        return self.patches_60.shape[0]

    def __getitem__(self, index):
        return self.patches_10_d60[index], self.patches_20_d120[index], self.patches_60_d360, self.patches_60[index]


class TrainingDataset20mFR(Dataset):
    def __init__(self, paths, norm):
        super(TrainingDataset20mFR, self).__init__()

        images_20 = []
        images_10 = []

        for i in range(len(bands_high_paths)):
            bands_high.append(open_tiff(bands_high_paths[i]))
            bands_intermediate_lr.append(open_tiff(bands_intermediate_lr_paths[i]))
            bands_low_lr.append(open_tiff(bands_low_lr_paths[i]))

        for i in range(len(paths)):
            bands_10, bands_20, _, _ = open_mat(paths[i])
            images_10.append(bands_10)
            images_20.append(bands_20)

        images_10 = torch.cat(images_10, 0)
        images_20 = torch.cat(images_20, 0)

        images_10 = norm(images_10)
        images_20 = norm(images_20)

        self.patches_10 = images_10
        self.patches_20 = images_20

    def __len__(self):
        return self.patches_20.shape[0]

    def __getitem__(self, index):
        return self.patches_10[index], self.patches_20[index]
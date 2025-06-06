import os
import torch
import numpy as np
import inspect
from tqdm import tqdm
from scipy import io
from torch.nn import functional as func
from torch.utils.data import Dataset, DataLoader

from .network import S2_SSC_CNN_model
from Utils.dl_tools import open_config
from Utils.interpolator_tools import ideal_interpolator
from Utils.spectral_tools import mtf


def S2_SSC_CNN(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_lr = torch.clone(ordered_dict.ms_lr).float()

    config_path = os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    # Model

    model = S2_SSC_CNN_model(bands_10.shape[1], bands_lr.shape[1])

    # Train procedure (zero shot)
    history = zero_shot(model, bands_10, bands_lr, device, ordered_dict, config)
    if config.save_training_stats:
        if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)), 'Stats', 'S2_SSC_CNN')):
            os.makedirs(os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)), 'Stats', 'S2_SSC_CNN'))
        io.savemat(os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)), 'Stats', 'S2_SSC_CNN',
                                'Training_S2_SSC_CNN_' + ordered_dict.dataset + '.mat'), history)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)),
                                                  config.save_weights_path, 'S2_SCC_CNN.pth')))
    model.eval()

    # Inference

    with torch.no_grad():
        scale = bands_10.max()
        bands_10 = bands_10 / scale
        bands_lr = bands_lr / scale
        if ordered_dict.ratio == 2:
            bands_lr_up = ideal_interpolator(bands_lr, ordered_dict.ratio).float()
        else:
            bands_lr_up = func.interpolate(bands_lr, scale_factor=ordered_dict.ratio, mode='bicubic')

        bands_10 = bands_10.to(device)
        bands_lr_up = bands_lr_up.to(device)

        out = model(bands_10, bands_lr_up)
        fused = out * scale

    return fused.detach().cpu()


def generate_cube(x, size=16, stride=8):
    (bs, dep, row, col) = x.shape
    cube = []
    for r in range(0, row-size+1, stride):
        for c in range(0, col-size+1, stride):
            patch = x[:, :, r:r + size, c:c + size]
            cube.append(patch)

    return torch.cat(cube, 0)


class SimpleDataset(Dataset):
    def __init__(self, S2_10, S2_lr, S2_GT):
        self.S2_10 = S2_10
        self.S2_20 = S2_lr
        self.S2_GT = S2_GT

    def __len__(self):
        return self.S2_10.shape[0]

    def __getitem__(self, idx):
        return self.S2_10[idx], self.S2_20[idx], self.S2_GT[idx]


def zero_shot(model, bands_10, bands_lr, device, ordered_dict, config):

    temp_path = os.path.join(os.path.join(os.path.dirname(inspect.getfile(S2_SSC_CNN_model)), config.save_weights_path))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # Input pre-processing
    bands_10_lp = mtf(bands_10, 'S2-10', ordered_dict.ratio)
    bands_lr_lp = mtf(bands_lr, ordered_dict.sensor, ordered_dict.ratio)

    input_10 = func.interpolate(bands_10_lp, scale_factor=1/ordered_dict.ratio, mode='nearest-exact')
    # pad if necessary
    padded = False
    if bands_lr_lp.shape[2] % ordered_dict.ratio != 0 or bands_lr_lp.shape[3] % ordered_dict.ratio != 0:
        bands_lr_lp = func.pad(bands_lr_lp, (0, ordered_dict.ratio - bands_lr_lp.shape[2] % ordered_dict.ratio, 0, ordered_dict.ratio - bands_lr_lp.shape[3] % ordered_dict.ratio), mode='reflect')
        padded = True
    bands_lr_downsampled = func.interpolate(bands_lr_lp, scale_factor=1/ordered_dict.ratio, mode='nearest-exact')

    if ordered_dict.ratio == 2:
        input_lr = ideal_interpolator(bands_lr_downsampled, ordered_dict.ratio).float()
    else:
        input_lr = func.interpolate(bands_lr_downsampled, scale_factor=ordered_dict.ratio, mode='bicubic')
    if padded:
        input_lr = input_lr[:, :, :bands_lr.shape[2], :bands_lr.shape[3]]


    scale = input_10.max()
    input_10 = input_10 / scale
    input_lr = input_lr / scale
    input_gt = bands_lr / scale

    # Generate patches

    patches_10 = generate_cube(input_10)
    patches_lr = generate_cube(input_lr)
    patches_gt = generate_cube(input_gt)

    # Dataset
    train_dataset = SimpleDataset(patches_10, patches_lr, patches_gt)
    val_dataset = SimpleDataset(input_10, input_lr, input_gt)

    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    # Environment settings
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
    criterion = torch.nn.MSELoss()

    # Move to device
    model = model.to(device)
    criterion = criterion.to(device)

    # Training

    history_loss = []
    history_loss_val = []
    min_loss = np.inf

    pbar = tqdm(range(config.epochs))

    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0
        for i, data in enumerate(train_loader):
            p_10, p_lr, p_gt = data
            optimizer.zero_grad()
            p_10 = p_10.to(device)
            p_lr = p_lr.to(device)
            p_gt = p_gt.to(device)
            output = model(p_10, p_lr)
            loss = criterion(output, p_gt)

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        running_loss = running_loss / len(train_loader)
        history_loss.append(running_loss)

        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader):
                p_10, p_lr, p_gt = data
                p_10 = p_10.to(device)
                p_lr = p_lr.to(device)
                p_gt = p_gt.to(device)
                output = model(p_10, p_lr)
                loss = criterion(output, p_gt)
                running_loss_val += loss.item()

            running_loss_val = running_loss_val / len(val_loader)
            history_loss_val.append(running_loss_val)
            if running_loss_val < min_loss:
                min_loss = running_loss_val
                torch.save(model.state_dict(), os.path.join(temp_path, 'S2_SCC_CNN.pth'))

        pbar.set_postfix(
            {'Loss': running_loss, 'Val Loss': running_loss_val})

    history = {'loss': history_loss, 'val_loss': history_loss_val}

    return history

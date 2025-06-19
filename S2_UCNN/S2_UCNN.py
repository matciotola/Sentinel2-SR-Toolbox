import os
import torch
import numpy as np
import inspect
from tqdm import tqdm
from scipy import io

from .network import S2UCNN_model, DownSampler
from Utils.dl_tools import open_config



def S2_UCNN(ordered_dict):

    config_path = os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_20 = torch.clone(ordered_dict.bands_20).float()
    bands_60 = torch.clone(ordered_dict.bands_60).float()

    scale_10 = bands_10.max()
    scale_20 = bands_20.max()
    scale_60 = bands_60.max()

    scale = torch.max(torch.max(scale_10, scale_20), scale_60)

    bands_10_norm = bands_10 / scale
    bands_20_norm = bands_20 / scale
    bands_60_norm = bands_60 / scale
    splitted = False
    if config.split_test_image_for_prior and (bands_10_norm.shape[-1] > 600 or bands_10_norm.shape[-2] > 600):
        splitted = True
        kc_60, kh_60, kw_60 = bands_60_norm.shape[1], 100, 100  # kernel size
        dc_60, dh_60, dw_60 = bands_60_norm.shape[1], 100, 100  # stride

        kc_20, kh_20, kw_20 = bands_20_norm.shape[1], 300, 300  # kernel size
        dc_20, dh_20, dw_20 = bands_20_norm.shape[1], 300, 300  # stride

        kc_10, kh_10, kw_10 = bands_10_norm.shape[1], 600, 600  # kernel size
        dc_10, dh_10, dw_10 = bands_10_norm.shape[1], 600, 600  # stride

        bands_10_patches = bands_10_norm.unfold(1, kc_10, dc_10).unfold(2, kh_10, dh_10).unfold(3, kw_10, dw_10)
        unfold_shape = list(bands_10_patches.shape)
        bands_10_norm = bands_10_patches.contiguous().view(-1, kc_10, kh_10, kw_10)


        bands_20_patches = bands_20_norm.unfold(1, kc_20, dc_20).unfold(2, kh_20, dh_20).unfold(3, kw_20, dw_20)
        bands_20_norm = bands_20_patches.contiguous().view(-1, kc_20, kh_20, kw_20)

        bands_60_patches = bands_60_norm.unfold(1, kc_60, dc_60).unfold(2, kh_60, dh_60).unfold(3, kw_60, dw_60)
        bands_60_norm = bands_60_patches.contiguous().view(-1, kc_60, kh_60, kw_60)

    # Model

    model = S2UCNN_model()

    # Train procedure (zero shot)
    history = zero_shot(model, bands_10_norm, bands_20_norm, bands_60_norm, device, config)
    if config.save_training_stats:
        if not os.path.exists(os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)), 'Stats', 'S2_UCNN')):
            os.makedirs(os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)), 'Stats', 'S2_UCNN'))
        io.savemat(os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)), 'Stats', 'S2_UCNN',
                                'Training_S2_UCNN_' + ordered_dict.dataset + '.mat'), history)

    model.load_state_dict(torch.load(os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)),
                                                  config.save_weights_path, 'S2_UCNN.pth')))
    model.eval()

    # Inference
    with torch.no_grad():

        bands_10_norm = bands_10_norm.to(device)
        bands_20_norm = bands_20_norm.to(device)
        bands_60_norm = bands_60_norm.to(device)
        out = model(bands_10_norm, bands_20_norm, bands_60_norm)
        fused = out * scale


    if splitted:
        outputs_patches = fused
        unfold_shape[4] = bands_10.shape[1] + bands_20.shape[1] + bands_60.shape[1]
        outputs = outputs_patches.view(unfold_shape)
        output_c = unfold_shape[1] * unfold_shape[4]
        output_h = unfold_shape[2] * unfold_shape[5]
        output_w = unfold_shape[3] * unfold_shape[6]
        outputs = outputs.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        fused = outputs.view(1, output_c, output_h, output_w)

    if ordered_dict.ratio == 2:
        return fused[:, 4:10, :, :].detach().cpu()
    else:
        return fused[:, 10:, :, :].detach().cpu()


def zero_shot(model, bands_10, bands_20, bands_60, device, config):

    downsampler = DownSampler()

    temp_path = os.path.join(os.path.join(os.path.dirname(inspect.getfile(S2UCNN_model)), config.save_weights_path))
    if not os.path.exists(temp_path):
        os.makedirs(temp_path)

    # Environment settings
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(config.beta_1, config.beta_2))
    criterion = torch.nn.L1Loss()

    # Move to device
    model = model.to(device)
    downsampler = downsampler.to(device)
    criterion = criterion.to(device)

    # Training

    history_loss_10 = []
    history_loss_20 = []
    history_loss_60 = []

    min_loss = np.inf

    pbar = tqdm(range(config.epochs))
    model.train()
    for epoch in pbar:
        pbar.set_description('Epoch %d/%d' % (epoch + 1, config.epochs))
        running_loss_10 = 0.0
        running_loss_20 = 0.0
        running_loss_60 = 0.0
        running_loss = 0.0
        for bs in range(bands_10.shape[0]):
            # Model training
            optimizer.zero_grad()
            input_10 = bands_10[bs:bs+1, :, :, :].to(device)
            input_20 = bands_20[bs:bs+1, :, :, :].to(device)
            input_60 = bands_60[bs:bs+1, :, :, :].to(device)
            output = model(input_10, input_20, input_60)

            output_10 = output[:, :4, :, :]
            output_20 = output[:, 4:10, :, :]
            output_60 = output[:, 10:, :, :]

            output_20_lr, output_60_lr = downsampler(output_20, output_60)
            loss_10 = criterion(output_10, input_10)
            loss_20 = criterion(output_20_lr, input_20)
            loss_60 = criterion(output_60_lr, input_60)
            loss = loss_10 + loss_20 + loss_60

            loss.backward()
            optimizer.step()
            running_loss_10 += loss_10.item()
            running_loss_20 += loss_20.item()
            running_loss_60 += loss_60.item()
            running_loss += running_loss_10 + running_loss_20 + running_loss_60

        history_loss_10.append(running_loss_10)
        history_loss_20.append(running_loss_20)
        history_loss_60.append(running_loss_60)

        if running_loss < min_loss:
            min_loss = running_loss
            torch.save(model.state_dict(), os.path.join(temp_path, 'S2_UCNN.pth'))

        pbar.set_postfix(
            {'loss_10': running_loss_10, 'loss_20': running_loss_20, 'loss_60': running_loss_60})

    history = {'loss_10': history_loss_10, 'loss_20': history_loss_20, 'loss_60': history_loss_60}

    return history

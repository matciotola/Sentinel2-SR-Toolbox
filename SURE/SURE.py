import os
import inspect
import torch
from skimage.restoration import estimate_sigma
from tqdm import tqdm

from .network import S2AttentionNet
from .tools import get_s2psf, back_projx
from .loss import BPLoss, SURELoss

from Utils.dl_tools import open_config



def SURE(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).float()
    bands_20 = torch.clone(ordered_dict.bands_20).float()
    bands_60 = torch.clone(ordered_dict.bands_60).float()

    config_path = os.path.join(os.path.dirname(inspect.getfile(SURE)), 'config.yaml')
    config = open_config(config_path)

    #os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda:" + config.gpu_number if torch.cuda.is_available() else "cpu")

    scale_10 = torch.max(bands_10)
    scale_20 = torch.max(bands_20)
    scale_60 = torch.max(bands_60)

    scale = torch.max(torch.max(scale_10, scale_20), scale_60)

    bands_10_norm = bands_10 / scale
    bands_20_norm = bands_20 / scale
    bands_60_norm = bands_60 / scale

    psf_20 = get_s2psf('band20').to(device)
    psf_60 = get_s2psf('band60').to(device)

    sigma_10 = torch.tensor(estimate_sigma(bands_10_norm.squeeze(0).permute(1, 2, 0).cpu().numpy(), channel_axis=-1))
    sigma_20 = torch.tensor(estimate_sigma(bands_20_norm.squeeze(0).permute(1, 2, 0).cpu().numpy(), channel_axis=-1))
    sigma_60 = torch.tensor(estimate_sigma(bands_60_norm.squeeze(0).permute(1, 2, 0).cpu().numpy(), channel_axis=-1))

    bands_20_interpolated = back_projx(bands_20_norm.squeeze(0), psf_20, 2, config.cond)
    bands_60_interpolated = back_projx(bands_60_norm.squeeze(0), psf_60, 6, config.cond)

    # Network input definition

    net_input_ = torch.cat((
                           bands_10_norm.to(device),
                           bands_20_interpolated.unsqueeze(0).to(device),
                           bands_60_interpolated.unsqueeze(0).to(device)), dim=1)

    final_net_input = torch.clone(net_input_)

    if config.split_test_image_for_prior:
        kc_60, kh_60, kw_60 = bands_60_norm.shape[1], 50, 50  # kernel size
        dc_60, dh_60, dw_60 = bands_60_norm.shape[1], 50, 50  # stride

        kc_20, kh_20, kw_20 = bands_20_norm.shape[1], 150, 150  # kernel size
        dc_20, dh_20, dw_20 = bands_20_norm.shape[1], 150, 150  # stride

        kc_10, kh_10, kw_10 = bands_10_norm.shape[1], 300, 300  # kernel size
        dc_10, dh_10, dw_10 = bands_10_norm.shape[1], 300, 300  # stride

        kc_inp, kh_inp, kw_inp = net_input_.shape[1], 300, 300  # kernel size
        dc_inp, dh_inp, dw_inp = net_input_.shape[1], 300, 300  # stride

        bands_10_patches = bands_10_norm.unfold(1, kc_10, dc_10).unfold(2, kh_10, dh_10).unfold(3, kw_10, dw_10)
        bands_10_norm = bands_10_patches.contiguous().view(-1, kc_10, kh_10, kw_10).to(device)

        bands_20_patches = bands_20_norm.unfold(1, kc_20, dc_20).unfold(2, kh_20, dh_20).unfold(3, kw_20, dw_20)
        bands_20_norm = bands_20_patches.contiguous().view(-1, kc_20, kh_20, kw_20).to(device)

        bands_60_patches = bands_60_norm.unfold(1, kc_60, dc_60).unfold(2, kh_60, dh_60).unfold(3, kw_60, dw_60)
        bands_60_norm = bands_60_patches.contiguous().view(-1, kc_60, kh_60, kw_60).to(device)

        ms_lr_patches = net_input_.unfold(1, kc_inp, dc_inp).unfold(2, kh_inp, dh_inp).unfold(3, kw_inp, dw_inp)
        net_input_ = ms_lr_patches.contiguous().view(-1, kc_inp, kh_inp, kw_inp).to(device)

    # Network definition

    net = S2AttentionNet()
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, eps=1e-3, amsgrad=True)

    net = net.to(device)
    #net_input_ = net_input_.to(device)

    # Loss definition
    criterion_bp = BPLoss(psf_20, psf_60, config.cond).to(device)
    criterion_sure = SURELoss(psf_20, psf_60, sigma_10, sigma_20, sigma_60, config.cond).to(device)

    bands_10_norm = bands_10_norm.to(device)
    bands_20_norm = bands_20_norm.to(device)
    bands_60_norm = bands_60_norm.to(device)

    # Zero-Shot Fitting
    pbar = tqdm(range(config.num_iter))
    for _ in pbar:
        running_bp_loss = 0.0
        running_sure_loss = 0.0
        for bs in range(net_input_.shape[0]):

            optim.zero_grad()
            net_input = net_input_[bs:bs+1, :, :, :].to(device)
            outputs = net(net_input)

            loss_bp = criterion_bp(outputs, bands_10_norm[bs:bs+1, :, :, :], bands_20_norm[bs:bs+1, :, :, :], bands_60_norm[bs:bs+1, :, :, :])

            inp_e = torch.randn(net_input.shape, dtype=net_input.dtype).to(device)
            outputs_e = (net(net_input + config.ep * inp_e) - outputs) / config.ep
            div = criterion_sure(inp_e, outputs_e)

            loss = loss_bp + config.alpha * div
            loss.backward()
            optim.step()
            running_bp_loss += loss_bp.item()
            running_sure_loss += div.item()

        pbar.set_postfix({'BP Loss': running_bp_loss / net_input_.shape[0], 'SURE Loss': running_sure_loss / net_input_.shape[0]})

    net = net.eval()
    with torch.no_grad():
        outputs = net(final_net_input).detach().cpu() * scale

    outputs_20 = outputs[:, 4:10, :, :]
    outputs_60 = outputs[:, 10:, :, :]

    if ordered_dict.ratio == 2:
        return outputs_20
    else:
        return outputs_60

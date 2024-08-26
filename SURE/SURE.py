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

    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu_number
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    net_input = torch.cat((
                           bands_10_norm.to(device),
                           bands_20_interpolated.unsqueeze(0).to(device),
                           bands_60_interpolated.unsqueeze(0).to(device)), dim=1)

    # Network definition

    net = S2AttentionNet()
    optim = torch.optim.Adam(net.parameters(), lr=config.learning_rate, eps=1e-3, amsgrad=True)

    net = net.to(device)
    net_input = net_input.to(device)

    # Loss definition
    criterion_bp = BPLoss(psf_20, psf_60, config.cond).to(device)
    criterion_sure = SURELoss(psf_20, psf_60, sigma_10, sigma_20, sigma_60, config.cond).to(device)

    bands_10_norm = bands_10_norm.to(device)
    bands_20_norm = bands_20_norm.to(device)
    bands_60_norm = bands_60_norm.to(device)

    # Zero-Shot Fitting
    pbar = tqdm(range(config.num_iter))
    for _ in pbar:
        optim.zero_grad()
        outputs = net(net_input)

        loss_bp = criterion_bp(outputs, bands_10_norm, bands_20_norm, bands_60_norm)

        inp_e = torch.randn(net_input.shape, dtype=net_input.dtype).to(device)
        outputs_e = (net(net_input + config.ep * inp_e) - outputs) / config.ep
        div = criterion_sure(inp_e, outputs_e)

        loss = loss_bp + config.alpha * div
        loss.backward()
        optim.step()

        pbar.set_postfix({'BP Loss': loss_bp.item(), 'SURE Loss': div.item()})

    net = net.eval()
    with torch.no_grad():
        outputs = net(net_input).detach().cpu()

    outputs_20 = outputs[:, 4:10, :, :]
    outputs_60 = outputs[:, 10:, :, :]

    if ordered_dict.ratio == 2:
        return outputs_20 * scale
    else:
        return outputs_60 * scale

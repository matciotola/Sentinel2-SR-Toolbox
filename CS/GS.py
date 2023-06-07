import torch

from Utils.spectral_tools import LPFilterPlusDecTorch
from Utils.pansharpening_aux_tools import batch_cov, estimation_alpha


def GS(ordered_dict):

    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)

    bands_low_minus_avg = bands_low - bands_low.mean(dim=(-2, -1), keepdim=True)
    bands_low_minus_int = bands_low.mean(dim=1, keepdim=True) - bands_low.mean(dim=(1, 2, 3), keepdim=True)
    bands_high = (bands_high - bands_high.mean(dim=(1, 2, 3), keepdim=True)) * (
                bands_low_minus_int.std(dim=(1, 2, 3)) / bands_high.std(dim=(1, 2, 3))) + bands_low_minus_int.mean(
        dim=(2, 3), keepdim=True)
    chain = torch.cat([torch.flatten(bands_low_minus_int.repeat(1, bands_low.shape[1], 1, 1), start_dim=-2),
                       torch.flatten(bands_low_minus_avg, start_dim=-2)], dim=1)
    cc = batch_cov(chain.transpose(1, 2))
    g = torch.ones(1, bands_low.shape[1] + 1, 1)
    g[:, 1:, :] = cc[:, 0, 1] / bands_low_minus_int.var(dim=(1, 2, 3))

    delta = bands_high - bands_low_minus_int
    delta_flatten = torch.flatten(delta, start_dim=-2)
    delta_r = delta_flatten.repeat(1, bands_low.shape[1] + 1, 1)

    v1 = torch.flatten(bands_low_minus_int, start_dim=-2)
    v2 = torch.flatten(bands_low_minus_avg, start_dim=-2)

    V = torch.cat([v1, v2], dim=1)

    gm = g[:, 0, 0].repeat(V.shape)

    V_hat = V + delta_r * gm

    V_hat = torch.reshape(V_hat[:, 1:, :], bands_low.shape)

    fused = V_hat - V_hat.mean(dim=(-2, -1), keepdim=True) + bands_low.mean(dim=(-2, -1), keepdim=True)

    return fused


def GSA(ordered_dict):

    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)
    ratio = ordered_dict.ratio

    bands_low_interp_minus_avg = bands_low - torch.mean(bands_low, dim=(2, 3), keepdim=True)
    bands_low_minus_avg = bands_low_lr - torch.mean(bands_low_lr, dim=(2, 3), keepdim=True)
    bands_high_minus_avg = bands_high - torch.mean(bands_high, dim=(2, 3), keepdim=True)

    bands_high_minus_avg = LPFilterPlusDecTorch(bands_high_minus_avg, ratio)
    B, _, h, w = bands_low_minus_avg.shape
    _, _, H, W = bands_low_interp_minus_avg.shape
    alpha_input = torch.cat([bands_low_minus_avg,
                             torch.ones((B, 1, h, w), dtype=bands_low_minus_avg.dtype,
                                        device=bands_low_minus_avg.device)], dim=1)
    alphas = estimation_alpha(alpha_input, bands_high_minus_avg)
    alpha_out = torch.cat([bands_low_interp_minus_avg,
                           torch.ones((B, 1, H, W), dtype=bands_low_minus_avg.dtype,
                                      device=bands_low_minus_avg.device)], dim=1)
    img = torch.sum(alpha_out * alphas, dim=1, keepdim=True)

    bands_low_minus_int = img - torch.mean(img, dim=(2, 3))

    chain = torch.cat([torch.flatten(bands_low_minus_int.repeat(1, bands_low_lr.shape[1], 1, 1), start_dim=-2),
                       torch.flatten(bands_low_interp_minus_avg, start_dim=-2)], dim=1)
    cc = batch_cov(chain.transpose(1, 2))
    g = torch.ones(1, bands_low_lr.shape[1] + 1, 1)
    g[:, 1:, :] = cc[:, 0, 1] / bands_low_minus_int.var(dim=(1, 2, 3))

    bands_high = bands_high - torch.mean(bands_high, dim=(2, 3))

    delta = bands_high - bands_low_minus_int

    delta_flatten = torch.flatten(delta, start_dim=-2)
    delta_r = delta_flatten.repeat(1, bands_low_lr.shape[1] + 1, 1)

    v1 = torch.flatten(bands_low_minus_int, start_dim=-2)
    v2 = torch.flatten(bands_low_interp_minus_avg, start_dim=-2)

    V = torch.cat([v1, v2], dim=1)

    gm = g[:, 0, 0].repeat(V.shape)

    V_hat = V + delta_r * gm
    V_hat = torch.reshape(V_hat[:, 1:, :], bands_low.shape)

    fused = V_hat - V_hat.mean(dim=(-2, -1), keepdim=True) + bands_low.mean(dim=(-2, -1), keepdim=True)

    return fused

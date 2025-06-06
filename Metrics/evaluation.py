import torch
import torch.nn.functional as F
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch

from . import metrics_rr as rr
from . import metrics_fr as fr
from .coregistration import fineshift

def evaluation_rr(fused, gt, ratio, flag_cut=True, dim_cut=11, L=16):
    if flag_cut:
        fused = fused[:, :, dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut]
        gt = gt[:, :, dim_cut - 1:-dim_cut, dim_cut - 1:-dim_cut]

    fused = torch.clip(fused, 0, 2 ** L)

    ergas = rr.ERGAS(ratio).to(fused.device)
    sam = rr.SAM().to(fused.device)
    q2n = rr.Q2n().to(fused.device)

    ergas_index, _ = ergas(fused, gt)
    sam_index, _ = sam(fused, gt)
    q2n_index, _ = q2n(fused, gt)

    return ergas_index.item(), sam_index.item(), q2n_index.item()


def evaluation_fr(fused, bands_10, bands_lr, ratio, sensor):
    starting = 1
    sigma = max(4, ratio)

    kernel = mtf_kernel_to_torch(gen_mtf(ratio, sensor, nbands=fused.shape[1]))

    filter = torch.nn.Conv2d(fused.shape[1], fused.shape[1], kernel_size=kernel.shape[2], groups=fused.shape[1],
                             padding='same', padding_mode='replicate', bias=False)
    filter.weight = torch.nn.Parameter(kernel.type(fused.dtype).to(fused.device))
    filter.weight.requires_grad = False

    fused_lp = filter(fused)
    fused_lp = fineshift(fused_lp, 1, 1, fused_lp.device)
    fused_lr = fused_lp[:, :, starting::ratio, starting::ratio]
    ergas = rr.ERGAS(ratio).to(fused.device)
    sam = rr.SAM().to(fused.device)
    q2n = rr.Q2n().to(fused.device)
    d_rho = fr.D_rho(sigma).to(fused.device)

    # Spectral Assessment
    ergas_index, _ = ergas(fused_lr, bands_lr)
    sam_index, _ = sam(fused_lr, bands_lr)
    q2n_index, _ = q2n(fused_lr, bands_lr)
    d_lambda_index = 1 - q2n_index

    # Spatial Assessment
    d_rho_index, _ = d_rho(fused, bands_10)

    return (d_lambda_index.item(), ergas_index.item(), sam_index.item(), d_rho_index.item())

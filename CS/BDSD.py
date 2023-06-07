import torch
from torch.nn.functional import pad
from torchvision.transforms import InterpolationMode as Inter
from torchvision.transforms.functional import resize

from Utils.spectral_tools import mtf


def BDSD(ordered_dict):
    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)
    indexes = ordered_dict.bands_selection
    ratio = ordered_dict.ratio
    ind = ordered_dict.ind

    bands_low_lr_lp, bands_low_lr, bands_high_lr = prepro_BDSD(bands_low, bands_high, [indexes[ind]], 2, bands_low.shape[-1], ordered_dict.mtf_high_name, ordered_dict.mtf_low_name)
    fused = []
    for i in range(bands_low.shape[1]):
        gamma = gamma_calculation_BDSD(bands_low_lr_lp[:, i, None, :, :], bands_low_lr[:, i, None, :, :], bands_high_lr, ratio, bands_low.shape[-1])
        fused.append(fuse_BDSD(bands_low[:, i, None, :, :], bands_high, gamma, ratio, bands_low.shape[-1]))

    fused = torch.cat(fused, dim=1)

    return fused


def prepro_BDSD(bands_low, bands_high, indexes, ratio, block_size, high_sensor='S2_10', low_sensor='S2_20'):
    assert (block_size % 2 == 0), f"block size for local estimation must be even"
    assert (block_size > 1), f"block size for local estimation must be positive"
    assert (block_size % ratio == 0), f"block size must be multiple of ratio"

    _, _, N, M = bands_high.shape
    _, _, n, m = bands_low.shape
    assert (N % block_size == 0) and (
            M % block_size == 0), f"height and widht of 10m bands must be multiple of the block size"

    bands_low = bands_low.float()
    bands_high = bands_high.float()
    starting = ratio // 2

    bands_high_lp = mtf(bands_high, high_sensor, ratio, indexes)
    bands_high_lr = bands_high_lp[:, :, starting::ratio, starting::ratio]
    bands_low_lr = resize(bands_low, [n // ratio, m // ratio], interpolation=Inter.BICUBIC, antialias=True)

    bands_low_lr_lp = mtf(bands_low_lr, low_sensor, ratio)

    return bands_low_lr_lp, bands_low_lr, bands_high_lr


def gamma_calculation_BDSD(bands_low_lr_lp, bands_low_lr, bands_high_lr, ratio, block_size):
    alg_input = torch.cat([bands_low_lr_lp, bands_low_lr, bands_high_lr], dim=1)
    gamma = blockproc(alg_input, (int(block_size // ratio), int(block_size // ratio)), estimate_gamma_cube, block_size,
                      ratio)
    return gamma


def fuse_BDSD(bands_low, bands_high, gamma, ratio, block_size):
    ## Fusion

    inputs = torch.cat([bands_low, bands_high, gamma], dim=1)

    fused = blockproc(inputs, (block_size, block_size), compH_injection, block_size, ratio)

    return fused


def blockproc(A, dims, func, S, ratio):
    results_row = []
    for y in range(0, A.shape[-2], dims[0]):
        results_cols = []
        for x in range(0, A.shape[-1], dims[1]):
            results_cols.append(func(A[:, :, y:y + dims[0], x:x + dims[1]], S))
        results_row.append(torch.vstack(results_cols))
    results_row = torch.vstack(results_row)
    """
    rows = []
    for y in range(results_row.shape[-2]):
        cols = []
        for x in range(results_row.shape[-1]):
            cols.append(results_row[ :, :, y, x])
        rows.append(torch.vstack(cols))
    rows = torch.vstack(rows)
    """
    return results_row


def estimate_gamma_cube(img, S):
    Nb = (img.shape[1] - 1) // 2

    low_lp_d = img[:, :Nb, :, :]
    low_lp = img[:, Nb:2 * Nb, :, :]
    high_lp_d = img[:, 2 * Nb, None, :, :]

    Hd = []

    Hd.append(torch.flatten(low_lp_d, start_dim=2))

    Hd.append(torch.flatten(high_lp_d, start_dim=2))
    Hd = torch.cat(Hd, dim=1).transpose(1, 2)

    Hd_p = torch.adjoint(Hd)
    HHd = torch.matmul(Hd_p, Hd)
    B = torch.linalg.solve(HHd, Hd_p)

    gamma = []
    for k in range(Nb):
        b = low_lp[:, k, :, :]
        bd = low_lp_d[:, k, :, :]
        b = torch.flatten(b, start_dim=1)[:, :, None]
        bd = torch.flatten(bd, start_dim=1)[:, :, None]
        gamma.append(torch.matmul(B, (b - bd)))
    gamma = torch.vstack(gamma)[:, None, :, :]
    gamma = pad(gamma, (0, S - Nb, 0, S - Nb - 1))

    return gamma


def compH_injection(img, S):
    Nb = img.shape[1] - 2
    lowres = img[:, :-2, :, :]
    highres = img[:, -2, :, :][:, None, :, :]
    gamma = img[:, -1, :, :]

    # Compute H
    Hlow = torch.flatten(lowres, start_dim=-2)
    Hhigh = torch.flatten(highres, start_dim=-2)

    H = torch.cat([Hlow, Hhigh], dim=1).transpose(1, 2)

    g = gamma[:, :Nb + 1, :Nb]

    mul_Hg = torch.matmul(H, g)
    hlow_en = Hlow + mul_Hg.transpose(1, 2)
    hlow_en = torch.reshape(hlow_en, lowres.shape)
    return hlow_en

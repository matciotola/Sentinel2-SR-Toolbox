import torch
from torch.nn.functional import pad
from spectral_tools import mtf
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter
from torch import nn

from spectral_tools import LPFilterPlusDecTorch, LPfilterGauss
from aux_tools import estimation_alpha, regress, batch_cov


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


def GS(bands_low, band_high):
    bands_low_minus_avg = bands_low - bands_low.mean(dim=(-2, -1), keepdim=True)
    bands_low_minus_int = bands_low.mean(dim=1, keepdim=True) - bands_low.mean(dim=(1, 2, 3), keepdim=True)
    band_high = (band_high - band_high.mean(dim=(1, 2, 3), keepdim=True)) * (
                bands_low_minus_int.std(dim=(1, 2, 3)) / band_high.std(dim=(1, 2, 3))) + bands_low_minus_int.mean(
        dim=(2, 3), keepdim=True)
    chain = torch.cat([torch.flatten(bands_low_minus_int.repeat(1, bands_low.shape[1], 1, 1), start_dim=-2),
                       torch.flatten(bands_low_minus_avg, start_dim=-2)], dim=1)
    cc = batch_cov(chain.transpose(1, 2))
    g = torch.ones(1, bands_low.shape[1] + 1, 1)
    g[:, 1:, :] = cc[:, 0, 1] / bands_low_minus_int.var(dim=(1, 2, 3))

    delta = band_high - bands_low_minus_int
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


def GSA(bands_low, bands_high, bands_low_orig, ratio):
    bands_low_interp_minus_avg = bands_low - torch.mean(bands_low, dim=(2, 3), keepdim=True)
    bands_low_minus_avg = bands_low_orig - torch.mean(bands_low_orig, dim=(2, 3), keepdim=True)
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

    chain = torch.cat([torch.flatten(bands_low_minus_int.repeat(1, bands_low_orig.shape[1], 1, 1), start_dim=-2),
                       torch.flatten(bands_low_interp_minus_avg, start_dim=-2)], dim=1)
    cc = batch_cov(chain.transpose(1, 2))
    g = torch.ones(1, bands_low_orig.shape[1] + 1, 1)
    g[:, 1:, :] = cc[:, 0, 1] / bands_low_minus_int.var(dim=(1, 2, 3))

    bands_high = bands_high - torch.mean(bands_high, dim=(2, 3))

    delta = bands_high - bands_low_minus_int

    delta_flatten = torch.flatten(delta, start_dim=-2)
    delta_r = delta_flatten.repeat(1, bands_low_orig.shape[1] + 1, 1)

    v1 = torch.flatten(bands_low_minus_int, start_dim=-2)
    v2 = torch.flatten(bands_low_interp_minus_avg, start_dim=-2)

    V = torch.cat([v1, v2], dim=1)

    gm = g[:, 0, 0].repeat(V.shape)

    V_hat = V + delta_r * gm
    V_hat = torch.reshape(V_hat[:, 1:, :], bands_low.shape)

    fused = V_hat - V_hat.mean(dim=(-2, -1), keepdim=True) + bands_low.mean(dim=(-2, -1), keepdim=True)

    return fused


def BT_H(bands_low, bands_high, ratio):

    min_bands_low = torch.amin(bands_low, dim=(2, 3), keepdim=True)

    bands_high_lp = LPfilterGauss(bands_high, ratio)
    alphas = estimation_alpha(bands_low, bands_high_lp)

    img = torch.sum((bands_low - min_bands_low) * alphas, dim=1, keepdim=True)

    img_hr = (bands_high - torch.mean(bands_high_lp, dim=(2, 3), keepdim=True)) / (
                torch.std(img, dim=(2, 3), keepdim=True) / torch.std(bands_high_lp, dim=(2, 3),
                                                                     keepdim=True)) + torch.mean(img, dim=(2, 3),
                                                                                                 keepdim=True)

    bands_low_minus_min = bands_low - min_bands_low
    bands_low_minus_min = torch.clip(bands_low_minus_min, 0, bands_low_minus_min.max())

    fused = bands_low_minus_min * (img_hr / (img + torch.finfo(torch.float32).eps)).repeat(1, bands_low.shape[1], 1,
                                                                                           1) + min_bands_low

    return fused



def PRACS(bands_low, bands_high, ratio, beta=0.95):
    B, C, H, W = bands_low.shape

    bands_low_hm = (bands_low - torch.mean(bands_low, dim=(2, 3), keepdim=True) + torch.mean(bands_high, dim=(2, 3),
                                                                                             keepdim=True) / torch.std(
        bands_high, dim=(2, 3), keepdim=True) * torch.std(bands_low, dim=(2, 3), keepdim=True)) * torch.std(bands_high,
                                                                                                            dim=(2, 3),
                                                                                                            keepdim=True) / torch.std(
        bands_low, dim=(2, 3), keepdim=True)
    bands_low_hm = torch.clip(bands_low_hm, 0, bands_low_hm.max())

    bands_high_lp = resize(
        resize(bands_high, [bands_high.shape[2] // ratio, bands_high.shape[3] // ratio], interpolation=Inter.BICUBIC,
               antialias=True), [bands_high.shape[2], bands_high.shape[3]], interpolation=Inter.BICUBIC, antialias=True)

    bb = torch.cat([torch.ones((B, 1, H, W), dtype=bands_low_hm.dtype, device=bands_low_hm.device), bands_low_hm],
                   dim=1)
    bb = torch.flatten(bb, start_dim=2).transpose(1, 2)
    bands_high_lp_f = torch.flatten(bands_high_lp, start_dim=2).transpose(1, 2)
    alpha = regress(bands_high_lp_f, bb)

    aux = torch.matmul(bb, alpha)

    img = torch.reshape(aux, (B, 1, H, W))

    corr_coeffs = []
    for b in range(B):
        corr_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img[b, :, :, :], bands_low_hm[b, c, None, :, :]], dim=0), start_dim=1)
            corr_bands.append(torch.corrcoef(stack)[0, 1])
        corr_bands = torch.vstack(corr_bands)
        corr_coeffs.append(corr_bands[None, :, :])

    corr_coeffs = torch.vstack(corr_coeffs)[:, :, :, None]

    img_h = corr_coeffs * bands_high.repeat(1, C, 1, 1) + (1 - corr_coeffs) * bands_low_hm

    img_h_lp = resize(
        resize(img_h, [bands_high.shape[2] // ratio, bands_high.shape[3] // ratio], interpolation=Inter.BICUBIC,
               antialias=True), [bands_high.shape[2], bands_high.shape[3]], interpolation=Inter.BICUBIC, antialias=True)
    img_h_lp_f = torch.flatten(img_h_lp, start_dim=2)
    gamma = []
    for i in range(C):
        aux = img_h_lp_f[:, i, None, :].transpose(1, 2)
        gamma.append(regress(aux, bb))
    gamma = torch.cat(gamma, dim=-1)

    img_prime = []
    for i in range(C):
        aux = torch.bmm(bb, gamma[:, :, i, None])
        img_prime.append(torch.reshape(aux, (B, 1, H, W)))

    img_prime = torch.cat(img_prime, 1)

    delta = img_h - img_prime - (
                torch.mean(img_h, dim=(2, 3), keepdim=True) - torch.mean(img_prime, dim=(2, 3), keepdim=True))

    aux3 = torch.mean(torch.std(bands_low, dim=(2, 3)), dim=1)

    w = []
    for b in range(B):
        w_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img_prime[b, c, None, :, :], bands_low[b, c, None, :, :]], dim=0),
                                  start_dim=1)
            w_bands.append(torch.corrcoef(stack)[0, 1])
        w_bands = torch.vstack(w_bands)
        w.append(w_bands[None, :, :, None])
    w = torch.vstack(w)
    w = beta * w * torch.std(bands_low, dim=(2, 3), keepdim=True) / aux3

    L_i = []
    for b in range(B):
        L_i_bands = []
        for c in range(C):
            stack = torch.flatten(torch.cat([img[b, 0, None, :, :], bands_low[b, c, None, :, :]], dim=0), start_dim=1)

            rho = torch.corrcoef(stack)[0, 1]
            aux = 1 - abs(1 - rho * bands_low[b, c, None, :, :] / img_prime[b, c, None, :, :])

            L_i_bands.append(torch.reshape(aux, (1, 1, H, W)))
        L_i_bands = torch.cat(L_i_bands, 1)
        L_i.append(L_i_bands)
    L_i = torch.cat(L_i, 0)

    det = w * L_i * delta
    fused = bands_low + det

    return fused


if __name__ == '__main__':
    from scipy import io
    import numpy as np
    from interpolator_tools import interp23tap_torch
    from show_results import show

    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_3.mat')

    pan = temp['I_PAN'].astype(np.float32)
    ms = temp['I_MS_LR'].astype(np.float32).transpose(2, 0, 1)

    pan = torch.tensor(pan)[None, None, :, :]
    ms = torch.tensor(ms)[None, :, :, :]
    ratio = 4
    ms_exp = interp23tap_torch(ms, 4, ms.device).float()

    fused = GSA(ms_exp, pan, ms, ratio)
    f = fused.detach().cpu().numpy()
    b10 = pan.detach().cpu().numpy()
    b20 = ms.detach().cpu().numpy()

    f = np.moveaxis(np.squeeze(f), 0, -1)
    b20 = np.moveaxis(np.squeeze(b20), 0, -1)
    b10 = np.squeeze(b10)

    show(b20, b10, f, ratio=ratio, method='GSA')

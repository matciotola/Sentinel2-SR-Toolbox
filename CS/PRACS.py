import torch

from Utils.pansharpening_aux_tools import regress
from Utils.imresize_bicubic import imresize


def PRACS(ordered_dict, beta=0.95):
    bands_low = torch.clone(ordered_dict.bands_low)
    bands_high = torch.clone(ordered_dict.bands_high)
    ratio = ordered_dict.ratio

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


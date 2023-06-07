import torch
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
from torch.nn.functional import conv2d, pad

def selection(bands_high, bands_low, ratio, sat_name='S2-10'):

    h = mtf_kernel_to_torch(gen_mtf(ratio, sat_name)).to(bands_high.device)

    # Decimation
    bands_high_lpf = conv2d(pad(bands_high, (h.shape[-1] // 2, h.shape[-1] // 2, h.shape[-1] // 2, h.shape[-1] // 2), mode='replicate'), h, groups=bands_high.shape[1])
    bands_high_lr = bands_high_lpf[:, :, 1::ratio, 1::ratio]

    # CorrCoeff Calculation

    corr_coeff_global = []

    for i in range(bands_high_lr.shape[1]):
        selected_high = bands_high_lr[:, i, :, :].flatten()[None, :]
        corr_coeff_selected_high = []
        for j in range(bands_low.shape[1]):
            selected_low = bands_low[:, j, :, :].flatten()[None, :]
            X = torch.cat([selected_high, selected_low], dim=0)
            corr_coeff_selected_high.append(torch.corrcoef(X)[0,1])
        corr_coeff_global.append(torch.Tensor(corr_coeff_selected_high))

    corr_coeff_global = torch.vstack(corr_coeff_global)

    _, selected_bands = torch.max(corr_coeff_global, dim=0)
    high_composed = bands_high[:, selected_bands, :, :]

    return high_composed, selected_bands

def vectorized_alpha_estimation (a, b):

    a_fl = torch.flatten(a, start_dim=-2).transpose(2, 1)
    b_fl = torch.flatten(b, start_dim=-2).transpose(2, 1)
    alpha = torch.linalg.lstsq(a_fl, b_fl)
    alpha = alpha.solution[:, :, :, None]

    return alpha
def synthesize(bands_high, bands_low,  ratio, sat_name='S2-10'):

    scale = 1 / ratio

    h = mtf_kernel_to_torch(gen_mtf(ratio, sat_name)).type(bands_high.dtype).to(bands_high.device)
    bands_high_lpf = conv2d(
                            pad(bands_high, (h.shape[-1] // 2,
                                             h.shape[-1] // 2,
                                             h.shape[-1] // 2,
                                             h.shape[-1] // 2),
                                mode='replicate'),
                            h,
                            groups=bands_high.shape[1])
    bands_high_lr = bands_high_lpf[:, :, 1::ratio, 1::ratio]

    padding_high_lr = torch.ones((bands_high_lr.shape[0], 1, bands_high_lr.shape[2], bands_high_lr.shape[3]), dtype=bands_high_lr.dtype,
                              device=bands_high_lr.device)
    bands_high_lr_plus = torch.cat([padding_high_lr, bands_high_lr], dim=1)

    alpha = vectorized_alpha_estimation(bands_high_lr_plus.double(), bands_low.double()).float()

    bands_high_exp = torch.ones((bands_high.shape[0], bands_high.shape[1] + 1, bands_high.shape[2], bands_high.shape[3]),
                                 dtype=bands_high.dtype,
                                 device=bands_high.device)

    bands_high_exp[:, 1:, :, :] = bands_high

    synthesized_bands = []
    for i in range(alpha.shape[2]):
        a = alpha[:, :, i, :, None]
        synthesized_bands.append(torch.sum(bands_high_exp * a, dim=1, keepdim=True))

    synthesized_bands = torch.cat(synthesized_bands, dim=1)

    selected_bands = list(range(bands_low.shape[1]))

    return synthesized_bands, selected_bands


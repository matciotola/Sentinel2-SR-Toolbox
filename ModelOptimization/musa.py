import torch
from torch import nn
import math
from tools import *
from Utils.imresize_bicubic import imresize
import pywt
from Utils.bm3d import bm3d_rgb_mod

def MuSA(ordered_dict):
    bands_high = torch.clone(ordered_dict.bands_high)
    bands_intermediate_lr = torch.clone(ordered_dict.bands_intermediate)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    if bands_low_lr.shape[1] == 3:
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    # HyperParameters
    bands_10_index = [1, 2, 3, 7]
    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]

    mu = 0.6
    nb = 12
    niters = 130
    iters_pp = 30
    tau = 0.005
    p = 5
    limsub = 9
    dx = 12
    dy = 12

    mtf = [0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.33, 0.26, 0.22, 0.23]
    ratio = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]

    sdf = torch.tensor(ratio) * torch.sqrt(-2 * torch.log(torch.tensor(mtf)) / torch.pi ** 2)
    sdf[torch.tensor(ratio) == 1] = 0

    _, _, nl, nc = bands_high.shape
    n = nl * nc

    # Normalize data

    bands_high_normalized, m_high = normalize_data(bands_high)
    bands_intermediate_lr_normalized, m_intermediate_lr = normalize_data(bands_intermediate_lr)
    bands_low_lr_normalized, m_low_lr = normalize_data(bands_low_lr)

    y_im_2 = generate_stack(bands_high_normalized, bands_intermediate_lr_normalized, bands_low_lr_normalized)
    global_mean = generate_mean_stack(m_high, m_intermediate_lr, m_low_lr)

    # Define blurring operators

    fbm = create_conv_kernel(sdf, ratio, nl, nc, nb, dx, dy).to(bands_high_normalized.device)
    fbm2 = create_conv_kernel_subspace(sdf, nl, nc, nb, dx, dy).to(bands_high_normalized.device)

    # Calculate subspace E

    bands_high = bands_high_normalized
    bands_intermediate = imresize(bands_intermediate_lr_normalized, 2)
    bands_low = imresize(bands_low_lr_normalized, 6)

    y1im = torch.cat(generate_stack(bands_high, bands_intermediate, bands_low), dim=1)

    y1 = conv2mat(y1im)

    y2 = conv_cm(y1, fbm2, nl, nc, y1.shape[1])
    y2_im = conv2im(y2, nl, nc, y2.shape[1])
    y2_n = conv2mat(y2_im[:, :, limsub:-limsub, limsub:-limsub])

    # SVD analysis
    matrix = y2_n @ y2_n.transpose(1, 2) / n
    e2, _, _ = torch.linalg.svd(matrix)

    e = e2[:, :, :p]

    # Subsampling

    _, y = create_subsampling(y_im_2, ratio, nl, nc, y2_im.shape[1])

    y_im = conv2im(y, nl, nc, nb)
    p = e.shape[2]
    fbmc = torch.conj(fbm)
    bty = conv_cm(y, fbmc, nl, nc, nb)

    iff = 0 * fbm

    # build the inverse filter in frequency domain with subsampling

    for i in range(nb):
        f_im = torch.abs(fbm[:, i, None, :, :] ** 2)
        kc, kh, kw = 1, nl // ratio[i], nc // ratio[i]  # kernel size
        dc, dh, dw = 1, nl // ratio[i], nc // ratio[i]  # stride
        patches = f_im.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.shape)
        f_patches = patches.contiguous().view(-1, kc, kh, kw).flatten(start_dim=2)
        p2 = f_patches.shape[0]
        f_patches = 1 / (torch.sum(f_patches, dim=0, keepdim=True) / mu + ratio[i] ** 2)

        aux = f_patches.repeat(p2, 1, 1).view(unfold_shape)
        aux_c = unfold_shape[1] * unfold_shape[4]
        aux_h = unfold_shape[2] * unfold_shape[5]
        aux_w = unfold_shape[3] * unfold_shape[6]
        aux = aux.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        aux = aux.view(1, aux_c, aux_h, aux_w)

        aux2 = f_im * aux
        iff[:, i:i + 1, :, :] = aux2

    # Solver

    z = torch.zeros([p, n], dtype=e.dtype, device=e.device)
    v1 = torch.zeros([nb, n], dtype=e.dtype, device=e.device)
    d1 = torch.clone(v1)
    v2 = torch.clone(z)
    d2 = torch.clone(v2)

    for i in range(niters):

        z = 0.5 * (e.transpose(-1, -2) @ (v1 + d1) + (v2 + d2))

        nu1 = e @ z - d1
        aux = bty + mu * nu1
        v1 = aux / mu - conv_cm(aux, iff, nl, nc, nb) / mu ** 2

        nu2 = z - d2
        v2 = torch.clone(nu2)

        if i > iters_pp:
            v2im = torch.reshape(nu2, (1, p, nc, nl)).transpose(2, 3)
            v2im_new = []
            for k in range(p):
                auxim1 = v2im[:, k:k+1, :, :]

                max_im1 = auxim1.max()
                min_im1 = auxim1.min()
                scale1 = max_im1 - min_im1
                auxim1 = (auxim1 - min_im1) / scale1

                y_im_clean = synthetize_pan(auxim1, y_im[:, bands_10_index, :, :])
                max_im = y_im_clean.max()
                min_im = y_im_clean.min()
                scale = max_im - min_im
                y_im_clean = (y_im_clean - min_im) / scale

                y_rgb = torch.cat([y_im_clean, auxim1, auxim1], dim=1)
                sigma_rgb = 255*torch.tensor([1e-4, 1 / scale1, 1 / scale1], dtype=y_im_clean.dtype, device=y_im_clean.device) * math.sqrt(tau / mu)
                y_rgb_np = torch.moveaxis(torch.squeeze(y_rgb), 0, -1).cpu().numpy()
                sigma_rgb_np = list(sigma_rgb.cpu().numpy())
                y_rgb_est = bm3d_rgb_mod(y_rgb_np, sigma_rgb_np)
                auxim1 = torch.tensor(y_rgb_est[:, :, -1], dtype=y_rgb.dtype, device=y_rgb.device)[None, None] * scale1 + min_im1
                v2im_new.append(auxim1)
            v2im_new = torch.cat(v2im_new, dim=1)
            v2im = torch.clone(v2im_new)
            v2 = torch.reshape(v2im.transpose(2, 3), [v2.shape[0], v2.shape[1], v2.shape[2]])#.transpose(1, 2)

        print(i, torch.norm(nu1+d1-v1, p='fro').item(), torch.norm(nu2+d2-v2, p='fro').item())
        d1 = -nu1 + v1
        d2 = -nu2 + v2

    x_hat = e @ z
    fused = conv2im(x_hat, nl, nc, nb)

    fused_20 = fused[:, bands_20_index, :, :]
    fused_60 = fused[:, bands_60_index, :, :]

    return fused_20, fused_60
def synthetize_pan(coarse, pan):
    bs, c1, a1, b1 = coarse.shape
    _, c2, a2, b2 = pan.shape

    gb0 = torch.flatten(pan.transpose(2, 3), start_dim=2)
    gb1 = torch.flatten(coarse.transpose(-2, -1), start_dim=-2)
    C = torch.cat([torch.ones([bs, 1, a1 * b1], dtype=gb0.dtype, device=gb0.device), gb0], dim=1).transpose(1, 2)

    xrc1 = torch.linalg.lstsq(C, gb1.transpose(-2, -1)).solution

    ff1 = torch.cat([torch.ones([bs, 1, a2 * b2], dtype=gb0.dtype, device=gb0.device), gb0], dim=1).transpose(-2, -1) @ xrc1
    z_r = torch.unflatten(ff1.transpose(-2, -1), dim=2, sizes=(b2, a2)).transpose(2, 3)

    return z_r



if __name__ == '__main__':
    from scipy import io
    import matplotlib
    from recordclass import recordclass
    import numpy as np

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    bands_10_index = [1, 2, 3, 7]
    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]

    y_im = io.loadmat('/home/matteo/Desktop/MATLAB/MusaExperiments/InputData/RealData/S2_Crops.mat')['S2']
    bands_high = []
    for i in bands_10_index:
        bands_high.append(y_im[i, :][0][None, None, :, :])

    bands_high = np.concatenate(bands_high, axis=1).astype(np.float64)
    bands_high = torch.from_numpy(bands_high)

    bands_intermediate_lr = []
    for i in bands_20_index:
        bands_intermediate_lr.append(y_im[i, :][0][None, None, :, :])

    bands_intermediate_lr = np.concatenate(bands_intermediate_lr, axis=1).astype(np.float64)
    bands_intermediate_lr = torch.from_numpy(bands_intermediate_lr)

    bands_low_lr = []
    for i in bands_60_index:
        bands_low_lr.append(y_im[i, :][0][None, None, :, :])

    bands_low_lr = np.concatenate(bands_low_lr, axis=1).astype(np.float64)
    bands_low_lr = torch.from_numpy(bands_low_lr)
    exp_info = {'bands_low_lr': bands_low_lr, 'bands_intermediate': bands_intermediate_lr, 'bands_high': bands_high}

    exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

    fused20, fused60 = MuSA(exp_input)

    fused_20 = fused20.numpy()
    fused_60 = fused60.numpy()
    plt.figure()
    plt.imshow(fused_20[0, 4, :, :], cmap='gray')
    plt.figure()
    plt.imshow(fused_60[0, 0, :, :], cmap='gray')


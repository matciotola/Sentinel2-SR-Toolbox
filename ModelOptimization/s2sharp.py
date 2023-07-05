from ModelOptimization.tools import generate_stack, generate_mean_stack, normalize_data, denormalize_data, \
                                    create_subsampling, compute_weights, conv_cm, conv2im
import torch
from math import sqrt

import pymanopt
from pymanopt import Problem
from pymanopt.manifolds import Stiefel
from pymanopt.optimizers.trust_regions import TrustRegions

from Utils.spectral_tools import fspecial_gauss
from Utils.imresize_bicubic import imresize


def S2Sharp(ordered_dict):
    bands_high = torch.clone(ordered_dict.bands_high)
    bands_intermediate_lr = torch.clone(ordered_dict.bands_intermediate)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    if bands_low_lr.shape[1] == 3:
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    # hyperparameters

    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]

    cd_iter = 10
    r = 7
    lambda_opt = 0.005

    g_step_only = 0
    gcv = 0
    limsub = 2
    dx = 12
    dy = 12

    sigmas = 1

    q = torch.tensor([1, 1.5, 4, 8, 15, 15, 20], dtype=bands_high.dtype, device=bands_high.device)

    ratio = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]

    mtf = [0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.33, 0.26, 0.22, 0.23]

    sdf = torch.tensor(ratio) * torch.sqrt(-2 * torch.log(torch.tensor(mtf)) / torch.pi ** 2)
    sdf[torch.tensor(ratio) == 1] = 0

    # extract img infos

    bs, _, nl, nc = bands_high.shape
    n = nl * nc
    y_im = generate_stack(bands_high, bands_intermediate_lr, bands_low_lr)
    nb = len(y_im)

    bands_high_norm, m_high = normalize_data(bands_high)
    bands_intermediate_lr_norm, m_intermediate_lr = normalize_data(bands_intermediate_lr)
    bands_low_lr_norm, m_low_lr = normalize_data(bands_low_lr)
    y_im_2 = generate_stack(bands_high_norm, bands_intermediate_lr_norm, bands_low_lr_norm)
    global_mean = torch.cat(generate_mean_stack(m_high, m_intermediate_lr, m_low_lr), 1)

    # define convolution operators

    fbm = create_conv_kernel(sdf, ratio, nl, nc, len(mtf), dx, dy)

    y, m, f = inizialization(y_im_2, sdf, nb, nl, nc, dx, dy, ratio, limsub, r)

    mask = torch.flatten(m.transpose(2,3), start_dim=2)
    z = torch.zeros([bs, r, n], dtype=y.dtype, device=y.device)

    fdh, fdv, fdhc, fdvc = create_diff_kernels(nl, nc, r)

    w = compute_weights(y, ratio, sigmas, nl, nc, nb)

    if gcv == 1:
        g_step_only = 1
    if g_step_only != 0:
        cd_iter = 1

    for j_cd in range(cd_iter):
        z = z_step(y, fbm, f, lambda_opt, nl, nc, nb, z, mask, q, fdh, fdv, fdhc, fdvc, w)
        f = f_step(f, z, y, fbm, nl, nc, nb, mask)

    x_hat_im = conv2im(f @ z, nl, nc, nb)
    fused = denormalize_data(x_hat_im, global_mean)

    fused_20 = fused[:, bands_20_index, :, :]
    fused_60 = fused[:, bands_60_index, :, :]

    if ordered_dict.ratio == 2:
        fused = fused_20
    else:
        fused = fused_60

    return fused


def inizialization(img, sdf, nb, nl, nc, dx, dy, d, limsub, r):
    fbm2 = create_conv_kernel_subspace(sdf, nl, nc, nb, dx, dy)
    y_lim = []
    for i in range(nb):
        y_lim.append(imresize(img[i], scale=d[i]))

    y_lim = torch.cat(y_lim, dim=1)

    y_im = torch.real(torch.fft.ifft2(torch.fft.fft2(y_lim) * fbm2))
    y_tr = y_im[:, :, limsub:-limsub, limsub:-limsub]
    y_2n = torch.flatten(y_tr.transpose(2, 3), start_dim=2)
    f, _, _ = torch.linalg.svd(y_2n, full_matrices=False)
    f = f[:, :, :r]

    m, y = create_subsampling(img, d, nl, nc, nb)
    m = torch.cat(m, dim=0)[None, :, :, :]
    return y, m, f


def f_step(f, z, y, fbm, nl, nc, nb, mask):

    f0 = torch.clone(f)
    btx_hat = conv_cm(f0@z, fbm, nl, nc, nb)
    mbtx_hat = mask * btx_hat
    bs, nb, r = f.shape
    mbzt = []
    a = []
    zbmt_y = []
    for ii in range(nb):
        temp = mask[:, ii:ii+1, :].repeat(1, r, 1) * conv_cm(z, fbm[:, ii:ii+1, :, :].repeat(1, r, 1, 1), nl, nc, nb)
        mbzt.append(torch.unsqueeze(temp, -1))
        a.append(torch.unsqueeze(temp @ temp.transpose(1, 2), -1))
        zbmt_y.append(temp @ y[:, ii:ii+1, :].transpose(1, 2))

    mbzt = torch.cat(mbzt, dim=-1)
    a = torch.cat(a, dim=-1)
    zbmt_y = torch.cat(zbmt_y, dim=-1)

    manifold = Stiefel(nb, r)

    @pymanopt.function.pytorch(manifold)
    def cost(f):
        ju = 0
        for i in range(f.shape[1]):
            mat1 = mbzt[:, :, :, i].transpose(1, 2)
            mat2 = f[:, i:i + 1, :].transpose(1, 2)
            mat3 = y[:, i:i + 1, :].transpose(1, 2)
            ju = ju + 0.5 * torch.norm(mat1 @ mat2 - mat3, p='fro') ** 2
        return ju

    problem = Problem(manifold=manifold, cost=cost)
    solver = TrustRegions(verbosity=0, log_verbosity=1, min_gradient_norm=1e-2)
    result = solver.run(problem, initial_point=f0.numpy())
    f1 = torch.from_numpy(result.point).type(f0.dtype).to(f0.device)

    return f1


def z_step(y, fbm, f, tau, nl, nc, nb, z, mask, q, fdh, fdv, fdhc, fdvc, w):
    r = f.shape[2]
    ubtmf_y = f.transpose(1, 2) @ conv_cm(y, torch.conj(fbm), nl, nc, nb)
    z = cg(z, f, y, ubtmf_y, fbm, mask, nl, nc, nb, r, tau, q, fdh, fdv, fdhc, fdvc, w)

    return z


def grad_cost_g(z, f, y, ubtmf_y, fbm, mask, nl, nc, nb, r, tau, q, fdh, fdv, fdhc, fdvc, w):

    x = f @ z
    bx = conv_cm(x, fbm, nl, nc, nb)
    hthbx = mask * bx
    zh = conv_cm(z, fdhc, nl, nc, r)
    zv = conv_cm(z, fdvc, nl, nc, r)

    zhw = zh * w
    zvw = zv * w

    grad_pen = conv_cm(zhw, fdh, nl, nc, r) + conv_cm(zvw, fdv, nl, nc, r)
    atag = f.transpose(-2, -1) @ conv_cm(hthbx, torch.conj(fbm), nl, nc, nb) + 2 * tau * (q[:, None].repeat(1, nl*nc)) * grad_pen
    grad_j = atag - ubtmf_y
    j = 0.5 * torch.sum(z*atag) - torch.sum(z*ubtmf_y)

    return j, grad_j, atag


def cg(z, f, y, ubtmf_y, fbm, mask, nl, nc, nb, r, tau, q, fdh, fdv, fdhc, fdvc, w):
    max_iter = 1000
    tol_grad_norm = 0.1
    cost, grad, _ = grad_cost_g(z, f, y, ubtmf_y, fbm, mask, nl, nc, nb, r, tau, q, fdh, fdv, fdhc, fdvc, w)
    grad_norm = torch.norm(grad)

    iter = 0
    res = -grad

    while(grad_norm > tol_grad_norm and iter < max_iter):
        iter += 1
        if iter == 1:
            desc_dir = res
        else:
            beta = torch.sum(res ** 2) / torch.sum(res_old ** 2)
            desc_dir = res + beta * desc_dir

        _, _, atap = grad_cost_g(desc_dir, f, y, ubtmf_y, fbm, mask, nl, nc, nb, r, tau, q, fdh, fdv, fdhc, fdvc, w)
        alpha = torch.sum(res ** 2) / torch.sum(torch.flatten(desc_dir.transpose(-2, -1)) * torch.flatten(atap.transpose(-2, -1)))
        z1 = z + alpha * desc_dir
        res_old = res
        res = res - alpha * atap
        grad_norm = torch.norm(res)
        z = torch.clone(z1)

    return z


def create_diff_kernels(nl, nc, r):
    dh = torch.zeros(1, nl, nc)
    dh[:, 0, 0] = 1
    dv = torch.clone(dh)
    dh[:, 0, -1] = -1
    dv[:, -1, 0] = -1

    fdh = torch.fft.fft2(dh).repeat(r, 1, 1)
    fdv = torch.fft.fft2(dv).repeat(r, 1, 1)
    fdhc = torch.conj(fdh)
    fdvc = torch.conj(fdv)

    return fdh, fdv, fdhc, fdvc


def create_conv_kernel(sdf, ratio, nl, nc, nb, dx, dy):
    middle_l = nl // 2
    middle_c = nc // 2

    ddx = dx // 2
    ddy = dy // 2

    FBM = []
    for i in range(nb):
        B = torch.zeros(nl, nc)
        if ratio[i] > 1:
            h = torch.tensor(fspecial_gauss((dy, dx), sdf[i].numpy()))

            B[middle_l - ddy:middle_l + ddy, middle_c - ddx :middle_c + ddx] = h
            B = torch.fft.fftshift(B)
            B = B / torch.sum(B)
            FBM.append(torch.fft.fft2(B)[None, :, :])
        else:
            B[0, 0] = 1
            FBM.append(torch.fft.fft2(B)[None, :, :])

    FBM = torch.vstack(FBM)

    return FBM[None, :, :, :]


def create_conv_kernel_subspace(sdf, nl, nc, nb, dx, dy):
    middle_l = round((nl + 1) / 2)
    middle_c = round((nc + 1) / 2)

    ddx = dx // 2
    ddy = dy // 2

    dx = dx + 1
    dy = dy + 1

    FBM = []

    s2 = max(sdf)

    for i in range(nb):
        B = torch.zeros(nl, nc)
        if sdf[i] < s2:
            h = torch.tensor(fspecial_gauss((dx, dy), sqrt(s2 ** 2 - sdf[i] ** 2)))
            B[middle_l - ddy:middle_l + ddy + 1, middle_c - ddx:middle_c + ddx + 1] = h
            B = torch.fft.fftshift(B)
            B = B / torch.sum(B)
            FBM.append(torch.fft.fft2(B)[None, :, :])

        else:
            B[0, 0] = 1
            FBM.append(B[None, :, :])
    FBM = torch.vstack(FBM)

    return FBM[None, :, :, :]


if __name__ == '__main__':
    from recordclass import recordclass
    import numpy as np
    from scipy import io
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    bands_10_index = [1, 2, 3, 7]
    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]

    y_im = io.loadmat('/home/matteo/Desktop/MATLAB/SSSS/Yim_cell.mat')['Yim_cell']

    bands_high = []
    for i in bands_10_index:
        bands_high.append(y_im[:, i][0][None, None, :, :])

    bands_high = np.concatenate(bands_high, axis=1).astype(np.float64)
    bands_high = torch.from_numpy(bands_high)

    bands_intermediate_lr = []
    for i in bands_20_index:
        bands_intermediate_lr.append(y_im[:, i][0][None, None, :, :])

    bands_intermediate_lr = np.concatenate(bands_intermediate_lr, axis=1).astype(np.float64)
    bands_intermediate_lr = torch.from_numpy(bands_intermediate_lr)

    bands_low_lr = []
    for i in bands_60_index:
        bands_low_lr.append(y_im[:, i][0][None, None, :, :])

    bands_low_lr = np.concatenate(bands_low_lr, axis=1).astype(np.float64)
    bands_low_lr = torch.from_numpy(bands_low_lr)

    exp_info = {'bands_low_lr': bands_low_lr, 'bands_intermediate': bands_intermediate_lr, 'bands_high': bands_high}

    exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())
    fused20, fused60 = S2Sharp(exp_input)

    fused_20 = fused20.numpy()
    plt.figure()
    plt.imshow(fused_20[0, 4, :, :], cmap='gray')

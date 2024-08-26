import torch
from .tools import generate_stack
from Utils.spectral_tools import fspecial_gauss
import gc
from tqdm import tqdm


def SSSS(ordered_dict):
    bands_10 = torch.clone(ordered_dict.bands_10).double()
    bands_20 = torch.clone(ordered_dict.bands_20).double()
    bands_60 = torch.clone(ordered_dict.bands_60).double()

    if bands_60.shape[1] == 3:
        bands_60 = bands_60[:, :-1, :, :]

    # HyperParameters

    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]
    # hyperparameters

    rv = torch.tensor([6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2])
    lambda_opt = 0.1
    mu = 0.1
    nr = 72

    # preprocessing

    if bands_60.shape[1] == 3:
        bands_60 = bands_60[:, :-1, :, :]

    # upsampling matrix

    y_im_list = generate_stack(bands_10, bands_20, bands_60)

    y_im = torch.zeros([1, len(y_im_list), bands_10.shape[-2], bands_10.shape[-1]], dtype=bands_10.dtype,
                       device=bands_10.device)
    for i in range(len(y_im_list)):
        y_im[:, i, ::rv[i], ::rv[i]] = y_im_list[i]

    bs, nb, c, d = y_im.shape

    # blurring matrix

    dx = 13
    dy = 13
    limsub = 6
    mtf = torch.tensor([0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.23, 0.33, 0.26, 0.22])
    sdf = rv * torch.sqrt(-2 * torch.log(mtf) / torch.pi ** 2)
    sdf[rv == 1] = 0

    # algorithm

    a = (c - 2 * limsub) // (nr - 2 * limsub)

    fused = torch.zeros([bs, nb, c, d], dtype=bands_10.dtype, device=bands_10.device)
    fused[:, :, :nr, :] = ssss_algorithm(y_im[:, :, :nr, :], rv, dx, dy, sdf, lambda_opt, mu)
    for i in tqdm(range(1, a + 1)):
        row_start = ((nr - 2 * limsub) * i)
        row_end = row_start + nr
        fused[:, :, row_start+limsub:row_end, :] = ssss_algorithm(y_im[:, :, row_start:row_end, :], rv, dx, dy,
                                                                  sdf, lambda_opt, mu)[:, :, limsub:, :]
        gc.collect()

    fused_20 = fused[:, bands_20_index, :, :]
    fused_60 = fused[:, bands_60_index, :, :]

    if ordered_dict.ratio == 2:
        fused = fused_20
    else:
        fused = fused_60

    return fused


def ssss_algorithm(y_im, rv, dx, dy, sdf, lambda_opt, mu):
    # hyperparameters
    net_degree = 1
    patch_size = 6
    band_for_net_learn = [1, 2, 3, 7]
    p = 5
    epochs = 30
    num_admm_iter = 100

    bs, nb, nr, nc = y_im.shape
    n = nr * nc
    boundary_width = dx // 2

    y = torch.flatten(y_im.transpose(2, 3), start_dim=2, end_dim=3)

    # blurring kernel and blurring matrix

    middlel = nr // 2
    middlec = nc // 2

    kernel = []
    for i in range(nb):
        if sdf[i] > 0:
            k = torch.tensor(fspecial_gauss((dx, dy), sdf[i].item()), dtype=y.dtype, device=y.device)
            kk = torch.zeros([nr, nc], dtype=y.dtype, device=y.device)
            kk[middlel - dx // 2:middlel + dx // 2 + 1, middlec - dy // 2:middlec + dy // 2 + 1] = k
            kk = torch.fft.fftshift(kk)
            kk = kk / kk.sum()
            kernel.append(kk[None, None, :, :])
        else:
            kk = torch.zeros([nr, nc], dtype=y.dtype, device=y.device)
            kk[0, 0] = 1
            kernel.append(kk[None, None, :, :])

    kernel = torch.cat(kernel, dim=1)
    fbm = torch.fft.fft2(kernel, dim=(2, 3))

    # network learning

    edge, alpha = net_learn(torch.mean(y_im[:, band_for_net_learn, :, :], dim=1, keepdim=True), patch_size,
                            boundary_width, net_degree)

    noe = edge.shape[1]

    # admm

    x_im = []
    for i in range(nb):
        x_im.append(pan4(y_im[:, i:i + 1, :, :], y_im[:, [1, 2, 3, 7], :, :], rv[i], fbm[:, i:i + 1, :, :]))

    x_im = torch.cat(x_im, dim=1)
    uz = torch.flatten(x_im.transpose(2, 3), start_dim=2, end_dim=3)

    u, _, _ = torch.linalg.svd(uz @ uz.transpose(1, 2))

    if p < nb:
        u = u[:, :, :p]
        z = u.transpose(1, 2) @ uz
    else:
        u = torch.eye(nb, dtype=y.dtype, device=y.device)
        z = uz

    utu_plus_i_inv = torch.inverse(u.transpose(1, 2) @ u + noe * torch.eye(p, dtype=y.dtype, device=y.device))
    fbmc = torch.conj(fbm)
    bty = torch.flatten(
        torch.real(torch.fft.ifft2(torch.fft.fft2(torch.reshape(y, [bs, y.shape[1], nc, nr]).transpose(2, 3)) * fbmc)).transpose(2,3),
        start_dim=2)

    iff = []
    for i in range(nb):
        s = fbm[:, i:i + 1, :, :]
        r = rv[i]
        iff.append(1 / (r ** 2 + (dh(torch.abs(s ** 2), r)) / mu))

    idx_pi_pj = []
    ptp_plus_i_inv = []
    for ij in range(noe):
        a, b = patch_idx(nr, nc, edge[:, ij, :], patch_size, mu, lambda_opt, alpha[:, ij, :].item())
        idx_pi_pj.append(a)
        ptp_plus_i_inv.append(b)

    d = torch.zeros([nb * n], dtype=y.dtype, device=y.device)
    v = torch.clone(d)
    d_ij = torch.zeros([n, p, noe], dtype=y.dtype, device=y.device)
    v_ij = torch.clone(d_ij)

    for i in range(num_admm_iter):
        nu = (u @ z) - torch.reshape(d, [bs, nb, n])
        aux = bty + (mu * nu)
        for b in range(nb):
            inp = torch.squeeze(aux[:, b, :])
            r = rv[b]
            s = fbm[:, b:b + 1, :, :]
            vv = (torch.fft.ifft2((dh((torch.fft.fft2(torch.reshape(inp, [bs, 1, nc, nr]).transpose(-2, -1)) * s),
                                      r) * iff[b]).repeat(1, 1, r, r) * torch.conj(s)))
            v[b * n:(b + 1) * n] = inp / mu - torch.real(vv.transpose(-2, -1).flatten()) / mu / mu

        for ij in range(noe):
            delta = torch.squeeze(z.transpose(1, 2)) - d_ij[:, :, ij]
            v_ij[:, :, ij] = delta
            v_ij[idx_pi_pj[ij], :, ij] = ptp_plus_i_inv[ij].type(y.dtype) @ (
                    mu / lambda_opt / alpha[:, ij, :] * delta[idx_pi_pj[ij], :])

        # z update
        z = (((torch.reshape((v + d), [nb, n]).transpose(-2, -1) @ u) + torch.sum(v_ij + d_ij, dim=2))
             @ utu_plus_i_inv).transpose(-2, -1)
        # z = (((torch.reshape((v + d), [n, nb]) @ u) + torch.sum(v_ij + d_ij, dim=2)) @ utu_plus_i_inv).transpose(-2,-1)

        # dual update
        d = d + v - (u @ z).flatten()

        d_ij = d_ij + v_ij - torch.squeeze(z.transpose(-2, -1))[:, :, None].repeat(1, 1, noe)

        if i == epochs:
            term2 = z @ bty.transpose(1, 2)
            for b in range(nb):
                r = rv[b]
                input_1 = torch.reshape(z, [bs, z.shape[1], nc, nr]).transpose(2, 3)
                input_2 = fbm[:, b:b + 1, :, :].repeat(1, p, 1, 1)
                bzt = torch.flatten(torch.real(torch.fft.ifft2(torch.fft.fft2(input_1) * input_2)).transpose(2, 3), start_dim=2)
                bzt3d = torch.reshape(bzt, [bs, p, nc, nr]).transpose(2, 3)
                mbzt3d = torch.zeros(bzt3d.shape, dtype=bzt3d.dtype, device=bzt3d.device)
                mbzt3d[:, :, ::r, ::r] = bzt3d[:, :, ::r, ::r]
                mbzt = torch.reshape(mbzt3d.transpose(2,3), [p, n]).transpose(-2, -1)

                term1 = torch.inverse(mbzt.transpose(-2, -1) @ mbzt)[None, :, :]  # .to_sparse()
                # u[:, b, :] = (term1 @ term2[:, :, b]).transpose(-2, -1)
                u[:, b, :] = (term1 @ term2[:, :, b, None]).transpose(-1, -2)
            utu_plus_i_inv = torch.inverse(u.transpose(1, 2) @ u + noe * torch.eye(p, dtype=y.dtype, device=y.device))

    fused = v.reshape([bs, nb, n]).reshape([bs, nb, nc, nr]).transpose(2, 3)

    return fused


def net_learn(img, ps, bw, degree):
    bs, nb, nr, nc = img.shape
    temp = img.repeat(1, 1, 2, 2)
    patches = torch.nn.functional.unfold(temp[:, :, :nr + ps - 1, :nc + ps - 1].transpose(2, 3), kernel_size=(ps, ps))

    all_idx = torch.ones([bs, nb, nr, nc], dtype=torch.long, device=img.device)
    all_idx[:, :, bw:nr - bw - ps + 1, bw:nc - bw - ps + 1] = 0
    invalid_idx = torch.where(all_idx.transpose(2, 3).flatten() == 1)[0]

    dist = []
    edge = []
    for i in range(bw, nr - bw - ps + 1, ps):
        for j in range(bw, nc - bw - ps + 1, ps):
            idx = (i + 1) + (j) * nr - 1
            diff = torch.sum(((patches[:, :, idx:idx + 1] @ torch.ones([bs, nb, nr * nc], dtype=patches.dtype,
                                                                       device=patches.device)) - patches) ** 2, dim=1)
            diff[:, idx] = torch.inf
            diff[:, invalid_idx] = torch.inf
            dist_value, idx_similar = torch.min(diff, 1)

            dist.append(dist_value[None, None, :])
            edge.append(torch.tensor([idx, idx_similar])[None, None, :])

    dist = torch.cat(dist, dim=1)
    dist = torch.sqrt(dist)
    alpha = dist ** (-1)

    edge = torch.cat(edge, dim=1)

    return edge, alpha


def dh(x, ratio):
    bs, nb, nr, nc = x.shape
    x1 = torch.nn.functional.unfold(x.transpose(2, 3), kernel_size=(nc // ratio, nr // ratio),
                                    stride=(nc // ratio, nr // ratio))
    # x1 = torch.flatten(x.transpose(2, 3), start_dim=2)
    x1 = torch.sum(x1, dim=2, keepdim=True)
    x1 = torch.reshape(x1, [bs, nb, nc // ratio, nr // ratio]).transpose(2, 3)
    return x1


def lsqnonneg(C, d):
    # Set default options
    tol = 10 * torch.finfo(C.dtype).eps * torch.norm(C, p=1) * C.shape[0]

    # Initialize variables
    n = C.shape[1]
    n_zeros = torch.zeros(n, dtype=C.dtype, device=C.device)
    w_zeros = torch.zeros(n, dtype=C.dtype, device=C.device)

    # Initialize sets
    P = torch.zeros(n, dtype=torch.bool, device=C.device)
    Z = torch.ones(n, dtype=torch.bool, device=C.device)
    x = n_zeros

    resid = d - C @ x
    w = C.t() @ resid

    # Set up iteration criterion
    outer_iter = 0
    iter = 0
    itmax = 3 * n

    # Outer loop to put variables into the set to hold positive coefficients
    while Z.any() and (w[Z] > tol).any():
        outer_iter += 1

        # Reset intermediate solution z
        z = n_zeros.clone()

        # Create wz, a Lagrange multiplier vector of variables in the zero set
        wz = w_zeros.clone()
        wz[P] = float('-inf')
        wz[Z] = w[Z]

        # Find the variable with the largest Lagrange multiplier
        _, t = wz.max(dim=0)

        # Move variable t from the zero set to the positive set
        P[t] = True
        Z[t] = False

        # Compute intermediate solution using only variables in the positive set
        z[P] = torch.linalg.lstsq(C[:, P], d)[0]

        # Inner loop to remove elements from the positive set which no longer belong
        while (z[P] <= 0).any():
            iter += 1

            if iter > itmax:
                return x

            # Find indices where intermediate solution z is approximately negative
            Q = (z <= 0) & P

            # Choose new x subject to keeping new x nonnegative
            alpha = torch.min(x[Q] / (x[Q] - z[Q]))
            x = x + alpha * (z - x)

            # Reset Z and P given intermediate values of x
            Z = ((torch.abs(x) < tol) & P) | Z
            P = ~Z

            # Reset z
            z = n_zeros.clone()

            # Re-solve for z
            z[P] = torch.linalg.lstsq(C[:, P], d)[0]

        x = z
        resid = d - C @ x
        w = C.t() @ resid

    return x


def pan4(y, x4, r, fbm):
    if r == 0:
        syn = y
    else:
        bs, nb, nr, nc = y.shape
        n = nr * nc
        bx4 = torch.real(torch.fft.ifft2(
            torch.fft.fft2(torch.reshape(torch.flatten(x4, start_dim=2), [bs, x4.shape[1], nr, nc])) * fbm.repeat(1, 4,
                                                                                                                  1,
                                                                                                                  1)))
        d = torch.flatten(y[:, :, ::r, ::r].transpose(2, 3), start_dim=2)
        c = torch.flatten(bx4[:, :, ::r, ::r].transpose(2, 3), start_dim=2)

        coef = lsqnonneg(torch.squeeze(c.transpose(1, 2)), torch.squeeze(d.transpose(1, 2)))

        syn = torch.reshape(torch.matmul(torch.flatten(x4, start_dim=2).transpose(1, 2), coef), [bs, nb, nr, nc])

        return syn


def patch_idx(nr, nc, edge, ps, mu, lambda_opt, alpha):
    i = torch.squeeze(edge[:, 0])
    j = torch.squeeze(edge[:, 1])

    n = nr * nc
    yi = torch.remainder(i, nr)
    if yi == 0:
        yi = nr
    xi = (i - yi) // nr

    yj = torch.remainder(j, nr)
    if yj == 0:
        yj = nr
    xj = (j - yj) // nr

    tempi = torch.zeros([2 * nr, 2 * nc])
    tempi[yi:yi + ps, xi:xi + ps] = torch.ones([ps, ps])

    mapi = dh(tempi[None, None, :, :], 2)
    idx_pi = torch.nonzero(mapi.transpose(2, 3).flatten() == 1, as_tuple=True)[0]
    pi = torch.zeros([ps * ps, n])
    pi[:, idx_pi] = torch.eye(ps * ps)

    tempj = torch.zeros([2 * nr, 2 * nc])
    tempj[yj:yj + ps, xj:xj + ps] = torch.ones([ps, ps])
    mapj = dh(tempj[None, None, :, :], 2)

    idx_pj = torch.nonzero(mapj.transpose(2, 3).flatten() == 1, as_tuple=True)[0]
    pj = torch.zeros([ps * ps, n])
    pj[:, idx_pj] = torch.eye(ps * ps)

    idx_pi_pj = \
        torch.nonzero((mapi.transpose(2, 3).flatten() == 1) + (mapj.transpose(2, 3).flatten() == 1), as_tuple=True)[0]

    card_i = len(idx_pi_pj)
    pi_minus_pj = pi - pj
    pi_minus_pj_active = pi_minus_pj[:, idx_pi_pj]

    ptp_plus_i_inv = torch.inverse(
        torch.matmul(pi_minus_pj_active.transpose(0, 1), pi_minus_pj_active) + torch.eye(card_i) * (
                mu / lambda_opt / alpha))

    return idx_pi_pj, ptp_plus_i_inv

if __name__ == '__main__':
    from scipy import io
    import numpy as np
    import matplotlib
    matplotlib.use('TkAgg')
    from matplotlib import pyplot as plt
    temp = io.loadmat('/home/matteo/Desktop/MATLAB/SSSS/Yim_cell.mat')['Yim_cell']

    bands_10 = []
    bands_20 = []
    bands_60 = []
    d = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]
    for i in range(len(d)):
        if d[i] == 1:
            bands_10.append(torch.from_numpy(temp[0, i].astype(np.float64)[None, :, :]))
        elif d[i] == 2:
            bands_20.append(torch.from_numpy(temp[0, i].astype(np.float64)[None, :, :]))
        else:
            bands_60.append(torch.from_numpy(temp[0, i].astype(np.float64)[None, :, :]))

    # bands_10 = torch.vstack(bands_10)[None, :, :, :]
    # bands_20 = torch.vstack(bands_20)[None, :, :, :]
    # bands_60 = torch.vstack(bands_60)[None, :, :, :]
    #
    # fused = SSSS(bands_10, bands_20, bands_60, 2)
    #
    # plt.figure()
    # plt.imshow(fused[0, 0, :, :].numpy(), cmap='gray')
    # plt.show()

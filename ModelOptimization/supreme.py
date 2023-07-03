from ModelOptimization.tools import *
from ModelOptimization.tools import conv_cm, compute_weights, regularization, conv2im
from Utils.imresize_bicubic import imresize


def SupReMe(ordered_dict):
    bands_high = torch.clone(ordered_dict.bands_high)
    bands_intermediate_lr = torch.clone(ordered_dict.bands_intermediate)
    bands_low_lr = torch.clone(ordered_dict.bands_low_lr)

    if bands_low_lr.shape[1] == 3:
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    # HyperParameters

    bands_20_index = [4, 5, 6, 8, 10, 11]
    bands_60_index = [0, 9]

    ratio = [6, 1, 1, 1, 2, 2, 2, 1, 2, 6, 2, 2]

    p = 7
    opt_lambda = 0.005
    reg_type = 'l2'
    limsub = 2
    dx = 12
    dy = 12

    mtf = [0.32, 0.26, 0.28, 0.24, 0.38, 0.34, 0.34, 0.26, 0.33, 0.26, 0.22, 0.23]

    sdf = torch.tensor(ratio) * torch.sqrt(-2 * torch.log(torch.tensor(mtf)) / torch.pi ** 2)
    sdf[torch.tensor(ratio) == 1] = 0

    # Normalize data

    bands_high_normalized, m_high = normalize_data(bands_high)
    bands_intermediate_lr_normalized, m_intermediate_lr = normalize_data(bands_intermediate_lr)
    bands_low_lr_normalized, m_low_lr = normalize_data(bands_low_lr)

    y_im_2 = generate_stack(bands_high_normalized, bands_intermediate_lr_normalized, bands_low_lr_normalized)
    global_mean = generate_mean_stack(m_high, m_intermediate_lr, m_low_lr)

    # Compute dimensions of the input

    _, _, nl, nc = bands_high_normalized.shape
    n = nl * nc

    # Define blurring operators

    fbm = create_conv_kernel(sdf, ratio, nl, nc, len(mtf), dx, dy)
    fbm2 = create_conv_kernel_subspace(sdf, nl, nc, len(mtf), dx, dy)


    # Generate LR MS image for subspace
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
    u, _, _ = torch.linalg.svd(matrix)

    u = u[:, :, :p]

    # Subsampling

    m, y = create_subsampling(y_im_2, ratio, nl, nc, y2_im.shape[1])

    # Solver

    x_hat_im = solver(y, fbm, u, ratio, opt_lambda, nl, nc, y.shape[1], reg_type)

    x_hat_im = denormalize_data(x_hat_im, torch.cat(global_mean, dim=1))

    fused_20 = x_hat_im[:, bands_20_index, :, :]
    fused_60 = x_hat_im[:, bands_60_index, :, :]

    return fused_20, fused_60


def solver(y, fbm, U, d, tau, nl, nc, nb, reg_type):
    # definitions
    bs = y.shape[0]

    niters = 100
    mu = 0.2

    p = U.shape[-1]
    n = nl * nc
    fbmc = torch.conj(fbm)
    bty = conv_cm(y, fbmc, nl, nc, nb)

    # operators for differences

    dh = torch.zeros(nl, nc, dtype=y.dtype, device=y.device)
    dh[0, 0] = 1
    dh[0, -1] = -1

    dv = torch.zeros(nl, nc, dtype=y.dtype, device=y.device)
    dv[0, 0] = 1
    dv[-1, 0] = -1

    fdh = torch.fft.fft2(dh)[None, None, :, :].repeat(bs, p, 1, 1)
    fdv = torch.fft.fft2(dv)[None, None, :, :].repeat(bs, p, 1, 1)
    fdhc = torch.conj(fdh)
    fdvc = torch.conj(fdv)

    # compute weights

    sigmas = 1
    w = compute_weights(y, d, sigmas, nl, nc, nb)

    iff = torch.zeros(fbm.shape, dtype=y.dtype, device=y.device)

    # build the inverse filter in frequency domain with subsampling

    for i in range(nb):
        f_im = torch.abs(fbm[:, i, None, :, :] ** 2)
        kc, kh, kw = 1, nl // d[i], nc // d[i]  # kernel size
        dc, dh, dw = 1, nl // d[i], nc // d[i]  # stride
        patches = f_im.unfold(1, kc, dc).unfold(2, kh, dh).unfold(3, kw, dw)
        unfold_shape = list(patches.shape)
        f_patches = patches.contiguous().view(-1, kc, kh, kw).flatten(start_dim=2)
        p2 = f_patches.shape[0]
        f_patches = 1 / (torch.sum(f_patches, dim=0, keepdim=True) / mu + d[i] ** 2)

        aux = f_patches.repeat(p2, 1, 1).view(unfold_shape)
        aux_c = unfold_shape[1] * unfold_shape[4]
        aux_h = unfold_shape[2] * unfold_shape[5]
        aux_w = unfold_shape[3] * unfold_shape[6]
        aux = aux.permute(0, 1, 4, 2, 5, 3, 6).contiguous()
        aux = aux.view(1, aux_c, aux_h, aux_w)

        aux2 = f_im * aux
        iff[:, i:i + 1, :, :] = aux2

    ifz = 1 / (torch.abs(fdh) ** 2 + torch.abs(fdv) ** 2 + 1)

    # initialization
    z = torch.zeros([bs, p, n], dtype=y.dtype, device=y.device)
    v1 = torch.zeros([bs, nb, n], dtype=y.dtype, device=y.device)
    v2 = torch.clone(z)
    v3 = torch.clone(z)
    d1 = torch.clone(v1)
    d2 = torch.clone(z)
    d3 = torch.clone(z)

    # SupReMe

    for i in range(niters):
        conv_inp = U.transpose(1, 2) @ (v1 + d1) + conv_cm(v2 + d2, fdhc, nl, nc, nb) + conv_cm(v3 + d3, fdvc, nl, nc, nb)
        z = conv_cm(conv_inp, ifz, nl, nc, nb)

        nu1 = U @ z - d1
        aux = bty + mu * nu1

        v1 = aux / mu - conv_cm(aux, iff, nl, nc, nb) / mu ** 2
        nu2 = conv_cm(z, fdh, nl, nc, nb) - d2
        nu3 = conv_cm(z, fdv, nl, nc, nb) - d3

        v2, v3 = regularization(nu2, nu3, tau, mu, w)

        error = torch.norm(nu2 + d2 - v2, p='fro') + torch.norm(nu3 + d3 - v3, p='fro') + torch.norm(nu1 + d1 - v1,
                                                                                                     p='fro')

        if error < 1e-3:
            break

        d1 = -nu1 + v1
        d2 = -nu2 + v2
        d3 = -nu3 + v3

    x_hat = U @ z
    x_hat_im = conv2im(x_hat, nl, nc, nb)

    return x_hat_im

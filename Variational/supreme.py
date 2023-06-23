from Variational.tools import *
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

    fbm = kernel_reshape(fbm)
    fbm2 = kernel_reshape(fbm2)

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

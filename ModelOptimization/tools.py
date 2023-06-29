import torch
from Utils.spectral_tools import fspecial_gauss
from math import sqrt
import torchvision.transforms.functional as TF


def kernel_reshape(kernel):
    kernel = torch.moveaxis(kernel, -1, 0)
    return kernel[None, :, :]


def normalize_data(img):
    m = torch.mean(img ** 2, dim=(2, 3), keepdim=True)
    normalized = torch.sqrt(img ** 2 / m)

    return normalized, m


def denormalize_data(img, m):
    return torch.sqrt(img ** 2 * m)


def create_conv_kernel(sdf, ratio, nl, nc, nb, dx, dy):
    middle_l = nl // 2
    middle_c = nc // 2

    ddx = dx // 2
    ddy = dy // 2

    rr = [x // 2 for x in ratio]

    FBM = []
    for i in range(nb):
        B = torch.zeros(nl, nc)
        if ratio[i] > 1:
            h = torch.tensor(fspecial_gauss((dy, dx), sdf[i].numpy()))

            B[middle_l - ddy - rr[i] + 1:middle_l + ddy - rr[i] + 1,
            middle_c - ddx - rr[i] + 1:middle_c + ddx - rr[i] + 1] = h
            B = torch.fft.fftshift(B)
            B = B / torch.sum(B)
            FBM.append(torch.fft.fft2(B)[None, :, :])
        else:
            B[0, 0] = 1
            FBM.append(B[None, :, :])

    FBM = torch.vstack(FBM)
    FBM = torch.moveaxis(FBM, 0, 2)
    return FBM


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
            B[middle_l - ddy - 1:middle_l + ddy, middle_c - ddx - 1:middle_c + ddx] = h
            B = torch.fft.fftshift(B)
            B = B / torch.sum(B)
            FBM.append(torch.fft.fft2(B)[None, :, :])

        else:
            B[0, 0] = 1
            FBM.append(B[None, :, :])
    FBM = torch.vstack(FBM)
    FBM = torch.moveaxis(FBM, 0, 2)

    return FBM


def generate_stack(bands_high, bands_intermediate, bands_low):
    img = []
    img.append(bands_low[:, 0, None, :, :])
    img.append(bands_high[:, 0, None, :, :])
    img.append(bands_high[:, 1, None, :, :])
    img.append(bands_high[:, 2, None, :, :])
    img.append(bands_intermediate[:, 0, None, :, :])
    img.append(bands_intermediate[:, 1, None, :, :])
    img.append(bands_intermediate[:, 2, None, :, :])
    img.append(bands_high[:, 3, None, :, :])
    img.append(bands_intermediate[:, 3, None, :, :])
    img.append(bands_low[:, 1, None, :, :])
    img.append(bands_intermediate[:, 4, None, :, :])
    img.append(bands_intermediate[:, 5, None, :, :])

    return img


def generate_mean_stack(m_high, m_intermediate, m_low):
    m = []
    m.append(m_low[:, 0, None, :, :])
    m.append(m_high[:, 0, None, :, :])
    m.append(m_high[:, 1, None, :, :])
    m.append(m_high[:, 2, None, :, :])
    m.append(m_intermediate[:, 0, None, :, :])
    m.append(m_intermediate[:, 1, None, :, :])
    m.append(m_intermediate[:, 2, None, :, :])
    m.append(m_high[:, 3, None, :, :])
    m.append(m_intermediate[:, 3, None, :, :])
    m.append(m_low[:, 1, None, :, :])
    m.append(m_intermediate[:, 4, None, :, :])
    m.append(m_intermediate[:, 5, None, :, :])

    return m


def conv2mat(img):
    return torch.flatten(img.transpose(2, 3), start_dim=2, end_dim=3)


def conv2im(img, nl, nc, nb):
    return torch.unflatten(img, dim=2, sizes=(nc, nl)).transpose(2, 3)


def conv_cm(img, kernel, nl, nc, nb):
    X = conv2mat(
        torch.real(
            torch.fft.ifft2(
                torch.fft.fft2(conv2im(img, nl, nc, nb)) * kernel
            )
        )
    )
    return X


def create_subsampling(img, d, nl, nc, nb):
    bs = img[0].shape[0]
    M = []
    Y = torch.zeros(bs, nb, nl * nc, dtype=img[0].dtype, device=img[0].device)
    for i in range(nb):
        im = torch.ones([nl // d[i], nc // d[i]], dtype=img[0].dtype, device=img[0].device)
        maux = torch.zeros([d[i], d[i]], dtype=img[0].dtype, device=img[0].device)
        maux[0, 0] = 1

        mm = torch.kron(im, maux)
        M.append(mm[None, :, :])
        mask = torch.nonzero(mm.flatten())[:, 0]
        temp = conv2mat(img[i])
        Y[:, i, mask] = temp
    torch.cat(M, dim=0)
    return M, Y


def imgradient_intermediate(img):
    gy = img[:, :, 1:, :] - img[:, :, :-1, :]
    gx = img[:, :, :, 1:] - img[:, :, :, :-1]

    gy = TF.pad(gy, [0, 0, 0, 1], padding_mode='constant')
    gx = TF.pad(gx, [0, 0, 1, 0], padding_mode='constant')
    g_mag = torch.sqrt(gy ** 2 + gx ** 2)

    return g_mag


def compute_weights(y, d, sigmas, nl, nc, nb):
    hr_bands = list(torch.nonzero(torch.tensor(d) == 1)[:, 0].numpy())
    grad = torch.zeros([y.shape[0], hr_bands[-1] + 1, nl, nc], dtype=y.dtype, device=y.device)
    for i in hr_bands:
        grad[:, i, :, :] = imgradient_intermediate(conv2im(y[:, i, None, :], nl, nc, nb)) ** 2
    # grad = torch.cat(grad, dim=1)
    grad = torch.sqrt(torch.max(grad, dim=1, keepdim=True)[0])
    grad = grad / torch.quantile(grad, 0.95)

    wim = torch.exp(-grad ** 2 / (2 * sigmas ** 2))
    wim = torch.clip(wim, 0.5, wim.max())

    w = conv2mat(wim)

    return w


def regularization(x1, x2, tau, mu, w):
    q = torch.tensor([1, 1.5, 4, 8, 15, 15, 20], dtype=x1.dtype, device=x1.device)[None, :, None]
    wr = q @ w

    y1 = (mu * x1) / (mu + tau * wr)
    y2 = (mu * x2) / (mu + tau * wr)

    return y1, y2


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
        conv_inp = U.transpose(1, 2) @ (v1 + d1) + conv_cm(v2 + d2, fdhc, nl, nc, nb) + conv_cm(v3 + d3, fdvc, nl, nc,
                                                                                                nb)
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

import torch
from Utils.spectral_tools import fspecial_gauss
from math import sqrt
import torchvision.transforms.functional as TF


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

            B[middle_l - ddy - rr[i] + 1:middle_l + ddy - rr[i] + 1, middle_c - ddx - rr[i] + 1:middle_c + ddx - rr[i] + 1] = h
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
            B[middle_l - ddy :middle_l + ddy + 1, middle_c - ddx:middle_c + ddx + 1] = h
            B = torch.fft.fftshift(B)
            B = B / torch.sum(B)
            FBM.append(torch.fft.fft2(B)[None, :, :])

        else:
            B[0, 0] = 1
            FBM.append(torch.fft.fft2(B)[None, :, :])
    FBM = torch.vstack(FBM)

    return FBM[None, :, :, :]


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

    return torch.reshape(img, (img.shape[0], img.shape[1], nc, nl)).transpose(2, 3)


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
        mask = torch.nonzero(mm.transpose(0,1).flatten())[:, 0]
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

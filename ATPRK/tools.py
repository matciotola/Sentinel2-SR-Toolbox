import torch
from torch import nn
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import numpy as np
def psf_template(s, w, b):
    h0 = torch.zeros((2 * w + 1) * s, (2 * w + 1) * s)
    for i in range(1, (2 * w + 1) * s + 1):
        for j in range(1, (2 * w + 1) * s + 1):
            dis2 = (torch.norm(torch.tensor([i - 0.5, j - 0.5]) - torch.tensor([(2 * w + 1) * s / 2, (2 * w + 1) * s / 2]))) ** 2
            h0[i -1, j - 1] = torch.exp(-dis2 / (2 * b ** 2))
    h0 = h0 / torch.sum(h0)
    return h0


def extend_plane(z, w):
    z_ext = TF.pad(z, [w, w, w, w], padding_mode='edge')
    return z_ext


def downsample_plane(plane, s, w, psf):
    plane = extend_plane(plane, w*s)
    _, sizec, sized = plane.shape

    S = torch.zeros([sizec//s, sized//s], dtype=plane.dtype, device=plane.device)
    for i in range(w*s+1, sizec-w*s+1, s):
        for j in range(w*s+1, sized-w*s+1, s):
            m = (i+s-1) // s
            n = (j+s-1) // s

            lw = plane[:, i-w*s-1:i+w*s+s-1, j-w*s-1:j+w*s+s-1]
            S[m-1, n-1] = torch.sum(lw * psf)

    S = S[w:-w, w:-w]

    return S


def downsample_cube(cube, s, w, psf):
    bs, c0, a0, b0 = cube.shape
    S = []
    for i in range(c0):
        S.append(downsample_plane(cube[:, i, :, :], s, w, psf)[None, :, :])

    S = torch.cat(S, dim=0)
    return S[None, :, :, :]


def evaluate_relation(img1, img2):

    rmse = torch.sqrt(F.mse_loss(img1, img2))
    cc = cross_correlation(img1, img2)

    return rmse, cc


def cross_correlation(img1, img2):
    _, a, b = img1.shape
    c1 = torch.sum(img1 * img2, dim=[1,2]) - a*b*torch.mean(img1, dim=[1,2])*torch.mean(img2, dim=[1,2])
    c2 = torch.sum(img1 ** 2, dim=[1,2]) - a*b*torch.mean(img1, dim=[1,2]) ** 2
    c3 = torch.sum(img2 ** 2, dim=[1,2]) - a*b*torch.mean(img2, dim=[1,2]) ** 2
    cc = c1 / torch.sqrt(c2 * c3)
    return cc

def semivariogram(img, h):

    a, b = img.shape
    n1 = 0
    r1 = 0
    for i in range(h, a):
        for j in range(b):
            r1 += (img[i, j] - img[i-h, j]) ** 2
            n1 += 1

    n2 = 0
    r2 = 0
    for i in range(a-h):
        for j in range(b):
            r2 += (img[i, j] - img[i+h, j]) ** 2
            n2 += 1

    n3 = 0
    r3 = 0
    for i in range(a):
        for j in range(h, b):
            r3 += (img[i, j] - img[i, j-h]) ** 2
            n3 += 1

    n4 = 0
    r4 = 0
    for i in range(a):
        for j in range(b-h):
            r4 += (img[i, j] - img[i, j+h]) ** 2
            n4 += 1
    r = r1 + r2 + r3 + r4
    n = n1 + n2 + n3 + n4
    rh = r / (2*n)

    return rh


def objective_function(x, xdata):
    f = x[0] * (1 - torch.exp(- xdata / x[1]))
    return f


class ObjFunct(nn.Module):
    def __init__(self, x_data, y_data):
        super(ObjFunct, self).__init__()
        self.x_data = x_data
        self.y_data = y_data

    def forward(self, x):
        output = x[0] * (1 - torch.exp(- self.x_data / x[1]))
        loss = F.mse_loss(output, self.y_data, reduction='none')
        return loss


def r_area_area(h, s, x):
    ll1 = torch.zeros([h+1, 1], dtype=x.dtype, device=x.device)
    m1, n1 = torch.where (ll1 == 0)
    ll2 = torch.zeros([s, s], dtype=x.dtype, device=x.device)
    n2, m2 = torch.where(ll2 == 0)

    raa = torch.zeros([h+1], dtype=x.dtype, device=x.device)
    for i in range(h+1):
        temp = 0
        for m in range(s ** 2):
            for n in range(s ** 2):
                p1 = torch.tensor([m1[i]*s + (m2[m] + 1), n1[i]*s + n2[m] + 1], dtype=x.dtype, device=x.device)
                p2 = torch.tensor([m1[0]*s + m2[n] + 1, n1[0]*s + n2[n] + 1], dtype=x.dtype, device=x.device)
                raa[i] = raa[i] + objective_function(x, torch.norm(p1 - p2, p=2))

    #raa = torch.tensor(raa, dtype=x.dtype, device=x.device)
    raa = raa / s ** 4

    return raa

def atp_deconvolution(h, s, x_area, sill_min, range_min, l_sill, l_range, rate, diff_min=1e6):

    fa0 = objective_function(x_area, torch.arange(1, s*h+1, dtype=x_area.dtype, device=x_area.device))
    fa0_vector = fa0[s-1::s][None, :]

    for i in range(1, l_sill + 1):
        for j in range(1, l_range + 1):
            xp = torch.tensor([(sill_min + i * rate)*x_area[0], (range_min + j * rate)*x_area[1]], dtype=x_area.dtype, device=x_area.device)
            raa0 = r_area_area(h, s, xp)
            raa = raa0[1:h+1] - raa0[0]
            dif = torch.norm(raa - fa0_vector, p=2)
            if dif < diff_min:
                x_best = xp
                diff_min = dif

    return x_best


def fine_coarse(p_vm, w, s, x, psf):
    ll1 = torch.zeros([2*w+1, 2*w+1], dtype=x.dtype, device=x.device)
    n1, m1 = torch.where(ll1 == 0)

    rvv = torch.zeros([(2*w+1)**2], dtype=x.dtype, device=x.device)
    for i in range((2*w+1)**2):
        tvv = torch.zeros([(2*w+1)*s, (2*w+1)*s], dtype=x.dtype, device=x.device)
        for ii in range((2*w+1)*s):
            for jj in range((2*w+1)*s):
                p1 = torch.tensor([(m1[i] - w) * s + ii + 1, (n1[i] - w) * s + jj + 1], dtype=x.dtype, device=x.device)
                tvv[ii, jj] = objective_function(x, torch.norm(p_vm - p1, p=2))
        rvv[i] = torch.sum(tvv * psf)

    return rvv


def atpk_noinform_yita(s, w, x, psf):

    tvv = t_coarse_coarse(w, s, x, psf)
    yita = torch.zeros([s, s, (2*w+1)**2 + 1], dtype=x.dtype, device=x.device)
    for i in range(s):
        for j in range(s):
            coords_vm = torch.tensor([w*s+i + 1, w*s+j + 1], dtype=x.dtype, device=x.device)
            rvv = fine_coarse(coords_vm, w, s, x, psf)

            matrix = torch.zeros([(2*w+1)**2 +1, (2*w+1)**2 + 1], dtype=x.dtype, device=x.device)
            matrix[:-1, :-1] = tvv
            matrix[-1, :] = 1
            matrix[:, -1] = 1
            matrix[-1, -1] = 0

            vector = torch.zeros([(2*w+1)**2 + 1], dtype=x.dtype, device=x.device)
            vector[:-1] = rvv
            vector[-1] = 1
            yita[i, j, :] = torch.inverse(matrix) @ vector

    return yita


def t_coarse_coarse(w, s, xx, psf):
    ll1 = torch.zeros(2*w+1, 2*w+1)
    n1, m1 = torch.where(ll1 == 0)
    tvv = torch.zeros([(2*w+1)**2, (2*w+1)**2], dtype=xx.dtype, device=xx.device)

    for i in range((2*w+1)**2):
        for j in range((2*w+1)**2):
            tv_v = torch.zeros([(2*w+1)*s, (2*w+1)*s], dtype=xx.dtype, device=xx.device)
            for ii in range((2*w+1)*s):
                for jj in range((2*w+1)*s):
                    t_v_v = torch.zeros([(2*w+1)*s, (2*w+1)*s], dtype=xx.dtype, device=xx.device)
                    for iii in range((2*w+1)*s):
                        for jjj in range((2*w+1)*s):
                            p1 = torch.tensor([(m1[i] - w) * s + iii + 1, (n1[i] - w) * s + jjj + 1], dtype=xx.dtype,
                                              device=xx.device)
                            p2 = torch.tensor([(m1[j] - w) * s + ii + 1, (n1[j] - w) * s + jj + 1], dtype=xx.dtype, device=xx.device)
                            t_v_v[iii, jjj] = objective_function(xx, torch.norm(p1 - p2, p=2))

                    tv_v[ii, jj] = torch.sum(t_v_v * psf)

            tvv[i,j] = torch.sum(tv_v * psf)

    return tvv


def atpk_noinform(s, w, ss, yita):

    _, _, c, d = ss.shape
    simulated_part = torch.zeros([c - w * 2, d - 2 * w], dtype=yita.dtype, device=yita.device)
    n1, m1 = torch.where(simulated_part == 0)
    number_m1 = m1.shape[0]
    m1 = m1 + w
    n1 = n1 + w
    p_vm = torch.zeros([c * s, d * s], dtype=yita.dtype, device=yita.device)
    for k in range(number_m1):
        for i in range(s):
            for j in range(s):
                local_w = ss[:, :, m1[k] - w:m1[k] + w + 1, n1[k] - w:n1[k] + w + 1]
                co = torch.squeeze(yita[i, j, :-1])
                p_vm[m1[k]* s + i, n1[k] * s + j] = torch.sum(torch.squeeze(local_w).flatten() @ co[:, None])

    return p_vm


def mldivide(y, X):

    y = y.double()
    X = X.double()

    Q, R = torch.linalg.qr(X)
    """
    p = torch.sum(torch.abs(R.diagonal(dim1=1, dim2=2)) > max(n, ncolX)*eps(R[:,0,0]), dim=1)
    P = torch.bmm(torch.pinverse(X), Q)
    P = torch.bmm(P, R)
    _, perm = torch.max(P, 1)
    """
    b = torch.linalg.solve(R, Q.transpose(1, 2) @ y)
    return b

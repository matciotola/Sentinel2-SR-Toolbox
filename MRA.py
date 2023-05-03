import torch
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode as Inter
import matplotlib
matplotlib.use('Qt5Agg')

from spectral_tools import mtf, LPFilter, LPfilterGauss
from aux_tools import batch_cov, estimation_alpha


def AWLP(bands_low, bands_high, ratio):

    bs, c, h, w = bands_low.shape

    mean_low = torch.mean(bands_low, dim=1, keepdim=True)

    img_intensity = bands_low / (mean_low + torch.finfo(bands_low.dtype).eps)

    bands_high = bands_high.repeat(1, c, 1, 1)

    bands_high_lp = resize(
                            resize(bands_high,
                                   [bands_high.shape[2] // ratio, bands_high.shape[3] // ratio],
                                   interpolation=Inter.BICUBIC,
                                   antialias=True),
                            [bands_high.shape[2], bands_high.shape[3]],
                            interpolation=Inter.BICUBIC,
                            antialias=True)

    bands_high = (bands_high - torch.mean(bands_high, dim=(2, 3), keepdim=True))*(torch.std(bands_low, dim=(2,3), keepdim=True)/torch.std(bands_high_lp, dim=(2,3), keepdim=True)) + torch.mean(bands_low, dim=(2,3), keepdim=True)

    bands_high_lpp = []
    for i in range(bands_high.shape[1]):
        bands_high_lpp.append(LPFilter(bands_high[:, i, None, :, :], ratio))

    bands_high_lpp = torch.cat(bands_high_lpp, dim=1)

    details = bands_high - bands_high_lpp

    fused = details * img_intensity + bands_low

    return fused


def MTF_GLP(bands_low, bands_high, sensor, ratio):

    bs, c, h, w = bands_low.shape

    bands_hr = bands_high.repeat(1, c, 1, 1)
    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2,3), keepdim=True))*(torch.std(bands_low, dim=(2,3), keepdim=True)/torch.std(LPfilterGauss(bands_hr, ratio), dim=(2,3), keepdim=True)) + torch.mean(bands_low, dim=(2,3), keepdim=True)

    bands_high_lp = mtf(bands_hr, sensor, ratio)
    bands_high_lr_lr = resize(bands_high_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT, antialias=False)
    bands_high_lr = interp23tap_torch(bands_high_lr_lr, ratio, bands_high_lr_lr.device)

    fused = bands_low + bands_hr - bands_high_lr

    return fused


def MTF_GLP_FS(bands_low, bands_high, sensor, ratio):

    bs, c, h, w = bands_low.shape

    bands_hr = bands_high.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)

    bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio, bands_hr_lr_lr.device)

    low_covs = []
    high_covs = []

    for i in range(bands_low.shape[1]):

        points_lr = torch.cat([bands_low[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_lr = torch.flatten(points_lr, start_dim=2)

        points_hr = torch.cat([bands_hr_lr[:, i, None, :, :], bands_hr[:, i, None, :, :]], dim=1)
        points_hr = torch.flatten(points_hr, start_dim=2)

        low_covs.append(batch_cov(points_lr.transpose(1, 2))[:, None, :, :])
        high_covs.append(batch_cov(points_hr.transpose(1, 2))[:, None, :, :])

    low_covs = torch.cat(low_covs, dim=1)
    high_covs = torch.cat(high_covs, dim=1)

    gamma = low_covs[:, :, 0, 1] / high_covs[:, :, 0, 1]
    gamma = gamma[:, :, None, None]
    fused = bands_low + gamma * (bands_hr - bands_hr_lr)

    return fused


def MTF_GLP_HPM(bands_low, bands_high, sensor, ratio):

    bs, c, h, w = bands_low.shape

    bands_hr = bands_high.repeat(1, c, 1, 1)

    bands_hr = (bands_hr - torch.mean(bands_hr, dim=(2, 3), keepdim=True))*(torch.std(bands_low, dim=(2, 3), keepdim=True)/torch.std(LPfilterGauss(bands_hr, ratio), dim=(2, 3), keepdim=True)) + torch.mean(bands_low, dim=(2, 3), keepdim=True)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio, bands_hr_lr_lr.device)

    fused = bands_low * (bands_hr / (bands_hr_lr +  torch.finfo(bands_low.dtype).eps))

    return fused


def MTF_GLP_HPM_H(bands_low, bands_high, sensor, ratio, decimation=True):

    bs, c, h, w = bands_low.shape

    min_bands_low = torch.amin(bands_low, dim=(2, 3), keepdim=True)
    bands_high_lp = LPfilterGauss(bands_high, ratio)

    inp = torch.cat([torch.ones(bands_high_lp.shape, dtype=bands_high_lp.dtype, device=bands_high_lp.device),
                     bands_low],
                    dim=1)

    alpha = estimation_alpha(inp, bands_high_lp)

    alpha_p = torch.bmm(torch.squeeze(alpha, -1).transpose(1, 2), torch.squeeze(
        torch.cat([torch.ones((bs, 1, 1, 1), device=alpha.device, dtype=alpha.dtype), min_bands_low], dim=1), -1))[:, :, :, None]

    bands_hr = bands_high.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)

    if decimation:
        bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
        bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio, bands_hr_lr_lr.device)

    bands_hr_pl = (bands_hr - alpha_p) / (bands_hr_lr - alpha_p + torch.finfo(bands_hr_lr.dtype).eps)

    bands_low_l = bands_low - min_bands_low

    fused = bands_low_l * bands_hr_pl + min_bands_low

    return fused

def MTF_GLP_HPM_R(bands_low, bands_high, sensor, ratio):

    bs, c, h, w = bands_low.shape

    bands_hr = bands_high.repeat(1, c, 1, 1)

    bands_hr_lp = mtf(bands_hr, sensor, ratio)
    bands_hr_lr_lr = resize(bands_hr_lp, [h // ratio, w // ratio], interpolation=Inter.NEAREST_EXACT)
    bands_hr_lr = interp23tap_torch(bands_hr_lr_lr, ratio, bands_hr_lr_lr.device)

    g = []
    for i in range(c):

        inp = torch.flatten(torch.cat([bands_low[:, i, None, :, :], bands_hr_lr[:, i, None, :, :]], dim=1), start_dim=2).transpose(1,2)
        C = batch_cov(inp)
        g.append((C[:, 0, 1] / C[:, 1, 1])[:, None])

    g = torch.cat(g, dim=1) [:, :, None, None]
    cb = torch.mean(bands_low, dim=(2, 3), keepdim=True) / g - torch.mean(bands_hr, dim=(2, 3), keepdim=True)

    fused = bands_low * (bands_hr + cb) / (bands_hr_lr + cb + torch.finfo(bands_low.dtype).eps)


    return fused

if __name__ == '__main__':
    from scipy import io
    import numpy as np
    from interpolator_tools import interp23tap_torch
    from show_results import show

    temp = io.loadmat('/home/matteo/Desktop/Datasets/WV3_Adelaide_crops/Adelaide_3.mat')

    pan = temp['I_PAN'].astype(np.float32)
    ms = temp['I_MS_LR'].astype(np.float32).transpose(2, 0, 1)

    pan = torch.tensor(pan)[None, None, :, :]
    ms = torch.tensor(ms)[None, :, :, :]
    ratio = 4
    ms_exp = interp23tap_torch(ms, 4, ms.device).float()

    fused = MTF_GLP_HPM_R(ms_exp, pan, 'WV3', ratio)
    f = fused.detach().cpu().numpy()
    b10 = pan.detach().cpu().numpy()
    b20 = ms.detach().cpu().numpy()

    f = np.moveaxis(np.squeeze(f), 0, -1)
    b20 = np.moveaxis(np.squeeze(b20), 0, -1)
    b10 = np.squeeze(b10)

    show(b20, b10, f, ratio=ratio, method='GSA')

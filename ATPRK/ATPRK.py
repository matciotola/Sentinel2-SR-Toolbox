from torchmin import least_squares
from tqdm import tqdm
try:
    from .tools import *
except:
    from tools import *

def SYNTH_ATPRK(ordered_dict):

    bands_10 = torch.clone(ordered_dict.bands_10) / 2 ** 16
    bands_lr = torch.clone(ordered_dict.ms_lr) / 2 ** 16
    ratio = ordered_dict.ratio
    # hyperparameters
    s = ratio
    w = 1
    sigma = ratio / 2
    sill_min = 1
    range_min = 0.5
    l_sill = 20
    l_range = 20
    rate = 0.1
    h = 20

    psfh = psf_template(s, w, sigma)

    fused = []
    for i in tqdm(range(bands_lr.shape[1])):
        fused.append(atprk_ms(bands_lr[:, i, None, :, :], bands_10, sill_min, range_min, l_sill, l_range, rate, h, w, psfh))

    return torch.cat(fused, dim=1) * 2 ** 16

def atprk_ms(bands_low_lr, bands_10, sill_min, range_min, l_sill, l_range, rate, h, w, psfh):
    _, c1, a1, b1 = bands_low_lr.shape
    _, c2, a2, b2 = bands_10.shape

    bands_10 = bands_10.double()
    bands_low_lr = bands_low_lr.double()

    s = a2 // a1

    # linear regression modeling

    bands_10_upscaled = downsample_cube(bands_10, s, w, psfh)
    gb0 = torch.flatten(bands_10_upscaled.transpose(2, 3), start_dim=2)
    gb1 = torch.flatten(bands_low_lr.transpose(2, 3), start_dim=2)

    ones_fill = torch.ones([gb0.shape[0], 1, gb0.shape[2]], dtype=gb0.dtype, device=gb0.device)

    a = torch.cat([ones_fill, gb0], dim=1).transpose(1,2)
    b = gb1.transpose(1,2)

    xrc1 = torch.linalg.lstsq(a, b).solution

    gbf = torch.flatten(bands_10.transpose(2, 3), start_dim=2)
    ones_fill_fr = torch.ones([gbf.shape[0], 1, gbf.shape[2]], dtype=gbf.dtype, device=gbf.device)

    ff1 = torch.cat([ones_fill_fr, gbf], dim=1).transpose(1, 2) @ xrc1
    z_r = ff1.reshape([bands_10.shape[0], 1, bands_10.shape[2], bands_10.shape[3]]).transpose(2, 3)

    # residual calculation

    z_r_upscaled = downsample_plane(z_r[:, 0, :, :], s, w, psfh)[None, None, :, :]
    rb = bands_low_lr - z_r_upscaled

    # ATPRK for residual
    rb_ext = extend_plane(rb, w)

    x0 = torch.tensor([0.1, 1], dtype=rb.dtype, device=rb.device)
    rh = []
    for i in range(h):
        rh.append(semivariogram(torch.squeeze(rb), i + 1))
    xdata = torch.arange(s, s * h + 1, s, dtype=rb.dtype, device=rb.device)
    rh = torch.tensor(rh, dtype=rb.dtype, device=rb.device)
    # x = least_square_curve_fitting(objective_function, x0, xdata, rh)

    # obj_fun = ObjFunct(xdata, rh)
    # x1 = least_squares(obj_fun, x0, max_nfev=400, tr_solver='exact').x
    x1 = lsqcurvefit(obj_func_v1, torch.clone(x0), xdata, rh)
    fa1 = objective_function(x1, torch.arange(1, s * h + 1, dtype=rb.dtype, device=rb.device))
    xp_best = atp_deconvolution(h, s, x1, sill_min, range_min, l_sill, l_range, rate)
    raa0 = r_area_area(h, s, xp_best)
    raa = raa0[1:] - raa0[0]
    x2 = lsqcurvefit(obj_func_v1, torch.clone(x0), xdata, raa)
    # obj_fun = ObjFunct(xdata, raa)
    # x2 = least_squares(obj_fun, x0, max_nfev=400, tr_solver='exact').x
    fa2 = objective_function(x2, torch.arange(1, s * h + 1, dtype=rb.dtype, device=rb.device))

    yita = atpk_noinform_yita(s, w, xp_best, psfh)

    p_vm = atpk_noinform(s, w, rb_ext, yita)

    # fusion

    z_atpk = p_vm[w * s:-w * s, w * s:-w * s]
    fused = z_r + z_atpk

    return fused



def SEL_ATPRK(ordered_dict):

    bands_10 = torch.clone(ordered_dict.bands_10) / 2 ** 16
    bands_lr = torch.clone(ordered_dict.ms_lr) / 2 ** 16
    ratio = ordered_dict.ratio

    # hyperparameters

    s = ratio
    w = 1
    sigma = ratio / 2

    sill_min = 1
    range_min = 0.5
    l_sill = 20
    l_range = 20
    rate = 0.1
    h = 20

    # selection of the best band

    psfh = psf_template(s, w, sigma)

    bands_10_downsampled = downsample_cube(bands_10, s, w, psfh)

    cc_matrix = torch.zeros([bands_lr.shape[1], bands_10.shape[1]], dtype=bands_lr.dtype, device=bands_lr.device)

    for i in range(bands_lr.shape[1]):
        for j in range(bands_10.shape[1]):
            rmse, cc = evaluate_relation(bands_lr[:, i, :, :], bands_10_downsampled[:, j, :, :])
            cc_matrix[i, j] = cc

    ii, jj = torch.max(cc_matrix, dim=1)

    # ATPRK
    fused = []
    for i in tqdm(range(bands_lr.shape[1])):
        fused.append(atprk_pan(bands_lr[:, i, None, :, :], bands_10[:, jj[i], None, :, :], sill_min, range_min, l_sill, l_range, rate, h, w, psfh))
    return torch.cat(fused, dim=1) * 2 ** 16


def atprk_pan(bands_low_lr, bands_10, sill_min, range_min, l_sill, l_range, rate, h, w, psfh):

    _, c1, a1, b1 = bands_low_lr.shape
    _, c2, a2, b2 = bands_10.shape

    bands_10 = bands_10.double()
    bands_low_lr = bands_low_lr.double()

    s = a2 // a1

    # linear regression modeling

    bands_10_upscaled = downsample_plane(bands_10[:, 0, :, :], s, w, psfh)[None, None, :, :]

    ones_fills = torch.ones(bands_10_upscaled.shape, dtype=bands_10_upscaled.dtype, device=bands_10_upscaled.device)
    bands_10_upscaled_col = torch.flatten(torch.cat([bands_10_upscaled.transpose(-2, -1), ones_fills], dim=1), start_dim=2)
    bands_low_col = torch.flatten(bands_low_lr.transpose(-2, -1), start_dim=2)

    # alpha = (bands_10_upscaled_col.transpose(1,2).pinverse() @ bands_low_col.transpose(1,2))#[:, None, :, :]

    alpha = mldivide(bands_low_col.transpose(1, 2), bands_10_upscaled_col.transpose(1, 2))

    ones_fills_hr = torch.ones(bands_10.shape, dtype=bands_10.dtype, device=bands_10.device)
    bands_high_col = torch.flatten(torch.cat([bands_10.transpose(-2, -1), ones_fills_hr], dim=1), start_dim=2)

    z_r_col = torch.sum(bands_high_col * alpha, dim=1, keepdim=True)
    z_r = z_r_col.reshape(bands_10.shape).transpose(2, 3)

    # residual calculation

    z_r_upscaled = downsample_plane(z_r[:, 0, :, :], s, w, psfh)[None, None, :, :]
    rb = bands_low_lr - z_r_upscaled

    # ATPK for residuals
    rb_ext = extend_plane(rb, w)

    x0 = torch.tensor([0.1, 1], dtype=rb.dtype, device=rb.device)
    rh = []
    for i in range(h):
        rh.append(semivariogram(torch.squeeze(rb),i+1))
    xdata = torch.arange(s, s*h+1, s, dtype=rb.dtype, device=rb.device)
    rh = torch.tensor(rh, dtype=rb.dtype, device=rb.device)
    #x = least_square_curve_fitting(objective_function, x0, xdata, rh)

    # obj_fun = ObjFunct(xdata, rh)
    # x1 = least_squares(obj_fun, torch.clone(x0), max_nfev=400, tr_solver='exact').x
    x1 = lsqcurvefit(obj_func_v1, torch.clone(x0), xdata, rh)
    fa1 = objective_function(x1, torch.arange(1, s*h+1, dtype=rb.dtype, device=rb.device))
    xp_best = atp_deconvolution(h, s, x1, sill_min, range_min, l_sill, l_range, rate)
    raa0 = r_area_area(h, s, xp_best)
    raa = raa0[1:] - raa0[0]
    # obj_fun = ObjFunct(xdata, raa)
    x2 = lsqcurvefit(obj_func_v1, torch.clone(x0), xdata, raa)
    # x2 = least_squares(obj_fun, torch.clone(x0), max_nfev=400, tr_solver='exact').x
    fa2 = objective_function(x2, torch.arange(1, s*h+1, dtype=rb.dtype, device=rb.device))

    yita = atpk_noinform_yita(s, w, xp_best, psfh)

    p_vm = atpk_noinform(s, w, rb_ext, yita)

    # fusion

    z_atpk = p_vm[w*s:-w*s, w*s:-w*s]
    fused = z_r + z_atpk

    return fused




if __name__ == '__main__':

    from scipy import io
    import numpy as np
    from recordclass import recordclass

    bands_low = io.loadmat('/media/matteo/T7/Dataset_Ugliano/MAT/60/New_York.mat')['S2_60m'].astype(np.float64)
    bands_high = io.loadmat('/media/matteo/T7/Dataset_Ugliano/MAT/10/New_York_60.mat')['S2_10m'].astype(np.float64)

    ratio = 6

    bands_low = torch.tensor(np.moveaxis(bands_low, -1, 0)[None, :, :, :])
    bands_high = torch.tensor(np.moveaxis(bands_high, -1, 0)[None, :, :, :])

    ord_dic = {'bands_low_lr': bands_low, 'bands_high': bands_high, 'ratio': ratio}

    exp_input = recordclass('exp_info', ord_dic.keys())(*ord_dic.values())

    fused = SEL_ATPRK(exp_input)


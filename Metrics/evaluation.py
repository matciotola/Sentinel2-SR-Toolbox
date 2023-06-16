import torch.nn.functional as F
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
try:
    import metrics_rr as rr
    import metrics as fr
    from coregistration import stacked_fineshift
except:
    from Metrics import metrics_rr as rr
    from Metrics import metrics as fr
    from Metrics.coregistration import stacked_fineshift

def evaluation_rr(out_lr, bands_low_lr, ratio):
    if out_lr.shape[1] == 2:
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    ergas = rr.ERGAS(ratio).to(out_lr.device)
    sam = rr.SAM().to(out_lr.device)
    q = rr.Q(out_lr.shape[1]).to(out_lr.device)
    q2n = rr.Q2n().to(out_lr.device)

    ergas_index, _ = ergas(out_lr, bands_low_lr)
    sam_index, _ = sam(out_lr, bands_low_lr)
    q_index = q(out_lr, bands_low_lr)
    q2n_index, _ = q2n(out_lr, bands_low_lr)

    return ergas_index.item(), sam_index.item(), q_index.item(), q2n_index.item()


def evaluation_fr(out, bands_high, bands_low_lr, ratio):

    if ratio == 2:
        sensor = 'S2-20'
        starting = 1
        sigma = 4
    elif ratio == 6 and out.shape[1] == 3:
        sensor = 'S2-60'
        starting = 3
        sigma = 6
    else:
        sensor = 'S2-60_bis'
        starting = 3
        sigma = 6
        bands_low_lr = bands_low_lr[:, :-1, :, :]

    kernel = mtf_kernel_to_torch(gen_mtf(ratio, sensor))

    out_lp = F.conv2d(out, kernel.type(out.dtype).to(out.device), padding='same', groups=out.shape[1])

    out_lp_shifted = stacked_fineshift(out_lp, 1, 1)
    out_lr = out_lp_shifted[:, :, starting::ratio, starting::ratio]

    ergas = rr.ERGAS(ratio).to(out.device)
    sam = rr.SAM().to(out.device)
    q = rr.Q(out.shape[1]).to(out.device)
    q2n = rr.Q2n().to(out.device)
    d_rho = fr.D_rho(sigma).to(out.device)

    ergas_index, _ = ergas(out_lr, bands_low_lr)
    sam_index, _ = sam(out_lr, bands_low_lr)
    q_index = q(out_lr, bands_low_lr)
    q2n_index, _ = q2n(out_lr, bands_low_lr)

    d_rho_index, _ = d_rho(out, bands_high)

    return ergas_index.item(), sam_index.item(), q_index.item(), q2n_index.item(), d_rho_index.item()


if __name__ == '__main__':
    from Utils.load_save_tools import open_tiff

    bands_10 = open_tiff('/media/matteo/T7/Dataset_Ugliano/10/New_York.tif')
    bands_20 = open_tiff('/media/matteo/T7/Dataset_Ugliano/20/New_York.tif')
    out = open_tiff('/media/matteo/T7/outputs_Ugliano/New_York/FR/20/SYNTH-BDSD.tiff')

    ciccio = evaluation_fr(out, bands_10, bands_20, 2)



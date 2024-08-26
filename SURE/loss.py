import torch
from torch import nn
from .tools import Ax, back_projx, invAAtx

class BPLoss(nn.Module):

    def __init__(self, psf_20, psf_60, cond):
        super(BPLoss, self).__init__()
        self.psf_20 = psf_20
        self.psf_60 = psf_60
        self.cond = cond

    def forward(self, outputs, bands_10, bands_20, bands_60):

        outputs_10 = outputs[:, :4, :, :]
        outputs_20 = outputs[:, 4:10, :, :]
        outputs_60 = outputs[:, 10:, :, :]

        hx_hat_20 = Ax(outputs_20.squeeze(0), self.psf_20, 2) - bands_20.squeeze(0)
        hx_hat_60 = Ax(outputs_60.squeeze(0), self.psf_60, 6) - bands_60.squeeze(0)
        px_hat_10 = (outputs_10 - bands_10).squeeze(0)
        px_hat_20 = back_projx(hx_hat_20, self.psf_20, 2, self.cond)
        px_hat_60 = back_projx(hx_hat_60, self.psf_60, 6, self.cond)

        loss_bp = torch.sum(px_hat_10 ** 2) + torch.sum(px_hat_20 ** 2) + torch.sum(px_hat_60 ** 2)

        return loss_bp


class SURELoss(nn.Module):
    def __init__(self, psf_20, psf_60, sigma_10, sigma_20, sigma_60, cond, ep=1e-5):
        super(SURELoss, self).__init__()
        self.psf_20 = psf_20
        self.psf_60 = psf_60
        self.sigma_10 = sigma_10 ** 2
        self.sigma_20 = sigma_20 ** 2
        self.sigma_60 = sigma_60 ** 2
        self.cond = cond
        self.ep = ep

    def forward(self, net_input, outputs):

        outputs_10 = outputs[:, :4, :, :]
        outputs_20 = outputs[:, 4:10, :, :]
        outputs_60 = outputs[:, 10:, :, :]

        h_oute_20 = Ax(outputs_20.squeeze(0), self.psf_20, 2)
        h_oute_60 = Ax(outputs_60.squeeze(0), self.psf_60, 6)
        tmp_20 = invAAtx(h_oute_20, self.psf_20, 2, self.cond)*self.sigma_20.reshape(6, 1, 1).to(outputs.device)
        tmp_60 = invAAtx(h_oute_60, self.psf_60, 6, self.cond)*self.sigma_60.reshape(2, 1, 1).to(outputs.device)

        out_ep_10 = outputs_10.squeeze(0)*self.sigma_10.reshape(4, 1, 1).to(outputs.device)
        out_ep_20 = back_projx(tmp_20, self.psf_20, 2, self.cond)
        out_ep_60 = back_projx(tmp_60, self.psf_60, 6, self.cond)

        div = torch.sum(net_input.squeeze(0)[:4, :, :]*out_ep_10) + torch.sum(net_input.squeeze(0)[4:10, :, :]*out_ep_20) +torch.sum(net_input.squeeze(0)[10:, :, :]*out_ep_60)

        return div



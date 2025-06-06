import torch
from torch import nn as nn


class SpectralLoss(nn.Module):
    def __init__(self, ):
        super(SpectralLoss, self).__init__()
        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):
        L = self.loss(outputs, labels)

        return L


class StructLoss(nn.Module):
    def __init__(self, nbands=6):
        super(StructLoss, self).__init__()

        self.eps = 1e-16

        self.gradients_y = nn.Conv2d(in_channels=nbands,
                                     padding='same', kernel_size=3,
                                     out_channels=nbands, bias=False, groups=nbands)
        self.gradients_x = nn.Conv2d(in_channels=nbands,
                                     padding='same', kernel_size=3,
                                     out_channels=nbands, bias=False, groups=nbands)
        self.gradients_D = nn.Conv2d(in_channels=nbands,
                                     padding='same', kernel_size=3,
                                     out_channels=nbands, bias=False, groups=nbands)
        self.gradients_d = nn.Conv2d(in_channels=nbands,
                                     padding='same', kernel_size=3,
                                     out_channels=nbands, bias=False, groups=nbands)

        Gy = ((torch.Tensor([[0, 1, 0], [0, 0, 0], [0, -1, 0]]))[None, None, :, :]).repeat([nbands, 1, 1, 1])
        Gx = ((torch.Tensor([[0, 0, 0], [1, 0, -1], [0, 0, 0]]))[None, None, :, :]).repeat([nbands, 1, 1, 1])
        GD = ((torch.Tensor([[1, 0, 0], [0, 0, 0], [0, 0, -1]]))[None, None, :, :]).repeat([nbands, 1, 1, 1])
        Gd = ((torch.Tensor([[0, 0, 1], [0, 0, 0], [-1, 0, 0]]))[None, None, :, :]).repeat([nbands, 1, 1, 1])

        self.gradients_y.weight.data = Gy
        self.gradients_x.weight.data = Gx
        self.gradients_D.weight.data = GD
        self.gradients_d.weight.data = Gd

        self.gradients_y.requires_grad_ = False
        self.gradients_x.requires_grad_ = False
        self.gradients_D.requires_grad_ = False
        self.gradients_d.requires_grad_ = False

        self.loss = nn.L1Loss(reduction='mean')

    def forward(self, outputs, labels):
        g_out_y = self.gradients_y(outputs)
        g_out_x = self.gradients_x(outputs)
        g_out_D = self.gradients_D(outputs)
        g_out_d = self.gradients_d(outputs)

        g_out = torch.cat([g_out_y, g_out_x, g_out_D, g_out_d], dim=1)

        g_lab_y = self.gradients_y(labels)
        g_lab_x = self.gradients_x(labels)
        g_lab_D = self.gradients_D(labels)
        g_lab_d = self.gradients_d(labels)

        g_labels = torch.cat([g_lab_y, g_lab_x, g_lab_D, g_lab_d], dim=1)
        L = torch.sqrt(self.loss(g_out, g_labels) + self.eps)

        return L


class RegLoss(nn.Module):
    def __init__(self, nbands=6):
        super(RegLoss, self).__init__()

        self.l1_regularization = nn.Conv2d(in_channels=nbands,
                                           padding='same', kernel_size=2,
                                           out_channels=2, bias=False)
        W = torch.Tensor([[[0, 1, 0], [0, 0, 0], [0, -1, 0]], [[0, 0, 0], [1, 0, -1], [0, 0, 0]]])
        W = torch.unsqueeze(W, dim=1)
        W = W.repeat([1, nbands, 1, 1])
        self.l1_regularization.weight.data = W
        self.l1_regularization.requires_grad_ = False

    def forward(self, outputs):
        L = self.l1_regularization(outputs)
        L = torch.abs(L)
        L = torch.mean(L)
        return L

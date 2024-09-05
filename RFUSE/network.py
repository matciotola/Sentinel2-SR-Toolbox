import torch.nn as nn
import torch.nn.functional as F
import torch
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch
from Utils.imresize_bicubic import imresize
from Utils.interpolator_tools import interp23tap_torch


class RFUSEModel(nn.Module):
    def __init__(self, bands_10=4, bands_lr=6, ratio=2):
        super(RFUSEModel, self).__init__()
        self.ratio = ratio
        in_channels = bands_10 + bands_lr
        # Network structure
        h_bh = gen_mtf(self.ratio, 'S2-10', kernel_size=9)
        h_bh = mtf_kernel_to_torch(h_bh)

        if ratio == 2:
            sensor = 'S2-20'
        else:
            sensor = 'S2-60'

        h_bl = gen_mtf(self.ratio, sensor, kernel_size=9)
        h_bl = mtf_kernel_to_torch(h_bl)

        self.depthconv_10 = nn.Conv2d(in_channels=bands_10,
                                      out_channels=bands_10,
                                      groups=bands_10,
                                      padding='same',
                                      padding_mode='replicate',
                                      kernel_size=h_bh.shape[-1],
                                      bias=False)

        self.depthconv_10.weight.data = h_bh
        self.depthconv_10.weight.requires_grad = False

        self.depthconv_lr = nn.Conv2d(in_channels=bands_lr,
                                      out_channels=bands_lr,
                                      groups=bands_lr,
                                      padding='same',
                                      padding_mode='replicate',
                                      kernel_size=h_bl.shape[-1],
                                      bias=False)

        self.depthconv_lr.weight.data = h_bl
        self.depthconv_lr.weight.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=48, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(in_channels=48, out_channels=32, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(in_channels=32, out_channels=bands_lr, kernel_size=3, padding='same')

    def forward(self, bands_high, bands_low):

        bands_high_hp = bands_high - self.depthconv_10(bands_high)
        bands_low_hp = bands_low - self.depthconv_lr(bands_low)

        inp = torch.cat((bands_high_hp, bands_low_hp), dim=1)

        x = F.relu(self.conv1(inp))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = self.conv4(x)

        x = x + bands_low

        return x
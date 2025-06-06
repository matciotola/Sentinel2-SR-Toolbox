import torch
from torch import nn


class attention_FRU(nn.Module):
    def __init__(self, num_channels_down, pad='reflect'):
        super(attention_FRU, self).__init__()
        # layers to generate conditional convolution weights
        self.gen_se_weights1 = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad),
            nn.Softplus(),
            nn.Sigmoid())

        # create conv layers
        self.conv_1 = nn.Conv2d(num_channels_down, num_channels_down, 1, padding_mode=pad)
        self.norm_1 = nn.BatchNorm2d(num_channels_down, affine=False)
        self.actvn = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, guide, x):
        se_weights1 = self.gen_se_weights1(guide)
        dx = self.conv_1(x)
        dx = self.norm_1(dx)
        dx = torch.mul(dx, se_weights1)
        out = self.actvn(dx)
        return out

class S2AttentionNet(nn.Module):
    def __init__(self, g_channel=4, in_channel=8, out_channel=12, num_channels_down=64, num_channels_up=64, num_channels_skip=32,
                 filter_size_down=3, filter_size_up=3, filter_skip_size=1):
        super().__init__()
        self.FRU = attention_FRU(num_channels_down)
        self.enc0 = nn.Sequential(
            nn.Conv2d(g_channel, num_channels_down, filter_size_down,
                      padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc1 = nn.Sequential(
            nn.Conv2d(in_channel, num_channels_down, filter_size_down,
                      padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))
        self.enc = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_down, filter_size_down,padding='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_down),
            nn.LeakyReLU(0.2, inplace=True))

        self.skip = nn.Sequential(
            nn.Conv2d(num_channels_down, num_channels_skip, filter_skip_size, padding ='same', padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_skip),
            nn.LeakyReLU(0.2, inplace=True))

        self.dc = nn.Sequential(
            nn.Conv2d((num_channels_skip + num_channels_up), num_channels_up, filter_size_up,padding='same',padding_mode='reflect'),
            nn.BatchNorm2d(num_channels_up),
            nn.LeakyReLU(0.2, inplace=True))
        self.out_layer = nn.Sequential(
            nn.Conv2d(num_channels_up, out_channel, 1, padding_mode='reflect'),
            nn.Sigmoid())
        self.conv = nn.Conv2d(num_channels_down,num_channels_down,filter_size_down, padding = 'same',padding_mode = 'reflect')
    def forward(self, inputs):
        '''inputs: concat(y10,y20up,y60up)'''
        # encoder part
        yg = inputs[:,:4,:,:]
        yin = inputs[:, 4:, :, :]
        y_en0 = self.enc0(yg)
        y_en1 = self.enc(y_en0)
        y_en2 = self.enc(y_en1)
        y_en3 = self.enc(y_en2)
        y_en4 = self.enc(y_en3)
        # decoder part with skip connections
        y_dc0 = self.enc(y_en4)
        y_dc1 = self.dc(torch.cat((self.skip(y_en4), y_dc0), dim=1))
        y_dc2 = self.dc(torch.cat((self.skip(y_en3), y_dc1), dim=1))
        y_dc3 = self.dc(torch.cat((self.skip(y_en2), y_dc2), dim=1))
        y_dc4 = self.dc(torch.cat((self.skip(y_en1), y_dc3), dim=1))
        y_dc5 = self.dc(torch.cat((self.skip(y_en0), y_dc4), dim=1))

        xout1 = self.enc1(yin)  # c=64+6+2
        xout1 = self.FRU(y_dc1, self.conv(xout1))
        xout2 = self.FRU(y_dc2, self.conv(xout1))
        xout3 = self.FRU(y_dc3, self.conv(xout2))
        xout4 = self.FRU(y_dc4, self.conv(xout3))
        xout5 = self.FRU(y_dc5, self.conv(xout4))

        out = self.out_layer(xout5)

        return out
import torch
from torch import nn
import torch.nn.functional as func
from Utils.spectral_tools import gen_mtf, mtf_kernel_to_torch


class S2UCNN_model(nn.Module):
    def __init__(self, nbands_10=4, nbands_20=6, nbands_60=2):
        super(S2UCNN_model, self).__init__()

        self.conv1_60 = nn.Conv2d(nbands_60, 64, 3, padding='same', padding_mode='reflect')
        self.cn1_60 = nn.LayerNorm(64)

        self.conv1_20 = nn.Conv2d(nbands_20, 64, 3, padding='same', padding_mode='reflect')
        self.cn1_20 = nn.LayerNorm(64)

        self.conv2_20 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn2_20 = nn.LayerNorm(64)

        self.conv3_20 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn3_20 = nn.LayerNorm(64)

        self.conv4_20 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn4_20 = nn.LayerNorm(64)

        self.convf_20 = nn.Conv2d(nbands_20, 64, 3, padding='same', padding_mode='reflect')
        self.cnf_20 = nn.LayerNorm(64)

        self.conv2_60 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn2_60 = nn.LayerNorm(64)

        self.conv3_60 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn3_60 = nn.LayerNorm(64)

        self.conv4_60 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn4_60 = nn.LayerNorm(64)

        self.conv5_60 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn5_60 = nn.LayerNorm(64)

        self.conv6_60 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn6_60 = nn.LayerNorm(64)

        self.convf = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cnf = nn.LayerNorm(64)

        self.conv1_10 = nn.Conv2d(nbands_10, 64, 3, padding='same', padding_mode='reflect')
        self.cn1_10 = nn.LayerNorm(64)

        self.conv2_10 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn2_10 = nn.LayerNorm(64)

        self.conv3_10 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn3_10 = nn.LayerNorm(64)

        self.conv4_10 = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn4_10 = nn.LayerNorm(64)

        self.conv2_f = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn2_f = nn.LayerNorm(64)

        self.conv3_f = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn3_f = nn.LayerNorm(64)

        self.conv4_f = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn4_f = nn.LayerNorm(64)

        self.conv5_f = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn5_f = nn.LayerNorm(64)

        self.conv6_f = nn.Conv2d(64, 64, 3, padding='same', padding_mode='reflect')
        self.cn6_f = nn.LayerNorm(64)

        self.conv_fin = nn.Conv2d(64, nbands_10+nbands_20+nbands_60, 1, padding='same', padding_mode='reflect')


    def forward(self, bands_10, bands_20, bands60):

        feat_60 = func.leaky_relu(self.cn1_60(self.conv1_60(bands60).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_60_up = func.interpolate(feat_60, scale_factor=3, mode='bilinear')

        bands_20_up = func.interpolate(bands_20, scale_factor=2, mode='bilinear')
        feat_20_f1 = func.leaky_relu(self.cnf_20(self.convf_20(bands_20_up).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        feat_60_1 = func.leaky_relu(self.cn2_60(self.conv2_60(feat_60_up).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        feat_20_1 = func.leaky_relu(self.cn1_20(self.conv1_20(bands_20).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_20_2 = func.leaky_relu(self.cn2_20(self.conv2_20(feat_20_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_20_3 = func.leaky_relu(self.cn3_20(self.conv3_20(feat_20_2 + feat_20_2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_20_4 = func.leaky_relu(self.cn4_20(self.conv4_20(feat_20_3 + feat_20_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        feat_60_2 = func.leaky_relu(self.cn3_60(self.conv3_60(feat_60_1 + feat_20_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_60_3 = func.leaky_relu(self.cn4_60(self.conv4_60(feat_60_2 + feat_20_2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_60_4 = func.leaky_relu(self.cn5_60(self.conv5_60(feat_60_3 + feat_20_3).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_60_5 = func.leaky_relu(self.cn6_60(self.conv6_60(feat_60_4 + feat_20_4).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        feat_20_f2 = func.interpolate(feat_60_5, scale_factor=2, mode='bilinear')

        feat_20_f3 = func.leaky_relu(self.cn1_10(self.conv1_10(bands_10).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        feat_10_1 = func.leaky_relu(self.cn1_10(self.conv1_10(bands_10).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_10_2 = func.leaky_relu(self.cn2_10(self.conv2_10(feat_10_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_10_3 = func.leaky_relu(self.cn3_10(self.conv3_10(feat_10_2 + feat_10_2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_10_4 = func.leaky_relu(self.cn4_10(self.conv4_10(feat_10_3 + feat_10_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)


        feat_f = feat_20_f1 + feat_20_f2 + feat_20_f3

        feat_f_1 = func.leaky_relu(self.cn2_f(self.conv2_f(feat_f).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_f_2 = func.leaky_relu(self.cn3_f(self.conv3_f(feat_f_1 + feat_10_1).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_f_3 = func.leaky_relu(self.cn4_f(self.conv4_f(feat_f_2 + feat_10_2).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_f_4 = func.leaky_relu(self.cn5_f(self.conv5_f(feat_f_3 + feat_10_3).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)
        feat_f_5 = func.leaky_relu(self.cn6_f(self.conv6_f(feat_f_4 + feat_10_4).permute(0, 2, 3, 1)).permute(0, 3, 1, 2), negative_slope=0.2)

        out = func.sigmoid(self.conv_fin(feat_f_5))

        return out


class DownSampler(nn.Module):
    def __init__(self, nbands_20=6, nbands_60=2):
        super(DownSampler, self).__init__()
        self.depth_20 = nn.Conv2d(nbands_20, nbands_20, 41, bias=False, padding='same', padding_mode='reflect', groups=nbands_20)
        self.depth_60 = nn.Conv2d(nbands_60, nbands_60, 41, bias=False, padding='same', padding_mode='reflect', groups=nbands_60)

        self.depth_20.weight.data = mtf_kernel_to_torch(gen_mtf(2, 'S2-20', kernel_size=41))
        self.depth_60.weight.data = mtf_kernel_to_torch(gen_mtf(6, 'S2-60', kernel_size=41))

        self.depth_20.requires_grad = False
        self.depth_60.requires_grad = False

    def forward(self, output_20, output_60):

        output_20_lp = self.depth_20(output_20)
        output_60_lp = self.depth_60(output_60)

        output_20_lr = func.interpolate(output_20_lp, scale_factor=1/2, mode='nearest-exact')
        output_60_lr = func.interpolate(output_60_lp, scale_factor=1/6, mode='nearest-exact')

        return output_20_lr, output_60_lr

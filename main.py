import os

import numpy as np
import torch
import torchvision
import argparse
from osgeo import gdal
import skimage.transform
from matplotlib import pyplot as plt
from Metrics import metrics_rr
from Utils.band_selection import selection, synthesize

from torchvision.transforms import InterpolationMode
import CS

def main(args):
    flag_cut_bounds = 1
    dim_cut = 11
    th_values = 1
    L = 11
    ratio20 = 2
    ratio60 = 6

    bands20 = int(args.num_bands_20)
    bands60 = int(args.num_bands_60)
    device = torch.device("cuda" if torch.cuda.is_available() and args.gpu else "cpu")

    # I_PAN_path = "./Dataset/BANDS10m/BANDS_10m_S2A_MSIL1C_20221128T095341_N0400_R079_T33TVF_20221128T115118.tif"
    # I_MS_20_path = "./Dataset/BANDS20m/BANDS_20m_S2A_MSIL1C_20221128T095341_N0400_R079_T33TVF_20221128T115118.tif"
    # I_MS_60_path = "./Dataset/BANDS60m/BANDS_60m_S2A_MSIL1C_20221128T095341_N0400_R079_T33TVF_20221128T115118.tif"

    I_PAN_path = "/media/matteo/T7/Dataset_Ugliano/10/S2A_MSIL1C_20190922T025541_N0208_R032_T50QKK_20190922T060638_00048.tif"
    I_MS_20_path = "/media/matteo/T7/Dataset_Ugliano/20/S2A_MSIL1C_20190922T025541_N0208_R032_T50QKK_20190922T060638_00048.tif"
    I_MS_60_path = "/media/matteo/T7/Dataset_Ugliano/60/S2A_MSIL1C_20190922T025541_N0208_R032_T50QKK_20190922T060638_00048.tif"

    bands_10 = torch.tensor(gdal.Open(I_PAN_path).ReadAsArray().astype('float32'))[None,:,:,:].float()
    bands_20 = torch.tensor(gdal.Open(I_MS_20_path).ReadAsArray().astype('float32'))[None,:,:,:].float()
    bands_60 = torch.tensor(gdal.Open(I_MS_60_path).ReadAsArray().astype('float32'))[None,:,:,:].float()

    print(bands_10.min(), bands_10.max())
    print(bands_20.min(), bands_20.max())


    PAN_select_20, list_select_20 = selection(bands_10, bands_20, ratio20)
    PAN_select_60, list_select_60 = selection(bands_10, bands_60, ratio60)

    PAN_synthesize_20 = synthesize(bands_10, bands_20, ratio20)
    PAN_synthesize_60 = synthesize(bands_10, bands_60, ratio60)

    ## TO DO: use interp23tap (To be implemented!)
    bands_20_interp = torchvision.transforms.functional.resize(bands_20,
                                                        ((int(bands_20.shape[-2] * ratio20),
                                                          int(bands_20.shape[-1] * ratio20))),
                                                        interpolation=InterpolationMode.BICUBIC)

    print(bands_20_interp.min(), bands_20_interp.max())

    bands_60 = torchvision.transforms.functional.resize(bands_60,
                                                        ((int(bands_60.shape[-2] * ratio60),
                                                          int(bands_60.shape[-1] * ratio60))),
                                                        interpolation=InterpolationMode.BICUBIC)
    ## BDSD

    # bands_low_lr_lp, bands_low_lr, bands_high_lr = CS.prepro_BDSD(bands_20, PAN_select_20, list_select_20, ratio20, bands_20.shape[-1],  "S2-10", "S2-20")
    #
    # fused = []
    #
    # for i in range(bands20):
    #     gamma = CS.gamma_calculation_BDSD(bands_low_lr_lp[:, i, None, :, :], bands_low_lr[:, i, None, :, :], bands_high_lr[:, i, None, :, :], ratio20, bands_20.shape[-1])
    #     fused.append(CS.fuse_BDSD(bands_20[:, i, None, :, :], PAN_select_20[:, i, None, :, :], gamma, ratio20, bands_20.shape[-1]))
    #
    # fused = torch.cat(fused, dim=1)
    # print(fused.min(), fused.max())
    #
    # f = fused.detach().cpu().numpy()
    # b10 = bands_10.detach().cpu().numpy()
    # b20 = bands_20.detach().cpu().numpy()
    # plt.figure()
    # ax1 = plt.subplot(1,3,1)
    # plt.imshow(b10[0,0,:,:], cmap='gray')
    # plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    # plt.imshow(f[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    # plt.imshow(b20[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    #
    #
    # ## BDSD
    #
    # bands_low_lr_lp, bands_low_lr, bands_high_lr = CS.prepro_BDSD(bands_20, bands_10, list(range(4)), ratio20,
    #                                                               bands_20.shape[-1], "S2-10", "S2-20")
    # bands_high_lr_lr = synthesize(bands_high_lr, bands_low_lr_lp[:,:,1::ratio20,1::ratio20], ratio20)
    # fused = []
    #
    # for i in range(bands20):
    #     gamma = CS.gamma_calculation_BDSD(bands_low_lr_lp[:, i, None, :, :], bands_low_lr[:, i, None, :, :],
    #                                       bands_high_lr_lr[:, i, None, :, :], ratio20, bands_20.shape[-1])
    #     fused.append(CS.fuse_BDSD(bands_20[:, i, None, :, :], PAN_synthesize_20[:, i, None, :, :], gamma, ratio20,
    #                               bands_20.shape[-1]))
    #
    # fused = torch.cat(fused, dim=1)
    # print(fused.min(), fused.max())
    #
    # f = fused.detach().cpu().numpy()
    # b10 = bands_10.detach().cpu().numpy()
    # b20 = bands_20.detach().cpu().numpy()
    # plt.figure()
    # ax1 = plt.subplot(1, 3, 1)
    # plt.imshow(b10[0, 0, :, :], cmap='gray')
    # plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    # plt.imshow(f[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    # plt.imshow(b20[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])


    ## GS

    # fused = []
    # for i in range(bands20):
    #     fused.append(CS.GSA(bands_20[:, i, None, :, :], PAN_select_20[:, i, None, :, :], bands_20_interp[:, i, None, :, :], ratio20))
    # fused = torch.cat(fused, dim=1)
    # f = fused.detach().cpu().numpy()
    # b10 = bands_10.detach().cpu().numpy()
    # b20 = bands_20_interp.detach().cpu().numpy()
    # plt.figure()
    # ax1 = plt.subplot(1, 3, 1)
    # plt.imshow(b10[0, 0, :, :], cmap='gray')
    # plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    # plt.imshow(f[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    # plt.imshow(b20[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.show()
    # return fused

    # ## BT-H
    #
    # fused = []
    # for i in range(bands20):
    #     fused.append(CS.BT_H(bands_20_interp[:, i, None, :, :], PAN_select_20[:, i, None, :, :], ratio20))
    # fused = torch.cat(fused, dim=1)
    # f = fused.detach().cpu().numpy()
    # b10 = bands_10.detach().cpu().numpy()
    # b20 = bands_20_interp.detach().cpu().numpy()
    # plt.figure()
    # ax1 = plt.subplot(1, 3, 1)
    # plt.imshow(b10[0, 0, :, :], cmap='gray')
    # plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    # plt.imshow(f[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    # plt.imshow(b20[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    # plt.show()
    # return fused


    ## PRACS

    fused = []
    for i in range(bands20):
        fused.append(CS.PRACS(bands_20_interp[:, i, None, :, :].float(), PAN_select_20[:, i, None, :, :], ratio20))
    fused = torch.cat(fused, dim=1)
    f = fused.detach().cpu().numpy()
    b10 = bands_10.detach().cpu().numpy()
    b20 = bands_20_interp.detach().cpu().numpy()
    plt.figure()
    ax1 = plt.subplot(1, 3, 1)
    plt.imshow(b10[0, 0, :, :], cmap='gray')
    plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1)
    plt.imshow(f[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    plt.subplot(1, 3, 3, sharex=ax1, sharey=ax1)
    plt.imshow(b20[0, 0, :, :], cmap='gray', clim=[b20[0, 0, :, :].min(), b20[0, 0, :, :].max()])
    plt.show()
    return fused



if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    parse = argparse.ArgumentParser(description="Super Resolution ToolBox")
    # parse.add_argument("--folder", default=r"/nas/homes/antonio.mazza/FUSE",
    #                    help="Folder where the images are stored.")
    # parse.add_argument("--folder", default=r"/home/ugliano2023/DatasetSuperRes(FUSE)Torch/",
    #                    help="Folder where the images are stored.")
    # parse.add_argument("--folder", default="../DatasetSuperRes(FUSE)Torch/",
    #                    help="Folder where the images are stored.")
    # parse.add_argument("--out_folder", default=r"/home/amazza/SentinelSuperRes/Competitors/FUSE",
    #                    help='Folder where you want to save the output.')
    # parse.add_argument("--out_folder", default=r"/home/ugliano2023/FUSE-Training",
    #                    help='Folder where you want to save the output.')
    # parse.add_argument("--out_folder", default="../Competitors/FUSE",
    #                    help='Folder where you want to save the output.')
    parse.add_argument("--num_bands_20", default='6', help='The number of 20m bands')
    parse.add_argument("--num_bands_60", default='3', help='The number of 60m bands')
    parse.add_argument("--gpu", action="store_false", help="Use this flag if you want to use the GPU.")
    parse.add_argument("--cpu", action="store_true", help="Use this flag if you want to use the CPU.")
    parse.add_argument("--gpu_number", default='1', help='The number of the GPU you want to use.')
    args = parse.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_number)
    torch.set_num_threads(1)

    fused = main(args)
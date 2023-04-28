# -*- coding: utf-8 -*-
"""
@author: Pietro Paolo Ugliano
"""
import os

import numpy as np
import torch
import torchvision
import argparse
from osgeo import gdal
import skimage.transform

from Metrics import metrics_rr
from Utils.band_selection import selection, synthesize
from BDSD import BDSD as BDSD
from torchvision.transforms import InterpolationMode


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



    PAN_select_20, list_select_20 = selection(bands_10, bands_20, ratio20)
    PAN_select_60, list_select_60 = selection(bands_10, bands_60, ratio60)

    PAN_synthesize_20 = synthesize(bands_10, bands_20, ratio20)
    PAN_synthesize_60 = synthesize(bands_10, bands_60, ratio60)
    # list_select_synt_20 = Select.select(PAN_synthesize_20, I_MS_20, ratio20)
    # list_select_synt_60 = Select.select(PAN_synthesize_60, I_MS_60, ratio60)
    """
    bands_20 = skimage.transform.resize(bands_20, ((int(bands_20.shape[0] * ratio20), int(bands_20.shape[1] * ratio20))),
                                       order=3, preserve_range=True)
    bands_60 = skimage.transform.resize(bands_60, ((int(bands_60.shape[0] * ratio60), int(bands_60.shape[1] * ratio60))),
                                       order=3, preserve_range=True)
    """
    bands_20 = torchvision.transforms.functional.resize(bands_20,
                                        ((int(bands_20.shape[0] * ratio20), int(bands_20.shape[1] * ratio20))),
                                        interpolation = InterpolationMode.BICUBIC)
    bands_60 = torchvision.transforms.functional.resize(bands_60,
                                        ((int(bands_60.shape[0] * ratio60), int(bands_60.shape[1] * ratio60))),
                                        interpolation = InterpolationMode.BICUBIC)


    ## NumPy Conversion - TEMPORARY
    bands_10 = np.squeeze(bands_10.permute(0, 2, 3, 1).detach().cpu().numpy())
    bands_20 = np.squeeze(bands_20.permute(0, 2, 3, 1).detach().cpu().numpy())
    bands_60 = np.squeeze(bands_60.permute(0, 2, 3, 1).detach().cpu().numpy())

    PAN_select_20 = np.squeeze(PAN_select_20.permute(0, 2, 3, 1).detach().cpu().numpy())
    PAN_select_60 = np.squeeze(PAN_select_60.permute(0, 2, 3, 1).detach().cpu().numpy())

    PAN_synthesize_20 = np.squeeze(PAN_synthesize_20.permute(0, 2, 3, 1).detach().cpu().numpy())
    PAN_synthesize_60 = np.squeeze(PAN_synthesize_60.permute(0, 2, 3, 1).detach().cpu().numpy())

    # %% CS
    # %% IHS

    # I_Fus_IHS_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_IHS_20[:,:,i] = np.squeeze(IHS.IHS(I_MS_20_band, PAN_select_20[:,:,i]))

    # I_Fus_IHS_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_IHS_60[:,:,i] = np.squeeze(IHS.IHS(I_MS_60_band, PAN_select_60[:,:,i]))

    # I_Fus_IHS_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_IHS_synt_20[:,:,i] = np.squeeze(IHS.IHS(I_MS_20_band, PAN_synthesize_20[:,:,i]))

    # I_Fus_IHS_synt_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_IHS_synt_60[:,:,i] = np.squeeze(IHS.IHS(I_MS_60_band, PAN_synthesize_60[:,:,i]))

    # showImages20.showImages20(PAN_select_20, I_Fus_IHS_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_IHS_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_IHS_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_IHS_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_IHS_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_IHS_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_IHS_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_IHS_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_IHS_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_IHS Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_IHS_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_IHS Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_IHS_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_IHS Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_IHS_synt_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_IHS Synthesized 60m - False Color 60m')

    # %% PCA

    # I_Fus_PCA_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # I_MS_20_band = np.zeros((I_MS_20.shape[0],I_MS_20.shape[1],2))
    # for i in range(bands20):
    #     I_MS_20_band[:,:,0] = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1]))
    #     I_MS_20_band[:,:,1] = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1]))
    #     # I_MS_20_band[:,:,2] = np.reshape(I_MS_20[:,:,cord[1]], (I_MS_20[:,:,cord[1]].shape[0], I_MS_20[:,:,cord[1]].shape[1]))
    #     I_Fus_PCA_20[:,:,i] = (PCA.PCAPansharpening(I_MS_20_band, PAN_select_20[:,:,i]))[:,:,0]

    # I_Fus_PCA_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # I_MS_60_band = np.zeros((I_MS_60.shape[0],I_MS_60.shape[1],2))
    # for i in range(bands60):
    #     I_MS_60_band[:,:,0] = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1]))
    #     I_MS_60_band[:,:,1] = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1]))
    #     # I_MS_60_band[:,:,2] = np.reshape(I_MS_60[:,:,cord[1]], (I_MS_60[:,:,cord[1]].shape[0], I_MS_60[:,:,cord[1]].shape[1]))
    #     I_Fus_PCA_60[:,:,i] = (PCA.PCAPansharpening(I_MS_60_band, PAN_select_60[:,:,i]))[:,:,0]

    # I_Fus_PCA_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # I_MS_20_band = np.zeros((I_MS_20.shape[0],I_MS_20.shape[1],2))
    # for i in range(bands20):
    #     I_MS_20_band[:,:,0] = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1]))
    #     I_MS_20_band[:,:,1] = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1]))
    #     # I_MS_20_band[:,:,2] = np.reshape(I_MS_20[:,:,cord[1]], (I_MS_20[:,:,cord[1]].shape[0], I_MS_20[:,:,cord[1]].shape[1]))
    #     I_Fus_PCA_synt_20[:,:,i] = (PCA.PCAPansharpening(I_MS_20_band, PAN_synthesize_20[:,:,i]))[:,:,0]

    # I_Fus_PCA_synt_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # I_MS_60_band = np.zeros((I_MS_60.shape[0],I_MS_60.shape[1],2))
    # for i in range(bands60):
    #     I_MS_60_band[:,:,0] = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1]))
    #     I_MS_60_band[:,:,1] = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1]))
    #     # I_MS_60_band[:,:,2] = np.reshape(I_MS_60[:,:,cord[1]], (I_MS_60[:,:,cord[1]].shape[0], I_MS_60[:,:,cord[1]].shape[1]))
    #     I_Fus_PCA_synt_60[:,:,i] = (PCA.PCAPansharpening(I_MS_60_band, PAN_synthesize_60[:,:,i]))[:,:,0]

    # showImages20.showImages20(PAN_select_20, I_Fus_PCA_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_PCA_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_PCA_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_PCA_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_PCA_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PCA_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_PCA_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PCA_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PCA_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_PCA Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PCA_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_PCA Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PCA_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_IHS Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PCA_synt_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_IHS Synthesized 60m - False Color 60m')

    # %% Brovey

    # I_Fus_Brovey_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_Brovey_20[:,:,i] = np.squeeze(Brovey.Brovey(I_MS_20_band, PAN_select_20[:,:,i]))

    # I_Fus_Brovey_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_Brovey_60[:,:,i] = np.squeeze(Brovey.Brovey(I_MS_60_band, PAN_select_60[:,:,i]))

    # I_Fus_Brovey_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_Brovey_synt_20[:,:,i] = np.squeeze(Brovey.Brovey(I_MS_20_band, PAN_synthesize_20[:,:,i]))

    # I_Fus_Brovey_synt_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_Brovey_synt_60[:,:,i] = np.squeeze(Brovey.Brovey(I_MS_60_band, PAN_synthesize_60[:,:,i]))

    # showImages20.showImages20(PAN_select_20, I_Fus_Brovey_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_Brovey_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_Brovey_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_Brovey_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_Brovey_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PCA_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_Brovey_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PCA_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_Brovey_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_Brovey Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_Brovey_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_Brovey Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_Brovey_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_Brovey Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_Brovey_synt_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_Brovey Synthesized 60m - False Color 60m')

    # %% GS

    # I_Fus_GS_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_GS_20[:,:,i] = np.squeeze(GS.GS(I_MS_20_band, PAN_select_20[:,:,i]))

    # I_Fus_GS_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_GS_60[:,:,i] = np.squeeze(GS.GS(I_MS_60_band, PAN_select_60[:,:,i]))

    # I_Fus_GS_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_GS_synt_20[:,:,i] = np.squeeze(GS.GS(I_MS_20_band, PAN_synthesize_20[:,:,i]))

    # I_Fus_GS_synt_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_GS_synt_60[:,:,i] = np.squeeze(GS.GS(I_MS_60_band, PAN_synthesize_60[:,:,i]))

    # showImages20.showImages20(PAN_select_20, I_Fus_GS_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_GS_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_GS_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_GS_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_GS_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_GS_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_GS_synt_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_GS_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_GS_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_GS Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_GS_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_GS Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_GS_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_GS Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_GS_synt_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_GS Synthesized 60m - False Color 60m')

    # %% BDSD

    I_Fus_BDSD_20 = np.zeros((bands_20.shape[0], bands_20.shape[1], bands_20.shape[2]))
    for i in range(bands20):
        I_MS_20_band = np.reshape(bands_20[:, :, i], (bands_20[:, :, i].shape[0], bands_20[:, :, i].shape[1], 1))
        I_Fus_BDSD_20[:, :, i] = np.squeeze(
            BDSD.BDSD(I_MS_20_band, PAN_select_20[:, :, i], ratio20, bands_20.shape[1], "S2-20"))

    I_Fus_BDSD_20_torch = torch.from_numpy(I_Fus_BDSD_20[None, :, :, :]).to(device)
    I_Fus_BDSD_20_torch = I_Fus_BDSD_20_torch.permute(0,3,1,2)
    I_MS_20_torch = torch.from_numpy(bands_20[None, :, :, :]).to(device)
    I_MS_20_torch = I_MS_20_torch.permute(0,3,1,2)
    qindex = metrics_rr.Q(bands20, device, block_size=540)
    qindex = qindex(I_Fus_BDSD_20_torch, I_MS_20_torch)
    print("Q index: {}".format(qindex))

    I_Fus_BDSD_60 = np.zeros((bands_60.shape[0], bands_60.shape[1], bands_60.shape[2]))
    for i in range(bands60):
        I_MS_60_band = np.reshape(bands_60[:, :, i], (bands_60[:, :, i].shape[0], bands_60[:, :, i].shape[1], 1))
        I_Fus_BDSD_60[:, :, i] = np.squeeze(
            BDSD.BDSD(I_MS_60_band, PAN_select_60[:, :, i], ratio60, bands_60.shape[1], "S2-60"))

    I_Fus_BDSD_synt_20 = np.zeros((bands_20.shape[0], bands_20.shape[1], bands_20.shape[2]))
    for i in range(bands20):
        I_MS_20_band = np.reshape(bands_20[:, :, i], (bands_20[:, :, i].shape[0], bands_20[:, :, i].shape[1], 1))
        I_Fus_BDSD_synt_20[:, :, i] = np.squeeze(
            BDSD.BDSD(I_MS_20_band, PAN_synthesize_20[:, :, i], ratio20, bands_20.shape[1], "S2-20"))

    I_Fus_BDSD_synt_60 = np.zeros((bands_60.shape[0], bands_60.shape[1], bands_60.shape[2]))
    for i in range(bands60):
        I_MS_60_band = np.reshape(bands_60[:, :, i], (bands_60[:, :, i].shape[0], bands_60[:, :, i].shape[1], 1))
        I_Fus_BDSD_synt_60[:, :, i] = np.squeeze(
            BDSD.BDSD(I_MS_60_band, PAN_synthesize_60[:, :, i], ratio60, bands_60.shape[1], "S2-60"))

    # showImages20.showImages20(PAN_select_20, I_Fus_BDSD_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_BDSD_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_BDSD_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_BDSD_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_BDSD_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_BDSD_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_BDSD_synt_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_BDSD_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_BDSD_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L,
    #                                   'RGB 10m - Fus_BDSD Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_BDSD_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L,
    #                                   'RGB 10m - FUS_BDSD Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_BDSD_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L,
    #                                   'RGB 10m - Fus_BDSD Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_BDSD_synt_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L,
    #                                  'RGB 10m - FUS_BDSD Synthesized 60m - False Color 60m')

    # %% PRACS

    # I_Fus_PRACS_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_PRACS_20[:,:,i] = np.squeeze(PRACS.PRACS(I_MS_20_band, PAN_select_20[:,:,i], ratio20))

    # I_Fus_PRACS_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_PRACS_60[:,:,i] = np.squeeze(PRACS.PRACS(I_MS_60_band, PAN_select_60[:,:,i], ratio60))

    # I_Fus_PRACS_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_PRACS_synt_20[:,:,i] = np.squeeze(PRACS.PRACS(I_MS_20_band, PAN_synthesize_20[:,:,i], ratio20))

    # I_Fus_PRACS_synt_60 = np.zeros((I_MS_60.shape[0], I_MS_60.shape[1], I_MS_60.shape[2]))
    # for i in range(bands60):
    #     I_MS_60_band = np.reshape(I_MS_60[:,:,i], (I_MS_60[:,:,i].shape[0], I_MS_60[:,:,i].shape[1],1))
    #     I_Fus_PRACS_synt_60[:,:,i] = np.squeeze(PRACS.PRACS(I_MS_60_band, PAN_synthesize_60[:,:,i], ratio60))

    # showImages20.showImages20(PAN_select_20, I_Fus_PRACS_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_PRACS_20 - Band_MS_20', True, list_select_20)
    # showImages60.showImages60(PAN_select_60, I_Fus_PRACS_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_PRACS_60 - Band_MS_60', True, list_select_60)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_PRACS_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PRACS_20 - Band_MS_20')
    # showImages60.showImages60(PAN_synthesize_60, I_Fus_PRACS_synt_60, I_MS_60, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_PRACS_60 - Band_MS_60')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PRACS_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_PRACS Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PRACS_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_PRACS Selected 60m - False Color 60m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PRACS_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_PRACS Synthesized 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_PRACS_synt_60, I_MS_60, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - FUS_PRACS Synthesized 60m - False Color 60m')

    # %% MRA
    # %% MTF_GLP

    # I_Fus_MTF_GLP_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_20[:,:,i] = np.squeeze(MTF_GLP.MTF_GLP(I_MS_20_band, PAN_select_20[:,:,i], "S2-20", ratio20))

    # I_Fus_MTF_GLP_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_synt_20[:,:,i] = np.squeeze(MTF_GLP.MTF_GLP(I_MS_20_band, PAN_synthesize_20[:,:,i], "S2-20", ratio20))

    # showImages20.showImages20(PAN_select_20, I_Fus_MTF_GLP_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_MTF_GLP_20 - Band_MS_20', True, list_select_20)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_MTF_GLP_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_MTF_GLP_20 - Band_MS_20')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP Synthesized 20m - False Color 20m')

    # %% MTF_GLP_FS

    # I_Fus_MTF_GLP_FS_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_FS_20[:,:,i] = np.squeeze(MTF_GLP_FS.MTF_GLP_FS(I_MS_20_band, PAN_select_20[:,:,i], "S2-20", ratio20))

    # I_Fus_MTF_GLP_FS_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_FS_synt_20[:,:,i] = np.squeeze(MTF_GLP_FS.MTF_GLP_FS(I_MS_20_band, PAN_synthesize_20[:,:,i], "S2-20", ratio20))

    # showImages20.showImages20(PAN_select_20, I_Fus_MTF_GLP_FS_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_MTF_GLP_FS_20 - Band_MS_20', True, list_select_20)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_MTF_GLP_FS_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_MTF_GLP_FS_20 - Band_MS_20')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_FS_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_FS Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_FS_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_FS Synthesized 20m - False Color 20m')

    # %% MTF_GLP_HPM

    # I_Fus_MTF_GLP_HPM_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_20[:,:,i] = np.squeeze(MTF_GLP_HPM.MTF_GLP_HPM(I_MS_20_band, PAN_select_20[:,:,i], "S2-20", ratio20))

    # I_Fus_MTF_GLP_HPM_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_synt_20[:,:,i] = np.squeeze(MTF_GLP_HPM.MTF_GLP_HPM(I_MS_20_band, PAN_synthesize_20[:,:,i], "S2-20", ratio20))

    # showImages20.showImages20(PAN_select_20, I_Fus_MTF_GLP_HPM_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_MTF_GLP_HPM_20 - Band_MS_20', True, list_select_20)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_MTF_GLP_HPM_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_MTF_GLP_HPM_20 - Band_MS_20')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM Synthesized 20m - False Color 20m')

    # %% MTF_GLP_HPM_H

    # I_Fus_MTF_GLP_HPM_H_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_H_20[:,:,i] = np.squeeze(MTF_GLP_HPM_H.MTF_GLP_HPM_Haze_min(I_MS_20_band, PAN_select_20[:,:,i], "S2-20", ratio20, True))

    # I_Fus_MTF_GLP_HPM_H_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_H_synt_20[:,:,i] = np.squeeze(MTF_GLP_HPM_H.MTF_GLP_HPM_Haze_min(I_MS_20_band, PAN_synthesize_20[:,:,i], "S2-20", ratio20, True))

    # showImages20.showImages20(PAN_select_20, I_Fus_MTF_GLP_HPM_H_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_MTF_GLP_HPM_H_20 - Band_MS_20', True, list_select_20)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_MTF_GLP_HPM_H_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_MTF_GLP_HPM_H_20 - Band_MS_20')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_H_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM_H Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_H_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM_H Synthesized 20m - False Color 20m')

    # %% MTF_GLP_HPM_R

    # I_Fus_MTF_GLP_HPM_R_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_R_20[:,:,i] = np.squeeze(MTF_GLP_HPM_R.MTF_GLP_HPM_R(I_MS_20_band, PAN_select_20[:,:,i], "S2-20", ratio20))

    # I_Fus_MTF_GLP_HPM_R_synt_20 = np.zeros((I_MS_20.shape[0], I_MS_20.shape[1], I_MS_20.shape[2]))
    # for i in range(bands20):
    #     I_MS_20_band = np.reshape(I_MS_20[:,:,i], (I_MS_20[:,:,i].shape[0], I_MS_20[:,:,i].shape[1],1))
    #     I_Fus_MTF_GLP_HPM_R_synt_20[:,:,i] = np.squeeze(MTF_GLP_HPM_R.MTF_GLP_HPM_R(I_MS_20_band, PAN_synthesize_20[:,:,i], "S2-20", ratio20))

    # showImages20.showImages20(PAN_select_20, I_Fus_MTF_GLP_HPM_R_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_10 - Band_Fus_MTF_GLP_HPM_R_20 - Band_MS_20', True, list_select_20)
    # showImages20.showImages20(PAN_synthesize_20, I_Fus_MTF_GLP_HPM_R_synt_20, I_MS_20, flag_cut_bounds, dim_cut, th_values, L, 'Band_synt_10 - Band_Fus_MTF_GLP_HPM_R_20 - Band_MS_20')

    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_R_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM_R Selected 20m - False Color 20m')
    # showImage4_3imgs.showImage4_3imgs(I_PAN, I_Fus_MTF_GLP_HPM_R_synt_20, I_MS_20, flag_cut_bounds,dim_cut,th_values,L,'RGB 10m - Fus_MTF_GLP_HPM_R Synthesized 20m - False Color 20m')

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

    main(args)
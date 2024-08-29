import os
import torch
import numpy as np
import time

import gc
from recordclass import recordclass

import csv

from CS.PRACS import PRACS
from CS.Brovey import BT_H
from CS.BDSD import BDSD, BDSD_PC
from CS.GS import GS, GSA

from MRA.GLP import MTF_GLP, MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_H, MTF_GLP_HPM_R
from MRA.AWLP import AWLP
from MRA.MF import MF

from ModelBasedOptimization.SR_D import SR_D
from ModelBasedOptimization.TV import TV

from ModelBasedOptimization.MuSA import MuSA
from ModelBasedOptimization.SSSS import SSSS
from ModelBasedOptimization.S2Sharp import S2Sharp
from ModelBasedOptimization.SupReMe import SupReMe
from ATPRK.ATPRK import SEL_ATPRK, SYNTH_ATPRK

from DSen2.DSen2 import DSen2
from FUSE.FUSE import FUSE
from RFUSE.R_FUSE import R_FUSE
from S2_SSC_CNN.S2_SCC_CNN import S2_SSC_CNN
from SURE.SURE import SURE

from Metrics.evaluation import evaluation_rr, evaluation_fr

from Utils.dl_tools import generate_paths
from Utils.load_save_tools import open_mat
from Utils import load_save_tools as ut
from Utils import interpolator_tools as it
from Utils.pan_generation import selection, synthesize

def pan_method(method, exp_dict):

    exp_input = {}
    pan = torch.clone(exp_dict.pan.double())
    exp_input['ms'] = torch.clone(exp_dict.ms.double())
    exp_input['ms_lr'] = torch.clone(exp_dict.ms_lr.double())
    exp_input['ratio'] = exp_dict.ratio
    exp_input['sensor'] = exp_dict.sensor


    exp_input = recordclass('exp_input', exp_info.keys())(*exp_info.values())

    fused = []

    for i in range(pan.shape[1]):
        exp_input.pan = pan[:, i:i + 1, :, :]
        exp_input.sensor_pan = exp_dict.sensor_pan[i]
        fused.append(method(exp_input)[:, i:i+1, :, :])

    fused = torch.cat(fused, 1)

    return fused

def EXP(ordered_dict):
    return ordered_dict.ms

ratio = 2
pansharpening_algorithm_dict = {
                                'BDSD': BDSD, 'BDSD-PC': BDSD_PC, 'GS': GS, 'GSA': GSA, # Component substitution
                                'BT-H': BT_H, 'PRACS': PRACS,  # Component substitution
                                'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS,  # Multi-Resolution analysis
                                'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H,  # Multi-Resolution analysis
                                'MTF-GLP-HPM-R': MTF_GLP_HPM_R, 'MF': MF,  # Multi-Resolution analysis
                                'SR-D': SR_D, 'TV': TV,  # Model-Based Optimization
                  }

ad_hoc_algorithm_dict = {'EXP': EXP,  # Baseline
                        'MuSA': MuSA, 'SSSS': SSSS, 'S2Sharp': S2Sharp, 'SupReMe': SupReMe, # Model-Based Optimization
                        'SEL-ATPRK': SEL_ATPRK, 'SYNTH-ATPRK': SYNTH_ATPRK,  # ATPRK
                        'DSen2': DSen2, 'FUSE': FUSE, 'R-FUSE': R_FUSE, 'S2-SSC-CNN': S2_SSC_CNN, 'SURE': SURE # Deep Learning
                         }
fieldnames_rr = ['Method', 'ERGAS', 'SAM', 'Q2n', 'Elapsed_time']
fieldnames_fr = ['Method', 'D_lambda', 'Repro-ERGAS', 'Repro-SAM', 'D_rho', 'Elapsed_time']


if __name__ == '__main__':
    from Utils.dl_tools import open_config

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    for dataset in config.datasets:
        ds_paths = []
        for experiment_folder in config.experiment_folders:
            if experiment_folder == 'Reduced_Resolution':
                experiment_folder = os.path.join(experiment_folder, '20')
            ds_paths += generate_paths(config.ds_root, dataset, 'Test', experiment_folder)


        for i, path in enumerate(ds_paths):
            print(path)
            name = path.split(os.sep)[-1].split('.')[0]
            bands_10, bands_20, bands_60, gt = open_mat(path)

            if gt is None:
                experiment_type = 'FR'
            else:
                experiment_type = 'RR'

            save_assessment = os.path.join(config.save_assessment, dataset, experiment_type)
            save_root = os.path.join(config.save_root, dataset, name, experiment_type)

            if not os.path.exists(os.path.join(save_assessment, experiment_type, '20')):
                os.makedirs(os.path.join(save_assessment, experiment_type, '20'))

            if experiment_type == 'RR':
                if not os.path.exists(os.path.join(save_assessment, experiment_type, '20', name + '_RR.csv')):
                    with open(os.path.join(save_assessment, experiment_type, '20', name + '_RR.csv'), 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                        writer.writeheader()
                    f.close()

            else:
                if not os.path.exists(os.path.join(save_assessment, experiment_type, '20', name + '_FR.csv')):
                    with open(os.path.join(save_assessment, experiment_type, '20', name + '_FR.csv'), 'w', encoding='UTF8', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                        writer.writeheader()
                    f.close()


            exp_info = {'ratio': ratio}
            exp_info['bands_10'] = bands_10
            exp_info['bands_20'] = bands_20
            exp_info['bands_60'] = bands_60
            exp_info['ms_lr'] = bands_20
            exp_info['ms'] = it.ideal_interpolator(bands_20, ratio)
            exp_info['dataset'] = dataset
            exp_info['sensor'] = 'S2-20'
            exp_info['name'] = name
            exp_info['root'] = config.ds_root
            exp_info['img_number'] = i
            exp_info['pan'] = 0
            exp_info['selected_bands'] = 0
            exp_info['pan_generator'] = 0
            exp_info['sensor_pan'] = 0
            for algorithm in config.algorithms:
                metrics_rr = []
                metrics_fr = []
                exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())
                if algorithm in pansharpening_algorithm_dict.keys():
                    temp_name_algorithm = algorithm
                    start_time = time.time()
                    for pan_generation in config.pan_generation:
                        if pan_generation == 'selection':
                            gen = 'SEL-'
                            print('Running algorithm: ' + gen + temp_name_algorithm)
                            pan, selected_bands = selection(bands_10, bands_20, ratio, 'S2-10')
                            exp_input.pan = pan
                            exp_input.selected_bands = selected_bands
                            exp_input.pan_generator = pan_generation
                            exp_input.sensor_pan = ['S2-SEL-' + str(b.item()) for b in selected_bands]

                        else:
                            gen = 'SYNTH-'
                            print('Running algorithm: ' + gen + temp_name_algorithm)
                            pan, selected_bands = synthesize(bands_10, bands_20, ratio, 'S2-10')
                            exp_input.pan = pan
                            exp_input.pan_generator = pan_generation
                            exp_input.sensor_pan = ['S2-10-PAN'] * bands_20.shape[1]

                        algorithm = gen + temp_name_algorithm
                        method = pansharpening_algorithm_dict[temp_name_algorithm]
                        start_time = time.time()
                        fused = pan_method(method, exp_input)
                        end_time = time.time()
                        elapsed_time = end_time - start_time
                        print('Elapsed time for executing the algorithm: ' + str(elapsed_time))
                        with torch.no_grad():
                            if experiment_type == 'RR':
                                metrics_values_rr = list(
                                    evaluation_rr_20(fused, torch.clone(gt), ratio=exp_info['ratio']))
                                metrics_values_rr.insert(0, algorithm)
                                metrics_values_rr.append(elapsed_time)
                                metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                                print(metrics_values_rr_dict)
                                metrics_rr.append(metrics_values_rr_dict)
                                with open(os.path.join(save_assessment, experiment_type, '20', name + '_RR.csv'), 'a',
                                          encoding='UTF8', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                                    writer.writerow(metrics_values_rr_dict)
                            else:
                                metrics_values_fr = list(
                                    evaluation_fr_20(fused, torch.clone(bands_10), torch.clone(bands_20),
                                                     ratio=exp_info['ratio'], sensor=exp_info['sensor']))
                                metrics_values_fr.insert(0, algorithm)
                                metrics_values_fr.append(elapsed_time)
                                metrics_values_fr_dict = dict(zip(fieldnames_fr, metrics_values_fr))
                                print(metrics_values_fr_dict)
                                metrics_fr.append(metrics_values_fr_dict)
                                with open(os.path.join(save_assessment, '20', name + '_FR.csv'), 'a',
                                          encoding='UTF8', newline='') as f:
                                    writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                                    writer.writerow(metrics_values_fr_dict)
                        if config.save_results:
                            if not os.path.exists(save_root):
                                os.makedirs(save_root)
                            ut.save_mat(np.round(
                                np.clip(np.squeeze(fused.permute(0, 2, 3, 1).numpy(), axis=0), 0, 2 ** 16 - 1)).astype(
                                np.uint16), os.path.join(save_root, algorithm + '.mat'), 'S2_20')

                        del fused
                        torch.cuda.empty_cache()
                        gc.collect()

                else:
                    print('Running algorithm: ' + algorithm)
                    start_time = time.time()
                    method = ad_hoc_algorithm_dict[algorithm]
                    fused = method(exp_input)
                    end_time = time.time()
                    elapsed_time = end_time - start_time
                    print('Elapsed time for executing the algorithm: ' + str(elapsed_time))
                    with torch.no_grad():
                        if experiment_type == 'RR':
                            metrics_values_rr = list(evaluation_rr_20(fused, torch.clone(gt), ratio=exp_info['ratio']))
                            metrics_values_rr.insert(0, algorithm)
                            metrics_values_rr.append(elapsed_time)
                            metrics_values_rr_dict = dict(zip(fieldnames_rr, metrics_values_rr))
                            print(metrics_values_rr_dict)
                            metrics_rr.append(metrics_values_rr_dict)
                            with open(os.path.join(save_assessment, experiment_type, '20', name + '_RR.csv'), 'a', encoding='UTF8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames_rr)
                                writer.writerow(metrics_values_rr_dict)
                        else:
                            metrics_values_fr = list(evaluation_fr_20(fused, torch.clone(bands_10), torch.clone(bands_20), ratio=exp_info['ratio'], sensor=exp_info['sensor']))
                            metrics_values_fr.insert(0, algorithm)
                            metrics_values_fr.append(elapsed_time)
                            metrics_values_fr_dict = dict(zip(fieldnames_fr, metrics_values_fr))
                            print(metrics_values_fr_dict)
                            metrics_fr.append(metrics_values_fr_dict)
                            with open(os.path.join(save_assessment, experiment_type, '20', name + '_FR.csv'), 'a', encoding='UTF8', newline='') as f:
                                writer = csv.DictWriter(f, fieldnames=fieldnames_fr)
                                writer.writerow(metrics_values_fr_dict)
                    if config.save_results:
                        if not os.path.exists(save_root):
                            os.makedirs(save_root)
                        ut.save_mat(np.round(np.clip(np.squeeze(fused.permute(0, 2, 3, 1).numpy(), axis=0), 0, 2 ** 16 - 1)).astype(np.uint16), os.path.join(save_root, algorithm + '.mat'), 'S2_20')

                    del fused
                    torch.cuda.empty_cache()
                    gc.collect()





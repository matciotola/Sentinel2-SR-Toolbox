import os
import torch
import numpy as np

import copy
import gc

from Utils.pan_generation import selection, synthesize

from CS.PRACS import PRACS
from CS.Brovey import BT_H
from CS.BDSD import BDSD
from CS.GS import GS, GSA
from MRA.GLP import MTF_GLP, MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_H, MTF_GLP_HPM_R
from MRA.AWLP import AWLP
from DSen2.DSen2 import DSen2
from FUSE.FUSE import FUSE
from RFUSE.R_FUSE import R_FUSE
from Utils.dl_tools import generate_paths, open_tiff
from recordclass import recordclass
from Utils import load_save_tools as ut
from Utils.image_preprocessing import downsample_protocol

pansharpening_algorithm_dict = {'BDSD': BDSD, 'GS': GS, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS, # Component substitution
                                'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS, # Multi-Resolution analysis
                                'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H, # Multi-Resolution analysis
                                'MTF-GLP-HPM-R': MTF_GLP_HPM_R # Multi-Resolution analysis
                                }
deep_learning_algorithm_dict = {'DSen2': DSen2, 'FUSE': FUSE, 'R-FUSE': R_FUSE}

pan_generation_dict = {'selection': selection, 'synthesize': synthesize}

def pansharp_method(method, input_info):

    exp_input = copy.deepcopy(input_info)

    fused = []

    bands_high = torch.clone(exp_input.bands_high).double()
    exp_input.bands_low = exp_input.bands_low.double()
    exp_input.bands_low_lr = exp_input.bands_low_lr.double()

    for i in range(bands_high.shape[1]):
        exp_input.bands_high = bands_high[:, i:i + 1, :, :]
        if generator == 'selection':
            exp_input.ind = i
        fused.append(method(exp_input)[:, None, :, :, :])

    fused = torch.cat(fused, 1)

    fused_final = []

    for i in range(exp_input.bands_low.shape[1]):
        fused_final.append(fused[:, i, i:i + 1, :, :])

    fused_final = torch.cat(fused_final, 1)
    return fused_final

def main(args):



    return



if __name__ == '__main__':
    from Utils.dl_tools import open_config
    from Utils.interpolator_tools import ideal_interpolator

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    paths_10, paths_20, paths_60 = generate_paths(config.tiff_root, config.tiff_images)

    for experiment_type in config.experiment_type:
        for scale in config.bands_sr:

            if experiment_type == 'FR':
                geo_info = ut.extract_info(paths_10[0])
            else:
                if scale == '20':
                    geo_info = ut.extract_info(paths_20[0])
                else:
                    geo_info = ut.extract_info(paths_60[0])

            if scale == '20':
                exp_info = {'ratio': 2, 'mtf_low_name': 'S2-20'}

                bands_high = open_tiff(paths_10[0])
                bands_low_lr = open_tiff(paths_20[0])
                bands_low = ideal_interpolator(bands_low_lr, exp_info['ratio'])
                bands_intermediate = None

            else:
                exp_info = {'ratio': 6, 'mtf_low_name': 'S2-60'}

                bands_high = open_tiff(paths_10[0])
                bands_intermediate = open_tiff(paths_20[0])
                bands_low_lr = open_tiff(paths_60[0])

            if experiment_type == 'RR':
                bands_high, bands_intermediate, bands_low_lr = downsample_protocol(bands_high, bands_intermediate, bands_low_lr, exp_info['ratio'])

            bands_low = ideal_interpolator(bands_low_lr, exp_info['ratio'])

            exp_info['bands_low_lr'] = bands_low_lr
            exp_info['bands_low'] = bands_low
            exp_info['bands_intermediate'] = bands_intermediate

            for generator in config.pan_generation:
                gen = pan_generation_dict[generator]
                exp_info['bands_high'], exp_info['bands_selection'] = gen(bands_high, bands_low_lr, exp_info['ratio'])
                exp_info['ind'] = 0
                exp_info['pan_generator'] = generator

                if generator == 'selection':
                    exp_info['mtf_high_name'] = 'S2-10'
                    gen = 'SEL-'
                else:
                    exp_info['mtf_high_name'] = 'S2-10-PAN'
                    gen = 'SYNTH-'

                exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

                for algorithm in config.pansharpening_based_algorithms:

                    print('Running algorithm: ' + algorithm)

                    method = pansharpening_algorithm_dict[algorithm]
                    fused = pansharp_method(method, exp_input)

                    if config.save_results:
                        save_root = os.path.join(config.save_root, config.tiff_images[0], experiment_type, scale)
                        ut.save_tiff(np.squeeze(fused.numpy(), axis=0), save_root, gen + algorithm + '.tiff', geo_info)

                    del fused
                    gc.collect()

            exp_input.bands_high = bands_high
            for dl_algorithm in config.deep_learning_algorithms:
                print('Running algorithm: ' + dl_algorithm)
                method = deep_learning_algorithm_dict[dl_algorithm]
                fused = method(exp_input)
                if config.save_results:
                    save_root = os.path.join(config.save_root, config.tiff_images[0], experiment_type, scale)
                    ut.save_tiff(np.squeeze(fused.numpy(), axis=0), save_root, dl_algorithm + '.tiff', geo_info)

                del fused
                torch.cuda.empty_cache()
                gc.collect()


import os
import torch
import numpy as np

import copy

from band_selection import selection, synthesize

from CS import BDSD, GS, GSA, BT_H, PRACS
from MRA import AWLP, MTF_GLP, MTF_GLP_FS, MTF_GLP_HPM, MTF_GLP_HPM_H, MTF_GLP_HPM_R
from DSen2.DSen2 import DSen2
from common_dl_tools import generate_paths, open_tiff
from recordclass import recordclass
import utils as ut

pansharpening_algorithm_dict = {'BDSD': BDSD, 'GS': GS, 'GSA': GSA, 'BT-H': BT_H, 'PRACS': PRACS, 'AWLP': AWLP, 'MTF-GLP': MTF_GLP, 'MTF-GLP-FS': MTF_GLP_FS, 'MTF-GLP-HPM': MTF_GLP_HPM, 'MTF-GLP-HPM-H': MTF_GLP_HPM_H, 'MTF-GLP-HPM-R': MTF_GLP_HPM_R}
ad_hoc_algorithm_dict = {'DSen2': DSen2}

pan_generation_dict = {'selection': selection, 'synthesize': synthesize}

def pansharp_method(method, input_info):

    exp_input = copy.deepcopy(input_info)

    fused = []

    bands_high = torch.clone(exp_input.bands_high)

    for i in range(bands_high.shape[1]):
        exp_input.bands_high = bands_high[:, i:i + 1, :, :]
        if generator == 'selection':
            exp_input.ind = i
        fused.append(method(exp_input)[:, None, :, :, :])

    fused = torch.cat(fused, 1)

    fused_final = []
    for i in range(exp_input.bands_low.shape[1]):
        fused_final.append(fused[:, exp_input.bands_selection[i], i:i + 1, :, :])

    fused_final = torch.cat(fused_final, 1)
    return fused_final

def main(args):



    return



if __name__ == '__main__':
    from common_dl_tools import open_config
    from interpolator_tools import interp23tap_torch

    config_path = 'preambol.yaml'
    config = open_config(config_path)

    paths_10, paths_20, paths_60 = generate_paths(config.tiff_root, config.tiff_images)

    geo_info = ut.extract_info(paths_10[0])

    for experiment in config.bands_sr:
        if experiment == '20':
            exp_info = {'ratio': 2, 'mtf_low_name': 'S2-20'}

            bands_high = open_tiff(paths_10[0])
            bands_low_lr = open_tiff(paths_20[0])
            bands_low = interp23tap_torch(bands_low_lr, exp_info['ratio'])


            exp_info['bands_low_lr'] = bands_low_lr
            exp_info['bands_low'] = bands_low

        for generator in config.pan_generation:
                gen = pan_generation_dict[generator]
                exp_info['bands_high'], exp_info['bands_selection'] = gen(bands_high, bands_low_lr, exp_info['ratio'])
                exp_info['ind'] = 0
                exp_info['pan_generator'] = generator

                if generator == 'selection':
                    exp_info['mtf_high_name'] = 'S2-10'
                else:
                    exp_info['mtf_high_name'] = 'S2-10-PAN'

                exp_input = recordclass('exp_info', exp_info.keys())(*exp_info.values())

                for algorithm in config.pansharpening_based_algorithms:

                    print('Running algorithm: ' + algorithm)

                    method = pansharpening_algorithm_dict[algorithm]
                    fused = pansharp_method(method, exp_input)

                    if config.save_results:
                        save_root = os.path.join(config.save_root, config.tiff_images[0])
                        ut.save_tiff(np.squeeze(fused.numpy(), axis=0), save_root, algorithm + '_' + generator + '.tiff', geo_info)



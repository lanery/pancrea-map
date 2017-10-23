# -*- coding: utf-8 -*-
# @Author: Ryan Lane
# @Date:   2017-10-22 11:33:08
# @Last Modified by:   Ryan Lane
# @Last Modified time: 2017-10-23 08:12:56

from tqdm import tqdm

from .bench_utils import dict_product

from ..odemis_data import load_data
from ..stitch import get_keys, get_shape, find_matches

import logging


def benchtest_transform(img1, img2):
    """
    """
    ORB_arg_dict = {
        'downscale': [1.5, 2],
        'n_keypoints': [600, 800, 1000],
        'fast_threshold': 0.05
    }

    ransac_arg_dict = {
        'min_samples': [4],
        'residual_threshold': 1,
        'max_trials': [600]
    }

    ORB_kws_list = dict_product(ORB_arg_dict)

    ORB_kws_list = dict_product(ORB_arg_dict)
    ransac_kws_list = dict_product(ransac_arg_dict)

    for ORB_kws in tqdm(ORB_kws_list, desc='ORB', ascii=True):
        for ransac_kws in tqdm(ransac_kws_list, desc='RANSAC', ascii=True):

        # log.info('ORB parameters')
        # log.info(ORB_kws)

        model = find_matches(img1, img2, ORB_kws=ORB_kws)


if __name__ == '__main__':

    logfile = 'pancrea//workbench//matchfinder.log'
    logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
    log = logging.getLogger(__name__)

    dir_name = 'sample_data//rat-pancreas'
    data = load_data(dir_name=dir_name)

    FM_imgs, EM_imgs, x_positions, y_positions = data

    img1 = FM_imgs['tile_4-2']
    img2 = FM_imgs['tile_4-3']

    benchtest_matchfinder(img1, img2)


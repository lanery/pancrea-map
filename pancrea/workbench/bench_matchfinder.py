# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 20-10-2017 17:27:08
"""

from tqdm import tqdm

from .bench_utils import dict_product

from ..odemis_data import load_data
from ..stitch import get_keys, get_shape, find_matches

import logging


def benchtest_matchfinder(img1, img2):
    """
    """
    ORB_arg_dict = {
        'downscale': [1.5, 2],
        'n_keypoints': [600, 800, 1000],
        'fast_threshold': 0.05
    }

    ORB_kws_list = dict_product(ORB_arg_dict)

    for ORB_kws in tqdm(ORB_kws_list, desc='ORB', ascii=True):

        log.info('ORB parameters')
        log.info(ORB_kws)

        kps1, kps2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)


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

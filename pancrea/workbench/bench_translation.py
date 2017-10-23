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
from ..stitch import get_keys, get_shape, get_translations_robust

import logging
logfile = 'pancrea//workbench//translation.log'
logging.basicConfig(filename=logfile, filemode='w', level=logging.INFO)
log = logging.getLogger(__name__)


def benchtest_translation(data):
    """
    """

    ORB_arg_dict = {
        'downscale': 2,
        'n_keypoints': [800, 1000],
        'fast_threshold': 0.05
    }

    ransac_arg_dict = {
        'min_samples': [4],
        'residual_threshold': 1,
        'max_trials': [600]
    }

    ORB_kws_list = dict_product(ORB_arg_dict)
    ransac_kws_list = dict_product(ransac_arg_dict)

    for ORB_kws in tqdm(ORB_kws_list, desc='ORB', ascii=True):
        for ransac_kws in tqdm(ransac_kws_list, desc='RANSAC', ascii=True):

            log.info('ORB parameters')
            log.info(ORB_kws)
            log.info('RANSAC parameters')
            log.info(ransac_kws)

            translations = get_translations_robust(data,
                                                   ORB_kws=ORB_kws,
                                                   ransac_kws=ransac_kws)

            log.info('Translations')
            log.info(translations)



if __name__ == '__main__':

    filenames = ['sample_data/rat-pancreas//tile_4-2.h5',
                 'sample_data/rat-pancreas//tile_4-3.h5',
                 'sample_data/rat-pancreas//tile_4-4.h5',
                 'sample_data/rat-pancreas//tile_5-2.h5',
                 'sample_data/rat-pancreas//tile_5-3.h5',
                 'sample_data/rat-pancreas//tile_5-4.h5']

    data = load_data(filenames=filenames)

    bench_translations(data)

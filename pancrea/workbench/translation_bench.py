# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 20-10-2017 17:27:08
"""

from itertools import product

from .. odemis_data import load_data
from .. stitch import get_keys, get_shape, get_translations_robust

import logging
logfile = 'pancrea//workbench//translation.log'
logging.basicConfig(filename=logfile, level=logging.INFO)
logger = logging.getLogger(__name__)


def dict_product(arg_dict):
    """
    """
    # Wrap all non-list dict values (floats/ints) in a list
    # so that they can be handled by product
    for k, v in arg_dict.items():
        try:
            arg_dict[k] = [float(v)]
        except TypeError:
            pass

    # Create cartesian product from dict values
    dict_list = []
    for val in product(*arg_dict.values()):
        dict_list.append(dict(zip(arg_dict, val)))

    return dict_list


def bench_translations(data):
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

    for ORB_kws in ORB_kws_list:
        for ransac_kws in ransac_kws_list:

            logger.info('ORB parameters')
            logger.info(ORB_kws)
            logger.info('RANSAC parameters')
            logger.info(ransac_kws)

            translations = get_translations_robust(data, 
                                                   ORB_kws=ORB_kws,
                                                   ransac_kws=ransac_kws)

            logger.info('Translations')
            logger.info(translations)



if __name__ == '__main__':

    filenames = ['sample_data/rat-pancreas//tile_4-2.h5',
                 'sample_data/rat-pancreas//tile_4-3.h5',
                 'sample_data/rat-pancreas//tile_4-4.h5',
                 'sample_data/rat-pancreas//tile_5-2.h5',
                 'sample_data/rat-pancreas//tile_5-3.h5',
                 'sample_data/rat-pancreas//tile_5-4.h5']

    data = load_data(filenames=filenames)

    bench_translations(data)

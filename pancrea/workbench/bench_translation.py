# -*- coding: utf-8 -*-
# @Author: Ryan Lane
# @Date:   2017-10-22 11:33:08
# @Last Modified by:   rlane
# @Last Modified time: 23-10-2017 16:42:35

import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from .bench_utils import dict_product

from ..odemis_data import load_data
from ..stitch import get_keys, get_shape, estimate_translation


def benchtest_translation(img1, img2):
    """
    Runs benchmark tests for `stitch.estimate_translation`

    Parameters
    ----------
    img1 : ndarray
        Image data for first image
    img2 : ndarray
        Image data for seond image. Must have some overlap with img1 otherwise
        this test is pointless and the transformation estimation will either
        error or be bogus

    Returns
    -------
    out : pd.DataFrame
        Results of the benchmark test in the form of a pandas dataframe
        Results include parameters for the FFT cross-correlation algorithms
        along with the translations in x and y and runtime. If estimate transform fails in any way, x and y translation are passed on as 0.
    """
    # Parameters for FFT cross-correlation
    FFT_arg_dict = {
        'upsample_factor': [1, 2, 5, 10],
        'space': ['real']
    }

    # Get all possible combinations of parameter space
    FFT_kws_list = dict_product(FFT_arg_dict)

    # Instantiate dataframe
    df_out = pd.DataFrame()

    FFT_desc = "Cylcing through FFT Parameters"

    # Loop through FFT parameter set to apply estimate_translation
    for FFT_kws in tqdm(FFT_kws_list, desc=FFT_desc, ascii=True):

        params = FFT_kws
        start = time.clock()

        try:
            shift = estimate_translation(img1, img2, FFT_kws=FFT_kws)
            dX, dY = shift
        except ValueError:
            dX, dY = (0, 0)

        end = time.clock()
        results = {
            'dX': dX,
            'dY': dY,
            't': (end - start)
        }

        df_out = df_out.append({**params, **results}, ignore_index=True)

    # Reorder dataframe columns so results come last
    cols = list(params.keys()) + list(results.keys())
    df_out = df_out.loc[:, cols]

    return df_out


if __name__ == '__main__':

    # Load data
    dir_name = 'sample_data//rat-pancreas'
    data = load_data(dir_name=dir_name)
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    # Select random image pair
    img_pairs = []
    for row in keys:
        img_pairs.append(list(zip(row, row[1:])))
    img_pairs = np.array(img_pairs).reshape(shape[0] * (shape[1] - 1), 2)
    rand_img_pair = img_pairs[np.random.randint(12)]

    # Run bench test
    df_out = benchtest_translation(FM_imgs[rand_img_pair[0]],
                                 FM_imgs[rand_img_pair[1]])

    # Save results to log file
    logfile = 'pancrea//workbench//log_translation.log'
    if os.path.exists(logfile):
        df_out.to_csv(logfile, mode='a', header=False, index=False)
    else:
        df_out.to_csv(logfile, header=df_out.columns, index=False)

# -*- coding: utf-8 -*-
# @Author: Ryan Lane
# @Date:   2017-10-22 11:33:08
# @Last Modified by:   rlane
# @Last Modified time: 23-10-2017 15:35:10

import os
import numpy as np
import pandas as pd
import time
from tqdm import tqdm

from .bench_utils import dict_product

from ..odemis_data import load_data
from ..stitch import get_keys, get_shape, estimate_transform



def benchtest_transform(img1, img2):
    """
    Runs benchmark tests for `stitch.estimate_transform`

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
        Results of the benchmark test in the form of a pandas dataframe.
        Results include parameters for ORB and RANSAC algorithms along
        with the translations in x and y and runtime. If estimate transform
        fails in any way, x and y translation are passed on as 0.
    """
    # Parameters for ORB
    ORB_arg_dict = {
        'downscale': [1.5, 2, 2.25, 2.5, 2.75, 3],
        'n_keypoints': [600, 800, 1000],
        'fast_threshold': 0.05
    }

    # Parameters for RANSAC
    ransac_arg_dict = {
        'min_samples': [4],
        'residual_threshold': 1,
        'max_trials': [600, 800, 1000]
    }

    # Get all possible combinations of parameter space
    ORB_kws_list = dict_product(ORB_arg_dict)
    ransac_kws_list = dict_product(ransac_arg_dict)

    # Instantiate dataframe
    df_out = pd.DataFrame()

    ORB_desc = "Cycling through ORB Parameters"
    ransac_desc = "Cylcing through RANSAC Parameters"

    # Loop through ORB and RANSAC parameter sets to apply estimate_transform
    for ORB_kws in tqdm(ORB_kws_list, desc=ORB_desc, ascii=True):
        for ransac_kws in tqdm(ransac_kws_list, desc=ransac_desc, ascii=True):

            params = {**ORB_kws, **ransac_kws}
            start = time.clock()

            try:
                model = estimate_transform(
                    img1, img2, ORB_kws=ORB_kws, ransac_kws=ransac_kws)
                dX, dY = model.translation
            except Exception:
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
    df_out = benchtest_transform(FM_imgs[rand_img_pair[0]],
                                 FM_imgs[rand_img_pair[1]])

    # Save results to log file
    logfile = 'pancrea//workbench//log_transform.log'
    if os.path.exists(logfile):
        df_out.to_csv(logfile, mode='a', header=False, index=False)
    else:
        df_out.to_csv(logfile, header=df_out.columns, index=False)

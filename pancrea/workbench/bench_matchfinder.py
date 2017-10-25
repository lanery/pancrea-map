# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 25-10-2017 17:39:31
"""

import os
import re
from glob import glob
import time
import numpy as np
import pandas as pd
from tqdm import tqdm
from textwrap import dedent
import matplotlib.pyplot as plt

from skimage.feature import plot_matches
from skimage.measure import ransac
from skimage.transform import AffineTransform

from .bench_utils import dict_product

from ..odemis_data import load_data
from ..stitch import get_keys, get_shape, find_matches


def benchtest_matchfinder(img1, img2):
    """
    """
    ORB_arg_dict = {
        'downscale': [2, 2.5],
        'n_keypoints': [600, 800, 1000],
        'fast_threshold': 0.05}

    # Get all possible combinations of parameter space
    ORB_kws_list = dict_product(ORB_arg_dict)

    # Instantiate dataframe
    df_out = pd.DataFrame()

    ORB_desc = "Cycling through ORB Parameters"
    for ORB_kws in tqdm(ORB_kws_list, desc=ORB_desc, ascii=True):

        # Find matches and record time
        start = time.clock()
        kps1, kps2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)
        end = time.clock()
        results = {
            'N_matches': len(matches),
            'time': (end - start)}

        # Plot matches
        fig, ax = plt.subplots()
        plot_matches(ax, img1, img2, kps1, kps2, matches,
                     '#09BB62', '#00F67A')
        save_plot(N_matches=results['N_matches'],
                  time=round(results['time'], 1),
                  ORB_kws=ORB_kws)

        # Append results to dataframe
        df_out = df_out.append({**ORB_kws, **results}, ignore_index=True)

    # Reorder dataframe columns so results come last
    cols = list(ORB_kws.keys()) + list(results.keys())
    df_out = df_out.loc[:, cols]
    return df_out


def benchtest_matchfinder_ransac(img1, img2):
    """
    """
    ORB_arg_dict = {
        'downscale': [2],
        'n_keypoints': [2000, 1000],
        'fast_threshold': 0.05}

    ransac_arg_dict = {
        'min_samples': [2, 5],
        'residual_threshold': [2, 1, 0.5],
        'max_trials': [5000]}

    # Get Cartesian product of parameter space
    ORB_kws_list = dict_product(ORB_arg_dict)
    ransac_kws_list = dict_product(ransac_arg_dict)

    # Instantiate dataframe
    df_out = pd.DataFrame()

    # Loop through ORB and RANSAC parameter sets
    ORB_desc = "Cycling through ORB Parameters"
    ransac_desc = "Cylcing through RANSAC Parameters"
    for ORB_kws in tqdm(ORB_kws_list, desc=ORB_desc, ascii=True):
        for ransac_kws in tqdm(ransac_kws_list, desc=ransac_desc, ascii=True):

            params = {**ORB_kws, **ransac_kws}

            start = time.clock()
            kps1, kps2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)

            src = kps2[matches[:, 1]][:, ::-1]
            dst = kps1[matches[:, 0]][:, ::-1]

            try:
                _, inliers = ransac((src, dst), AffineTransform, **ransac_kws)
            except np.linalg.LinAlgError:
                inliers = np.zeros(len(matches), dtype=bool)

            end = time.clock()
            results = {
                'N_matches': inliers.sum(),
                'time': end - start}

            # Plot RANSAC-approved matches
            fig, ax = plt.subplots()
            plot_matches(ax, img1, img2, kps1, kps2, matches[inliers],
                         '#09BB62', '#00F67A')
            save_plot(N_matches=results['N_matches'],
                      time=round(results['time'], 1),
                      ORB_kws=ORB_kws, ransac_kws=ransac_kws)

            # Append results to dataframe
            df_out = df_out.append({**params, **results}, ignore_index=True)

    # Reorder dataframe columns so results come last
    cols = list(params.keys()) + list(results.keys())
    df_out = df_out.loc[:, cols]
    return df_out


def save_plot(N_matches, time, ORB_kws=None, ransac_kws=None):
    """
    """
    # Format text to add to plot
    head_text = """\
    Matches...{}
    Time......{}s
    """.format(N_matches, time)

    ORB_text = dedent("""\
    Downscale........{downscale}
    N keypoints......{n_keypoints}
    Fast Threshold...{fast_threshold}""").format(**ORB_kws)

    ransac_text = """\
    Min Samples......{min_samples}
    Res. Threshold...{residual_threshold}
    Max Trials.......{max_trials}""".format(**ransac_kws)

    # Add text to plot
    ax = plt.gca()
    ax.text(0.5, 1.01, head_text, transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='bottom',
            fontproperties='monospace')
    ax.text(0, 1, ORB_text, transform=ax.transAxes,
            horizontalalignment='left', verticalalignment='bottom',
            fontproperties='monospace')
    ax.text(1, 1, ransac_text, transform=ax.transAxes,
            horizontalalignment='right', verticalalignment='bottom',
            fontproperties='monospace')

    # Unfortunately complicated autonumbering scheme
    try:
        fns = glob('pancrea//workbench//match_finder//*')
        names = [os.path.basename(fn) for fn in fns]
        num_list = [re.findall('\d+', name) for name in names]
        nums = [int(num[-1]) for num in num_list if len(num) > 0]
        num = max(nums) + 1
    except Exception as e:
        print(e)
        num = 0

    # Save fig
    filename = ('pancrea//workbench//match_finder'
                '//benchtest_{}.png').format(num)
    plt.savefig(filename)


if __name__ == '__main__':

    # Load data
    filenames = glob(
        '../SECOM/*/orange_1200x900_overlap-50/dmonds*_[012]x*[012]y*')
    data = load_data(filenames=filenames)
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    # Select random image pair
    # img_pairs = []
    # for row in keys:
    #     img_pairs.append(list(zip(row, row[1:])))
    # img_pairs = np.array(img_pairs).reshape(shape[0] * (shape[1] - 1), 2)
    # rand_img_pair = img_pairs[np.random.randint(len(img_pairs))]

    # Run bench test
    df_out = benchtest_matchfinder_ransac(FM_imgs[keys[0, 0]],
                                          FM_imgs[keys[0, 1]])

    # Save results to log file
    logfile = 'pancrea//workbench//log_matchfinder.log'
    if os.path.exists(logfile):
        df_out.to_csv(logfile, mode='a', header=False, index=False)
    else:
        df_out.to_csv(logfile, header=df_out.columns, index=False)

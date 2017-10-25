# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   20-10-2017 10:39:06
@Last Modified by:   rlane
@Last Modified time: 25-10-2017 15:59:27
"""

import os
import re
from glob import glob
import numpy as np
import pandas as pd
from tqdm import tqdm
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
        'fast_threshold': 0.05
    }

    ORB_kws_list = dict_product(ORB_arg_dict)

    for ORB_kws in tqdm(ORB_kws_list, desc='ORB', ascii=True):

        kps1, kps2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)

        fig, ax = plt.subplots()
        plot_matches(ax, img1, img2, kps1, kps2, matches,
                     '#09BB62', '#00F67A')
        save_plot(len(matches), ORB_kws=ORB_kws)


def benchtest_matchfinder_ransac(img1, img2):
    """
    """
    ORB_arg_dict = {
        'downscale': [2, 2.5],
        'n_keypoints': [600, 800, 1000],
        'fast_threshold': 0.05
    }

    ransac_arg_dict = {
        'min_samples': [4],
        'residual_threshold': 1,
        'max_trials': [1000]
    }

    ORB_kws_list = dict_product(ORB_arg_dict)
    ransac_kws_list = dict_product(ransac_arg_dict)

    ORB_desc = "Cycling through ORB Parameters"
    ransac_desc = "Cylcing through RANSAC Parameters"

    for ORB_kws in tqdm(ORB_kws_list, desc=ORB_desc, ascii=True):
        for ransac_kws in tqdm(ransac_kws_list, desc=ransac_desc, ascii=True):

            kps1, kps2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)

            src = kps2[matches[:, 1]][:, ::-1]
            dst = kps1[matches[:, 0]][:, ::-1]

            _, inliers = ransac((src, dst), AffineTransform, **ransac_kws)

            fig, ax = plt.subplots()
            plot_matches(ax, img1, img2, kps1, kps2, matches[inliers],
                         '#09BB62', '#00F67A')
            save_plot(inliers.sum(), ORB_kws=ORB_kws, ransac_kws=ransac_kws)


def save_plot(N_matches, ORB_kws=None, ransac_kws=None):
    """
    """
    # Format text to add to plot
    N_matches_text = "{} Matches".format(N_matches)

    ORB_text = """\
    Downscale........{downscale}
    N keypoints......{n_keypoints}
    Fast Threshold...{fast_threshold}""".format(**ORB_kws)

    ransac_text = """\
    Min Samples..........{min_samples}
    Residual Threshold...{residual_threshold}
    Max Trials...........{max_trials}""".format(**ransac_kws)

    # Add text to plot
    ax = plt.gca()
    ax.text(0.5, 1.1, N_matches_text, transform=ax.transAxes,
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
                '//benchtest_{}.svg').format(num)
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
    benchtest_matchfinder_ransac(FM_imgs[keys[0, 0]],
                                 FM_imgs[keys[0, 1]])

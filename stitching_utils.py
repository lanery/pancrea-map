# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   28-09-2017 12:24:34
@Last Modified by:   rlane
@Last Modified time: 06-10-2017 16:32:29
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.external import tifffile
from skimage.feature import plot_matches

import stitching


#-------------------+
# Utility Functions |
#-------------------+
def load_tiff(tiff_file):
    """
    Utility function to load multi-page tiff files

    Parameters
    ----------
    tiff_file : file
        Multi-page ome tiff file rendered by Odemis
    """
    ome_dict = {}
    with tifffile.TiffFile(tiff_file) as tif:
        for p, page in enumerate(tif):
            if page.tags['page_name'].value == b'Filtered colour 1':
                ome_dict['FM'] = page
            elif page.tags['page_name'].value == b'Secondary electrons':
                ome_dict['EM'] = page
            else:
                ome_dict['?M'] = page
    return ome_dict


def compare(images, **kwargs):
    """
    Utility function to display images side-by-side

    Parameters
    ----------
    images : dict
        Images to display with keys as labels
    """
    fig, axes = plt.subplots(1, len(images), **kwargs)

    if not isinstance(images, dict):
        images = {k: v for k, v in zip(range(len(images)), images)}

    for n, (label, img) in enumerate(images.items()):
        axes[n].imshow(img)
        axes[n].set_title(label)
        axes[n].axis('off')

    fig.tight_layout()


def display_matches(img1, img2, remove_outliers=True,
                    ransac_kws=None, ORB_kws=None):
    """
    """
    ransac_kws = {} if ransac_kws is None else ransac_kws
    ORB_kws = {} if ORB_kws is None else ORB_kws

    if remove_outliers:
        _, kps1, kps2, matches, inliers = stitching._estimate_transform(
            img1, img2, return_data=True, ORB_kws=ORB_kws,
            **ransac_kws)

        fig, ax = plt.subplots()
        plot_matches(ax, img1, img2, kps1, kps2,
                     matches[inliers], '#09BB62', '#00F67A')
    
    else:
        kps1, kps2, matches = _find_matches(img1, img2, ORB_kws=ORB_kws)

        fig, ax = plt.subplots()
        plot_matches(ax, img1, img2, kps1, kps2,
                     matches, '#09BB62', '#00F67A')

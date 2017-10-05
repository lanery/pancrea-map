# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   28-09-2017 12:24:34
@Last Modified by:   rlane
@Last Modified time: 05-10-2017 18:50:19
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.external import tifffile
from skimage.measure import label


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
        _, kps1, kps2, matches, inliers = estimate_transform(
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


def generate_costs(diff_image, mask, vertical=True, gradient_cutoff=2.):
    """
    Ensures equal-cost paths from edges to region of interest.
    
    Parameters
    ----------
    diff_image : (M, N) ndarray of floats
        Difference of two overlapping images.
    mask : (M, N) ndarray of bools
        Mask representing the region of interest in ``diff_image``.
    vertical : bool
        Control operation orientation.
    gradient_cutoff : float
        Controls how far out of parallel lines can be to edges before
        correction is terminated. The default (2.) is good for most cases.
        
    Returns
    -------
    costs_arr : (M, N) ndarray of floats
        Adjusted costs array, ready for use.
    """
    if vertical is not True:
        return tweak_costs(diff_image.T, mask.T, vertical=vertical,
                           gradient_cutoff=gradient_cutoff).T
    
    # Start with a high-cost array of 1's
    costs_arr = np.ones_like(diff_image)
    
    # Obtain extent of overlap
    row, col = mask.nonzero()
    cmin = col.min()
    cmax = col.max()

    # Label discrete regions
    cslice = slice(cmin, cmax + 1)
    labels = label(mask[:, cslice])
    
    # Find distance from edge to region
    upper = (labels == 0).sum(axis=0)
    lower = (labels == 2).sum(axis=0)
    
    # Reject areas of high change
    ugood = np.abs(np.gradient(upper)) < gradient_cutoff
    lgood = np.abs(np.gradient(lower)) < gradient_cutoff
    
    # Give areas slightly farther from edge a cost break
    costs_upper = np.ones_like(upper, dtype=np.float64)
    costs_lower = np.ones_like(lower, dtype=np.float64)
    costs_upper[ugood] = upper.min() / np.maximum(upper[ugood], 1)
    costs_lower[lgood] = lower.min() / np.maximum(lower[lgood], 1)
    
    # Expand from 1d back to 2d
    vdist = mask.shape[0]
    costs_upper = costs_upper[np.newaxis, :].repeat(vdist, axis=0)
    costs_lower = costs_lower[np.newaxis, :].repeat(vdist, axis=0)
    
    # Place these in output array
    costs_arr[:, cslice] = costs_upper * (labels == 0)
    costs_arr[:, cslice] +=  costs_lower * (labels == 2)
    
    # Finally, place the difference image
    costs_arr[mask] = diff_image[mask]
    
    return costs_arr

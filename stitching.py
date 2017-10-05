# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   28-09-2017 12:22:52
@Last Modified by:   rlane
@Last Modified time: 05-10-2017 18:49:12
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import ORB, match_descriptors, plot_matches
from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.transform import rescale, warp
from skimage.measure import ransac


#-------------------+
# Feature Detection |
#-------------------+
def _detect_features(img, **ORB_kws):
    """

    # TODO: For speed improvement...
        Clip image such that feature detection is only run on overlapping
        regions
    """
    default_ORB_kws = dict(downscale=1.2, n_scales=8, n_keypoints=800,
                           fast_n=9, fast_threshold=0.05, harris_k=0.04)
    ORB_kws = {**default_ORB_kws, **ORB_kws}

    orb = ORB(**ORB_kws)

    orb.detect_and_extract(img.astype(float))
    kps = orb.keypoints
    dps = orb.descriptors

    return kps, dps


def _find_matches(img1, img2, **ORB_kws):
    """
    """
    kps1, dps1 = _detect_features(img1, **ORB_kws)
    kps2, dps2 = _detect_features(img2, **ORB_kws)

    matches = match_descriptors(dps1, dps2, cross_check=True)
    return kps1, kps2, matches


#------------------------------------------+
# Transform Estimation and Outlier Removal |
#------------------------------------------+
def estimate_transform(img1, img2, return_data=False, ORB_kws=None,
                        **ransac_kws):
    """
    """
    ORB_kws = {} if ORB_kws is None else ORB_kws
    kps1, kps2, matches = _find_matches(img1, img2, **ORB_kws)

    src = kps1[matches[:, 0]][:, ::-1]
    dst = kps2[matches[:, 1]][:, ::-1]

    default_ransac_kws = dict(min_samples=4, residual_threshold=1,
                              max_trials=300)
    ransac_kws = {**default_ransac_kws, **ransac_kws}

    model_robust, inliers = ransac((src, dst), ProjectiveTransform, 
                                   **ransac_kws)
    if return_data:
        return model_robust, kps1, kps2, matches, inliers

    else:
        return model_robust


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


#----------------------------+
# Apply Estimated Transforms |
#----------------------------+
def _apply_transform(img1, img2, model_robust):
    """
    """
    r, c = img1.shape[:2]
    corners = np.array([[0, 0],
                        [0, r],
                        [c, 0],
                        [c, r]])

    warped_corners = model_robust(corners)

    all_corners = np.vstack((warped_corners, corners))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)
    output_shape = (corner_max - corner_min)

    output_shape = np.ceil(output_shape[::-1]).astype(int)

    offset = SimilarityTransform(translation=-corner_min)

    img1_warped = warp(img1, offset.inverse, order=3,
                       output_shape=output_shape, cval=-1)

    img1_mask = (img1_warped != -1)
    img1_warped[~img1_mask] = 0

    
    transform = (model_robust + offset).inverse

    img2_warped = warp(img2, transform, order=3,
                       output_shape=output_shape, cval=-1)

    img2_mask = (img2_warped != -1)
    img2_warped[~img2_mask] = 0

    return img1_warped, img2_warped

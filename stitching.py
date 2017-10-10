# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   28-09-2017 12:22:52
@Last Modified by:   rlane
@Last Modified time: 10-10-2017 11:11:51
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import ORB, match_descriptors
from skimage.transform import ProjectiveTransform, SimilarityTransform
from skimage.transform import rescale, warp
from skimage.measure import ransac, label
from skimage.graph import route_through_array
from skimage.color import gray2rgb

import stitching_utils


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


def _find_matches(anchor, joiner, **ORB_kws):
    """
    """
    kps_anchor, dps_anchor = _detect_features(anchor, **ORB_kws)
    kps_joiner, dps_joiner = _detect_features(joiner, **ORB_kws)

    matches = match_descriptors(dps_anchor, dps_joiner, cross_check=True)
    return kps_anchor, kps_joiner, matches


#------------------------------------------+
# Transform Estimation and Outlier Removal |
#------------------------------------------+
def _estimate_transform(anchor, joiner, return_data=False, ORB_kws=None,
                        **ransac_kws):
    """
    """
    ORB_kws = {} if ORB_kws is None else ORB_kws
    kps_anchor, kps_joiner, matches = _find_matches(anchor, joiner, **ORB_kws)

    src = kps_joiner[matches[:, 1]][:, ::-1]
    dst = kps_anchor[matches[:, 0]][:, ::-1]

    default_ransac_kws = dict(min_samples=4, residual_threshold=1,
                              max_trials=300)
    ransac_kws = {**default_ransac_kws, **ransac_kws}

    model_robust, inliers = ransac((src, dst), ProjectiveTransform, 
                                   **ransac_kws)
    if return_data:
        return model_robust, kps_anchor, kps_joiner, matches, inliers

    else:
        return model_robust


#----------------------------+
# Apply Estimated Transforms |
#----------------------------+
def _apply_transform(anchor, joiner, model_robust):
    """
    """
    r, c = anchor.shape[:2]
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
    anchor_warped = warp(anchor, offset.inverse, order=3,
                         output_shape=output_shape, cval=-1)

    anchor_mask = (anchor_warped != -1)
    anchor_warped[~anchor_mask] = 0

    transform = (model_robust + offset).inverse
    joiner_warped = warp(joiner, transform, order=3,
                         output_shape=output_shape, cval=-1)

    joiner_mask = (joiner_warped != -1)
    joiner_warped[~joiner_mask] = 0

    # Define seed points
    ymax = output_shape[1] - 1
    xmax = output_shape[0] - 1

    mask_pts = [[0, ymax // 2],
                [xmax, ymax // 2]]

    return anchor_warped, joiner_warped, anchor_mask, joiner_mask, mask_pts


#----------------+
# Generate Costs |
#----------------+
def _generate_costs(diff_image, mask, vertical=True, gradient_cutoff=2.):
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


#----------------------------+
# Find the Minimum Cost Path |
#----------------------------+
def _stitch(anchor, joiner, model_robust):
    """
    """
    anchor_warped, joiner_warped, anchor_mask, joiner_mask, mask_pts = (
        _apply_transform(anchor, joiner, model_robust))

    costs = _generate_costs(np.abs(joiner_warped - anchor_warped),
                            joiner_mask & anchor_mask)
    costs[0, :] = 0
    costs[-1, :] = 0

    pts, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                 fully_connected=True)
    pts = np.array(pts)

    anchor_mask = np.zeros_like(anchor_warped, dtype=np.uint8)
    anchor_mask[pts[:, 0], pts[:, 1]] = 1
    anchor_mask = (label(anchor_mask, connectivity=1, background=-1) == 1)

    joiner_mask = ~(anchor_mask).astype(bool)

    stitched = np.where(anchor_mask, anchor_warped, joiner_warped)

    return stitched

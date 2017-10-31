# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 31-10-2017 11:16:59
"""

import os
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from skimage.feature import ORB, match_descriptors
from skimage.feature import register_translation
from skimage.transform import SimilarityTransform, AffineTransform, warp
from skimage.graph import route_through_array
from skimage.measure import label, ransac

from .odemis_data import load_data


def get_shape(data):
    """
    Get overall shape of the image array (y rows of images by x columns)
    """
    FM_imgs, EM_imgs, x_positions, y_positions = data

    xs = np.array(list(x_positions.values()))
    ys = np.array(list(y_positions.values()))

    cols = np.unique(np.round(xs, decimals=-1)).size
    rows = np.unique(np.round(ys, decimals=-1)).size

    shape = (rows, cols)
    return shape


def get_keys(data):
    """
    Get keys of image array (basenames of image files)
    """
    import pandas as pd

    FM_imgs, EM_imgs, x_positions, y_positions = data
    shape = get_shape(data)

    df = pd.DataFrame([x_positions, y_positions]).T
    df = df.sort_values([1, 0], ascending=[False, True])
    keys = df.index.values.reshape(shape)

    return keys


def detect_features(img, ORB_kws=None):
    """
    Detect features using ORB
    Wrapper for `skimage.feature.ORB`

    Parameters
    ----------

    Returns
    -------
    """
    ORB_kws = {} if ORB_kws is None else ORB_kws
    default_ORB_kws = {
        'downscale': 2,
        'n_keypoints': 2000,
        'fast_threshold': 0.05}

    ORB_kws = {**default_ORB_kws, **ORB_kws}

    orb = ORB(**ORB_kws)

    orb.detect_and_extract(img.astype(float))
    kps = orb.keypoints
    dps = orb.descriptors

    return kps, dps


def find_matches(img1, img2, crop_kws=None, ORB_kws=None):
    """
    Find matches between images
    Wrapper for `skimage.feature.match_descriptors`

    Parameters
    ----------

    Returns
    -------
    """
    if crop_kws is None:
        kps_img1, dps_img1 = detect_features(img1, ORB_kws=ORB_kws)
        kps_img2, dps_img2 = detect_features(img2, ORB_kws=ORB_kws)

    else:
        try:
            m, n = img1.shape
            o1 = 1 / (1 - (crop_kws['overlap'] / 100))
            o2 = 1 / (crop_kws['overlap'] / 100)

            if crop_kws['direction'] == 'horizontal':
                img1 = img1[:, int(n/o1):]
                img2 = img2[:, :int(n/o2)]

                kps_img1, dps_img1 = detect_features(img1, ORB_kws=ORB_kws)
                kps_img2, dps_img2 = detect_features(img2, ORB_kws=ORB_kws)

                kps_img1[:, 1] += int(n / o1)

            else:  # assume images are stacked vertically
                img1 = img1[int(m/o1):, :]
                img2 = img2[:int(m/o2), :]

                kps_img1, dps_img1 = detect_features(img1, ORB_kws=ORB_kws)
                kps_img2, dps_img2 = detect_features(img2, ORB_kws=ORB_kws)

                kps_img1[:, 0] += int(m / o1)

        except KeyError as exc:
            msg = "`crop_kws` must contain {}.".format(exc)
            raise KeyError(msg)

    matches = match_descriptors(dps_img1, dps_img2, cross_check=True)
    return kps_img1, kps_img2, matches


def estimate_transform(img1, img2, crop_kws=None,
                       ORB_kws=None, ransac_kws=None):
    """
    Estimate Affine transformation between two images
    Wrapper for `skimage.measure.ransac` assuming AffineTransform

    Parameters
    ----------

    Returns
    -------
    """
    kps_img1, kps_img2, matches = find_matches(
        img1, img2, crop_kws=crop_kws, ORB_kws=ORB_kws)

    src = kps_img2[matches[:, 1]][:, ::-1]
    dst = kps_img1[matches[:, 0]][:, ::-1]

    ransac_kws = {} if ransac_kws is None else ransac_kws
    default_ransac_kws = {
        'min_samples': 5,
        'residual_threshold': 10,
        'max_trials': 5000}

    ransac_kws = {**default_ransac_kws, **ransac_kws}

    model, inliers = ransac((src, dst), AffineTransform, **ransac_kws)
    return model


def estimate_translation(img1, img2, FFT_kws=None):
    """
    Estimate lateral translation between two images
    Wrapper for `skimage.feature.register_translation`

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.register_translation
    """

    FFT_kws = {} if FFT_kws is None else FFT_kws
    default_FFT_kws = {
        'upsample_factor': 1,
        'space': 'real'
    }
    FFT_kws = {**default_FFT_kws, **FFT_kws}

    shifts, error, phase_difference = register_translation(
        img1, img2, **FFT_kws)

    return shifts
    

def get_translations(data, method='robust', crop_kws=None, FFT_kws=None,
                     ORB_kws=None, ransac_kws=None):
    """

    Parameters
    ----------

    Returns
    -------
    """
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    if crop_kws is not None:
        crop_kws['direction'] = 'horizontal'

    h_shifts = []
    for row in tqdm(keys, ascii=True):
        h_shifts.append(np.zeros(2))
        for k1, k2 in tqdm(zip(row, row[1:]), ascii=True):

            if method == 'robust':
                model = estimate_transform(FM_imgs[k1], FM_imgs[k2],
                                           crop_kws=crop_kws,
                                           ORB_kws=ORB_kws,
                                           ransac_kws=ransac_kws)
                shift = model.translation
                print(shift)
            else:
                shift = estimate_translation(FM_imgs[k1], FM_imgs[k2],
                                             FFT_kws=FFT_kws)

            h_shifts.append(shift)

    if crop_kws is not None:
        crop_kws['direction'] = 'vertical'

    v_shifts = []
    for col in tqdm(keys.T, ascii=True):
        v_shifts.append(np.zeros(2))
        for k1, k2 in tqdm(zip(col, col[1:]), ascii=True):

            if method == 'robust':
                model = estimate_transform(FM_imgs[k1], FM_imgs[k2],
                                           crop_kws=crop_kws,
                                           ORB_kws=ORB_kws,
                                           ransac_kws=ransac_kws)
                shift = model.translation
                print(shift)
            else:
                shift = estimate_translation(FM_imgs[k1], FM_imgs[k2],
                                             FFT_kws=FFT_kws)

            v_shifts.append(shift)

    h_shifts = np.array(h_shifts).reshape(shape + (2,))
    v_shifts = np.array(v_shifts).reshape(shape[::-1] + (2,))

    cum_h_shifts = np.cumsum(h_shifts, axis=1)
    cum_v_shifts = np.cumsum(v_shifts, axis=1)

    translations = cum_h_shifts + np.swapaxes(cum_v_shifts, 0, 1)
    translations = translations.reshape(np.product(shape), 2)
    translations[:, 1] = translations[:, 1] - translations[:, 1].min()
    translations = translations.astype(np.int64)

    return translations


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
    costs_arr[:, cslice] += costs_lower * (labels == 2)

    # Finally, place the difference image
    costs_arr[mask] = diff_image[mask]

    return costs_arr


def warp_images(data, translations):
    """

    Parameters
    ----------

    Returns
    -------
    """
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    # translations = get_translations(data)

    transforms = {}
    for k, t in zip(keys.flatten(), translations):
        transforms[k] = AffineTransform(translation=t)

    H_px, W_px = FM_imgs[keys.flatten()[0]].shape

    output_shape = np.array(
        [translations[:, 1].max() - translations[:, 1].min() + H_px,
         translations[:, 0].max() - translations[:, 0].min() + W_px])

    warpeds = {}
    masks = {}

    for i, k in enumerate(keys.flatten()):

        warped = warp(FM_imgs[k], transforms[k].inverse, order=3,
                      output_shape=output_shape, cval=-1)

        mask = (warped != -1)
        warped[~mask] = 0

        warpeds[k] = warped
        masks[k] = mask

    return warpeds, masks


def get_puzzle_pieces(keys, shape, warpeds, masks):
    """
    """

    h_mcp_masks = []
    v_mcp_masks = []

    output_shape = np.array(list(warpeds.values())[0].shape)
    xmax, ymax = output_shape - 1
    Nx, Ny = shape[::-1]

    for row in keys:
        for i, (k1, k2) in enumerate(zip(row, row[1:])):

            costs = generate_costs(np.abs(warpeds[k2] - warpeds[k1]),
                                   masks[k2] & masks[k1])
            costs[0, :] = 0
            costs[-1, :] = 0

            mask_pts = [[0, ymax * (i+1) // Nx],
                        [xmax, ymax * (i+1) // Nx + 1]]

            mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                         fully_connected=True)
            mcp = np.array(mcp)

            mcp_mask = np.zeros(output_shape, dtype=np.uint8)
            mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
            mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)

            h_mcp_masks.append(mcp_mask)

    for col in keys.T:
        for i, (k1, k2) in enumerate(zip(col, col[1:])):

            costs = generate_costs(np.abs(warpeds[k2] - warpeds[k1]),
                                   masks[k2] & masks[k1])
            costs[:, 0] = 0
            costs[:, -1] = 0

            mask_pts = [[xmax * (i+1) // Ny, 0],
                        [xmax * (i+1) // Ny, ymax]]

            mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                         fully_connected=True)
            mcp = np.array(mcp)

            mcp_mask = np.zeros(output_shape, dtype=np.uint8)
            mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
            mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)

            v_mcp_masks.append(mcp_mask)

    h_mcp_masks = np.array(h_mcp_masks)
    v_mcp_masks = np.array(v_mcp_masks)

    Nx, Ny = shape[::-1]
    h_mcp_masks = h_mcp_masks.reshape(Nx - 1, Ny, *h_mcp_masks.shape[-2:])
    mcp_masks = {}

    for i, row in enumerate(keys):
        cum_mask = h_mcp_masks[i].sum(axis=0)

        for j, k in enumerate(row):

            h_mcp_mask = np.where(cum_mask == Nx - (j + 1), 1, 0)
            v_mcp_mask = np.where(v_mcp_masks[j] == Ny - (i + 1), 1, 0)
            mcp_mask = np.sum((h_mcp_mask, v_mcp_mask), axis=0)
            mcp_masks[k] = np.where(mcp_mask == mcp_mask.max(), 1, 0)

    return mcp_masks


def get_costs(warpeds, masks):
    """
    """
    output_shape = np.array(list(warpeds.values())[0].shape)
    xmax, ymax = output_shape - 1
    Nx, Ny = shape[::-1]

    h_costs = []
    v_costs = []

    for col in keys.T:
        for i, (k1, k2) in enumerate(zip(col, col[1:])):

            costs = generate_costs(np.abs(warpeds[k2] - warpeds[k1]),
                                   masks[k2] & masks[k1])
            h_costs.append(costs)


def preview(data):
    """
    """
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    fig, axes = plt.subplots(*shape)

    for k_row, ax_row in zip(keys, axes):
        for k, ax in zip(k_row, ax_row):

            ax.imshow(FM_imgs[k])
            ax.axis('off')

    fig.subplots_adjust(wspace=0.05, hspace=0.05)


def crude_tile(warpeds, masks):
    """
    """

    # warpeds, masks = warp_images(data, translations)

    warpeds_stitched = np.sum(list(warpeds.values()), axis=0)
    masks_stitched = np.sum(list(masks.values()), axis=0)

    stitched_norm = np.true_divide(warpeds_stitched, masks_stitched,
                                   out=np.zeros_like(warpeds_stitched),
                                   where=(masks_stitched != 0))

    return stitched_norm



def tile_images(data, translations):
    """

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Will have to separate FM and EM images
    """
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    warpeds, masks = warp_images(data, translations)

    mcp_masks = get_puzzle_pieces(keys, shape, warpeds, masks)

    stitched = []
    for k in keys.flatten():
        patch = np.where(mcp_masks[k], warpeds[k], 0)
        stitched.append(patch)

    stitched = np.sum(stitched, axis=0)

    return stitched



if __name__ == '__main__':
    # dir_name = 'rat-pancreas'
    # dir_name = 'nano-diamonds'
    # filenames = ['rat-pancreas//tile_4-2.h5',
    #              'rat-pancreas//tile_4-3.h5',
    #              'rat-pancreas//tile_4-4.h5',
    #              'rat-pancreas//tile_5-2.h5',
    #              'rat-pancreas//tile_5-3.h5',
    #              'rat-pancreas//tile_5-4.h5']
    filenames = glob(
        '../SECOM/*/orange_1200x900_overlap-30/dmonds*_[01]x*[012]y*')

    data = load_data(filenames=filenames)
    FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)
    shape = get_shape(data)

    crop_kws = {
        'overlap': 30
    }

    ORB_kws = {
        'downscale': 2,
        'n_keypoints': 2000,
        'fast_threshold': 0.05}

    ransac_kws = {
        'min_samples': 5,
        'residual_threshold': 10,
        'max_trials': 5000}

    translations = get_translations(data,
                                    crop_kws=crop_kws,
                                    ORB_kws=ORB_kws,
                                    ransac_kws=ransac_kws)

    # stitched = tile_images(data, translations)

    # warpeds, h_mcp_masks, v_mcp_masks = warp_images(data, translations)

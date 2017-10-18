# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 18-10-2017 18:50:10
"""

import os
from glob import glob
import numpy as np
from skimage.external import tifffile
import matplotlib.pyplot as plt
import h5py

from skimage.feature import ORB, match_descriptors
from skimage.feature import register_translation
from skimage.transform import SimilarityTransform, AffineTransform, warp
from skimage.graph import route_through_array
from skimage.measure import label, ransac

import odemis_data
import odemis_utils


def get_shape(data):
    """
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    xs = np.array(list(x_positions.values()))
    ys = np.array(list(y_positions.values()))

    cols = np.unique(np.round(xs, decimals=-1)).size
    rows = np.unique(np.round(ys, decimals=-1)).size

    shape = (rows, cols)
    return shape


def get_keys(data):
    """
    """
    import pandas as pd

    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    shape = get_shape(data)

    df = pd.DataFrame([x_positions, y_positions]).T
    df = df.sort_values([1, 0], ascending=[False, True])
    keys = df.index.values.reshape(shape)

    return keys


# def get_translations_fast(data):
#     """
#     """
#     img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
#     shape = get_shape(data)
#     keys = get_keys(data)

#     shifts = []
#     for row in keys:
#         shifts.append(np.zeros(2))
#         for k1, k2 in zip(row, row[1:]):
#             shift, error, phase_difference = register_translation(
#                 FM_imgs[k1], FM_imgs[k2])
#             shifts.append(shift[::-1])

#     # Convert shifts to numpy array
#     shifts = np.array(shifts).reshape(shape + (2,))

#     # Accumulate translations
#     cum_shifts = np.cumsum(shifts, axis=1)

#     translations = {}
#     for row_keys, row_shifts in zip(keys, cum_shifts):
#         for k, shift in zip(row_keys, row_shifts):
#             translations[k] = SimilarityTransform(translation=shift)

#     return translations


def detect_features(img, ORB_kws=None):
    """
    """
    ORB_kws = {} if ORB_kws is None else ORB_kws

    orb = ORB(**ORB_kws)

    orb.detect_and_extract(img.astype(float))
    kps = orb.keypoints
    dps = orb.descriptors

    return kps, dps


def find_matches(img1, img2, ORB_kws=None):
    """
    """
    kps_img1, dps_img1 = detect_features(img1, ORB_kws=ORB_kws)
    kps_img2, dps_img2 = detect_features(img2, ORB_kws=ORB_kws)

    matches = match_descriptors(dps_img1, dps_img2, cross_check=True)
    return kps_img1, kps_img2, matches


def estimate_transform(img1, img2, ORB_kws=None, ransac_kws=None):
    """
    """
    kps_img1, kps_img2, matches = find_matches(img1, img2, ORB_kws=ORB_kws)
    src = kps_img2[matches[:, 1]][:, ::-1]
    dst = kps_img1[matches[:, 0]][:, ::-1]

    ransac_kws = {} if ransac_kws is None else ransac_kws

    model, inliers = ransac((src, dst), AffineTransform,
                            min_samples=4, residual_threshold=1,
                            **ransac_kws)
    return model


def get_translations_robust(data):
    """
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    shape = get_shape(data)
    keys = get_keys(data)

    ORB_kws = {'downscale': 2,
               'n_keypoints': 800,
               'fast_threshold': 0.05}
    ransac_kws = {'max_trials': 600}

    h_shifts = []
    for row in keys:
        h_shifts.append(np.zeros(2))
        for k1, k2 in zip(row, row[1:]):

            model = estimate_transform(FM_imgs[k1], FM_imgs[k2],
                                       ORB_kws=ORB_kws,
                                       ransac_kws=ransac_kws)

            shift = model.translation
            h_shifts.append(shift)

    v_shifts = []
    for col in keys.T:
        v_shifts.append(np.zeros(2))
        for k1, k2 in zip(col, col[1:]):

            model = estimate_transform(FM_imgs[k1], FM_imgs[k2],
                                       ORB_kws=ORB_kws,
                                       ransac_kws=ransac_kws)

            shift = model.translation
            v_shifts.append(shift)

    h_shifts = np.array(h_shifts).reshape(shape + (2,))
    v_shifts = np.array(v_shifts).reshape(shape[::-1] + (2,))

    cum_h_shifts = np.cumsum(h_shifts, axis=1)
    cum_v_shifts = np.cumsum(v_shifts, axis=1)

    translations = (cum_h_shifts.reshape(shape[-1]*2, 2) +
                    cum_v_shifts.reshape(shape[-1]*2, 2))

    return translations

    translations = translations - translations[:, 1].min()
    translations = translations.astype(np.int64)

    transforms = {}
    for k, t in zip(keys.flatten(), translations):
        transforms[k] = AffineTransform(translation=t)

    return translations, transforms


def tile_images(data, translations, transforms):
    """
    Notes
    -----
    Will have to separate FM and EM images
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = get_keys(data)

    # translations, transforms = get_translations_robust(data)

    # import pickle
    # with open('translations.pickle', 'rb') as handle:
    #     translations = pickle.load(handle)

    H_px, W_px = FM_imgs[keys.flatten()[0]].shape

    output_shape = np.array(
        [translations[:, 1].max() - translations[:, 1].min() + H_px,
         translations[:, 0].max() - translations[:, 0].min() + W_px])

    warpeds = {}
    masks = {}

    for i, k in enumerate(keys.flatten()):

        warped = warp(FM_imgs[k], transforms[k], order=3,
                      output_shape=output_shape, cval=-1)

        mask = (warped != -1)
        warped[~mask] = 0

        warpeds[k] = warped
        masks[k] = mask

    warpeds_stitched = np.sum(list(warpeds.values()), axis=0)
    masks_stitched = np.sum(list(masks.values()), axis=0)

    stitched_norm = np.true_divide(warpeds_stitched, masks_stitched,
                                   out=np.zeros_like(warpeds_stitched),
                                   where=(masks_stitched != 0))
    return warpeds, masks


if __name__ == '__main__':
    dir_name = 'rat-pancreas'
    # dir_name = 'nano-diamonds'
    filenames = ['rat-pancreas//tile_4-2.h5',
                 'rat-pancreas//tile_4-3.h5',
                 'rat-pancreas//tile_4-4.h5',
                 'rat-pancreas//tile_5-2.h5',
                 'rat-pancreas//tile_5-3.h5',
                 'rat-pancreas//tile_5-4.h5']

    data = odemis_data.load_data(filenames=filenames)
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    # stitched = tile_images(data)

    # fig, ax = plt.subplots()
    # ax.imshow(stitched)

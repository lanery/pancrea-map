# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 18-10-2017 14:41:46
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

import odemis_utils


def load_data(dir_name=None, filenames=None):
    """
    """
    if dir_name is not None:
        tiff_files = sorted(glob(dir_name + '/*ome.tiff'))
        h5_files = sorted(glob(dir_name + '/*.h5'))

    elif filenames is not None:
        tiff_files = [tf for tf in filenames if 'ome.tiff' in tf]
        h5_files = [h5 for h5 in filenames if 'h5' in h5]

    else:
        msg = "No data origin provided."
        raise ValueError(msg)

    if not (tiff_files or h5_files):
        msg = "No image data in '{}'".format(dir_name)
        raise FileNotFoundError(msg)

    img_dict = {}
    FM_imgs = {}
    EM_imgs = {}
    x_positions = {}
    y_positions = {}

    if len(tiff_files) > 0:

        for tiff_file in tiff_files:
            k = os.path.basename(tiff_file).split('.')[0]
            img_dict[k] = tifffile.TiffFile(tiff_file)

        for k, tiff in img_dict.items():
            try:
                FM_imgs[k] = odemis_utils.auto_bc(tiff.pages[1].asarray())
            except IndexError:
                pass

            try:
                EM_imgs[k] = tiff.pages[2].asarray()
            except IndexError:
                pass

            x_positions[k] = tiff.pages[1].tags['x_position'].value[0]
            y_positions[k] = tiff.pages[1].tags['y_position'].value[0]

    if len(h5_files) > 0:

        for h5_file in h5_files:
            k = os.path.basename(h5_file).split('.')[0]
            img_dict[k] = h5py.File(h5_file)

        for k, h5 in img_dict.items():
            try:
                FM_imgs[k] = h5['Acquisition0']['ImageData']['Image'].value
                FM_imgs[k] = odemis_utils.auto_bc(FM_imgs[k])
                if len(FM_imgs[k].shape) > 3:
                    FM_imgs[k] = FM_imgs[k][0,0,0,:,:]
            except KeyError:
                pass

            try:
                EM_imgs[k] = h5['SEMimage']['ImageData']['Image'].value
            except KeyError:
                pass

            x_positions[k] = (
                h5['Acquisition0']['ImageData']['XOffset'].value * 1e6)
            y_positions[k] = (
                h5['Acquisition0']['ImageData']['YOffset'].value * 1e6)

    data = img_dict, FM_imgs, EM_imgs, x_positions, y_positions
    return data


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


def sort_keys(x_positions, y_positions, shape):
    """
    """
    import pandas as pd

    df = pd.DataFrame([x_positions, y_positions]).T
    df = df.sort_values([1, 0], ascending=[False, True])
    keys = df.index.values.reshape(shape).tolist()
    return keys


def get_translations(data):
    """
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    shape = get_shape(data)
    keys = sort_keys(x_positions, y_positions, shape)

    shifts = []
    for row in keys:
        shifts.append(np.zeros(2))
        for k1, k2 in zip(row, row[1:]):
            shift, error, phase_difference = register_translation(
                FM_imgs[k1], FM_imgs[k2])
            shifts.append(shift[::-1])

    # Convert shifts to numpy array
    shifts = np.array(shifts).reshape(shape + (2,))

    # Accumulate translations
    cum_shifts = np.cumsum(shifts, axis=1)
    # cum_shifts[1,:,1] += 1500

    translations = {}
    for row_keys, row_shifts in zip(keys, cum_shifts):
        for k, shift in zip(row_keys, row_shifts):
            translations[k] = SimilarityTransform(translation=shift)

    return translations


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
    keys = sort_keys(x_positions, y_positions, shape)
    keys_T = np.array(keys).T.tolist()

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
    for col in keys_T:
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

    tot_shifts = (cum_h_shifts.reshape(shape[-1]*2, 2) +
                  cum_v_shifts.reshape(shape[-1]*2, 2)).reshape(shape + (2,))

    translations = {}
    for row_keys, row_shifts in zip(keys, tot_shifts):
        for k, shift in zip(row_keys, row_shifts):
            translations[k] = SimilarityTransform(translation=shift)

    return translations


def tile_images(data):
    """
    Notes
    -----
    Will have to separate FM and EM images
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = list(img_dict.keys())

    translations = get_translations_robust(data)

    # import pickle
    # with open('translations.pickle', 'rb') as handle:
    #     translations = pickle.load(handle)

    shifts = np.array([tr.translation for tr in translations.values()],
                      dtype=np.int64)

    H_px, W_px = data[1][list(data[1].keys())[0]].shape
    output_shape = np.array([(shifts[:, 1].max() - shifts[:, 1].min()) + H_px,
                             (shifts[:, 0].max() - shifts[:, 0].min()) + W_px])

    warpeds = {}
    masks = {}

    for i, k in enumerate(keys):

        warped = warp(FM_imgs[k], translations[k].inverse, order=3,
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
    return stitched_norm


if __name__ == '__main__':
    dir_name = 'rat-pancreas2'
    # dir_name = 'test_images'
    filenames = ['rat-pancreas2//tile_4-2.h5',
                 'rat-pancreas2//tile_4-3.h5',
                 'rat-pancreas2//tile_4-4.h5',
                 'rat-pancreas2//tile_5-2.h5',
                 'rat-pancreas2//tile_5-3.h5',
                 'rat-pancreas2//tile_5-4.h5']

    data = load_data(filenames=filenames)
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    # stitched = tile_images(data)

    # fig, ax = plt.subplots()
    # ax.imshow(stitched)

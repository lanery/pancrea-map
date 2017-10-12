# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 12-10-2017 16:46:49
"""

import os
from glob import glob
import numpy as np
from skimage.external import tifffile
import matplotlib.pyplot as plt
import h5py

from skimage.feature import register_translation
from skimage.transform import SimilarityTransform, warp
from skimage.graph import route_through_array
from skimage.measure import label

import odemis_utils


def calc_HFW(M, pixel_size=6.5, N_pixels=2560):
    """
    Calculate Horizontal Field Width of an image

    Parameters
    ----------
    M : scalar
        Magnification of the objective lens
    pixel_size : scalar
        Pixel size (assuming square pixels) of pixel from sCMOS chip
    N_pixels : scalar
        Number of pixels in the horizontal dimension

    Returns
    -------
    HFW : scalar
        The calculated Horizontal Field Width

    Notes
    -----
    Based on known specifications of the Andor Zyla 5.5 sCMOS camera (pixel
    size and number of pixels) and magnification of Nikon objective lens.
    http://www.andor.com/scientific-cameras/neo-and-zyla-scmos-cameras/zyla-55-scmos#specifications
    """
    HFW = (pixel_size * N_pixels) / M
    return HFW


def calc_N_pixels(dist, M, pixel_size=6.5):
    """
    """
    N_pixels = dist * M / pixel_size
    return N_pixels


def calibrate_stage_movement(delta_x, delta_y=None):
    """
    Notes
    -----
    Would be way better if the exact stage calibration were known or if
    there was a superior method for calculating stage position/movement
    from images
    """
    if delta_y is None:
        movement_in_microns = delta_x * (1 / 1677.904)

    else:
        movement_in_microns_x = delta_x * (1 / 1677.904)
        movement_in_microns_y = delta_y * (1 / 1677.904)
        movement_in_microns = (movement_in_microns_x, movement_in_microns_y)

    return movement_in_microns


def load_data(dir_name):
    """
    """
    ome_files = sorted(glob(dir_name + '\\*ome.tiff'))
    h5_files = sorted(glob(dir_name + '\\*.h5'))

    img_dict = {}
    FM_imgs = {}
    EM_imgs = {}
    x_positions = {}
    y_positions = {}

    if len(ome_files) > 0:

        for ome_file in ome_files:
            k = os.path.basename(ome_file).split('.')[0]
            img_dict[k] = tifffile.TiffFile(ome_file)

        for k, tiff in img_dict.items():
            try:
                FM_imgs[k] = odemis_utils.auto_bc(tiff.pages[1].asarray())
            except IndexError:
                pass

            try:
                EM_imgs[k] = tiff.pages[2].asarray()
            except IndexError:
                pass

            x_positions[k] = calibrate_stage_movement(
                tiff.pages[1].tags['x_position'].value[0]) - 1e6
            y_positions[k] = calibrate_stage_movement(
                tiff.pages[1].tags['y_position'].value[0]) - 1e6

    if len(h5_files) > 0:

        for h5_file in h5_files:
            k = os.path.basename(h5_file).split('.')[0]
            img_dict[k] = h5py.File(h5_file)

        for k, h5 in img_dict.items():
            try:
                FM_imgs[k] = h5['Acquisition0']['ImageData']['Image'].value
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


def get_translations_from_stage_positions(data):
    """

    # Code that I didn't want to lose forever
    # img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    # xs = np.array(list(x_positions.values()))
    # ys = np.array(list(y_positions.values()))

    # H_px, W_px = list(FM_imgs.values())[0].shape
    # x_pixel_range = int(calc_N_pixels(np.max(xs) - np.min(xs), M=M) + W_px)
    # y_pixel_range = int(calc_N_pixels(np.max(ys) - np.min(ys), M=M) + H_px)

    # output_shape = np.array([y_pixel_range, x_pixel_range])

    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = list(img_dict.keys())

    xs = np.array(list(x_positions.values()))
    ys = np.array(list(y_positions.values()))

    # Vectorize stage movements
    raw_translations = np.vstack((np.diff(xs), np.diff(ys))).T
    # Add (0, 0) initial stage movement
    raw_translations = np.insert(raw_translations, 0, 0, axis=0)
    # Convert from physical distance to pixel distance
    raw_translations = calc_N_pixels(raw_translations, M=M)
    # Accumulate translations
    cum_translations = np.cumsum(raw_translations, axis=0)

    translations = {}

    for i, k in enumerate(keys):
        translations[k] = SimilarityTransform(
            translation=cum_translations[i])

    return translations


def get_translations(data):
    """
    TODO: Have some scheme for organizing / sorting images
    """
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = list(img_dict.keys())

    xs = np.array(list(x_positions.values()))
    ys = np.array(list(y_positions.values()))

    keys_sorted = np.array(keys)[xs.argsort()]

    shifts = []
    for k1, k2 in zip(keys_sorted, keys_sorted[1:]):
        shift, error, phase_difference = register_translation(
            FM_imgs[k1], FM_imgs[k2])
        shifts.append(shift[::-1])

    # Accumulate translations
    cum_shifts = np.cumsum(shifts, axis=0)

    # Add (0, 0) translation for first image
    translations = {keys_sorted[0]: SimilarityTransform(
        translation=np.array([0, 0]))}

    for i, k in enumerate(keys_sorted[1:]):
        translations[k] = SimilarityTransform(
            translation=cum_shifts[i])    

    return translations


def tile_images(dir_name):
    """
    Notes
    -----
    Will have to separate FM and EM images
    """
    data = load_data(dir_name)
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data
    keys = list(img_dict.keys())

    translations = get_translations(data)

    shifts = np.array([tr.translation for tr in translations.values()])

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
    # dir_name = 'rat-pancreas'
    dir_name = 'test_images2'

    # data = load_data('test_images2')
    # img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    stitched = tile_images(dir_name)

    fig, ax = plt.subplots()
    ax.imshow(stitched)

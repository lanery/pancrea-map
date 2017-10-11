# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 11-10-2017 18:12:36
"""

import os
from glob import glob
import numpy as np
from skimage.external import tifffile
import matplotlib.pyplot as plt; plt.set_cmap('magma'); plt.close()
import h5py

from skimage.transform import SimilarityTransform, warp

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
                FM_imgs[k] = tiff.pages[1].asarray()
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


def tile_images(dir_name, M=40):
    """
    Notes
    -----
    Will have to separate FM and EM images
    """
    HFW = calc_HFW(M)

    data = load_data(dir_name)
    img_dict, FM_imgs, EM_imgs, x_positions, y_positions = data

    xs = np.array(list(x_positions.values()))
    ys = np.array(list(y_positions.values()))

    H_px, W_px = list(FM_imgs.values())[0].shape
    x_pixel_range = int(calc_N_pixels(np.max(xs) - np.min(xs), M=M) + W_px)
    y_pixel_range = int(calc_N_pixels(np.max(ys) - np.min(ys), M=M) + H_px)

    output_shape = np.array([y_pixel_range, x_pixel_range])

    raw_translations = np.vstack((np.diff(xs), np.diff(ys))).T
    raw_translations = np.insert(raw_translations, 0, 0, axis=0)
    translations = calc_N_pixels(raw_translations, M=M)
    cum_translations = np.cumsum(translations, axis=0)

    warpeds = {}
    masks = {}

    for i, (k, FM_img) in enumerate(FM_imgs.items()):

        offset = SimilarityTransform(translation=cum_translations[i])

        warped = warp(FM_img, offset.inverse, order=3,
                      output_shape=output_shape, cval=-1)

        mask = (warped != -1)
        warped[~mask] = 0

        warpeds[k] = warped
        masks[k] = mask


    return warpeds, masks



if __name__ == '__main__':
    # dir_name = 'rat-pancreas'
    dir_name = 'test_images2'
    M = 48

    warpeds, masks = tile_images(dir_name, M=M)


    stitched = np.sum([warpeds['test3'],
                       warpeds['test5'],
                       warpeds['test7'],
                       warpeds['test9']], axis=0)

    stitched = odemis_utils.auto_bc(stitched)

    plt.imshow(stitched)

    plt.show()

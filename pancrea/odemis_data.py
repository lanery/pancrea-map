# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   18-10-2017 15:41:15
@Last Modified by:   rlane
@Last Modified time: 25-10-2017 18:22:24

Module for reading and writing image data
"""

import os
from glob import glob
import h5py

from skimage.external import tifffile
from skimage.transform import pyramid_reduce

from .odemis_utils import auto_bc


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
                FM_imgs[k] = auto_bc(tiff.pages[1].asarray())
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
                FM_img_raw = h5['Acquisition0']['ImageData']['Image'].value
                FM_img_abc = auto_bc(FM_img_raw)
                # FM_img_abc = FM_img_raw
                if len(FM_img_abc.shape) > 3:
                    FM_img_sliced = FM_img_abc[0,0,0,:,:]
                # FM_img_red = pyramid_reduce(FM_img_sliced)
                FM_imgs[k] = FM_img_sliced
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

    data = FM_imgs, EM_imgs, x_positions, y_positions
    return data

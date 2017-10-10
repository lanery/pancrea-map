# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   10-10-2017 12:00:47
@Last Modified by:   rlane
@Last Modified time: 10-10-2017 18:43:15
"""

import os
from glob import glob
import numpy as np
from skimage.external import tifffile
import matplotlib.pyplot as plt

# from skimage.feature import ORB, match_descriptors
# from skimage.transform import ProjectiveTransform, SimilarityTransform
# from skimage.transform import rescale, warp
# from skimage.measure import ransac, label
# from skimage.graph import route_through_array

import stitching_utils


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


def calibrate_stage_movement(delta_x):
    """
    Notes
    -----
    This is a really awful function and should totally be removed once
    it is known how to properly calibrate stage movements or, preferably,
    read the correct stage position from images
    """
    odemis_2_reality = 
    return


def tile_images(dir_name):
    """
    """
    M = 10  # magnification

    # Load data into dict with filename (minus extension) as key
    ome_files = sorted(glob(dir_name + '\\*ome.tiff'))
    img_dict = {}

    for ome_file in ome_files:
        k = os.path.basename(ome_file).split('.')[0]
        img_dict[k] = tifffile.TiffFile(ome_file)

    return img_dict



if __name__ == '__main__':
    dir_name = 'test_images'
    img_dict = tile_images(dir_name)

# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   01-11-2017 11:42:39
@Last Modified by:   rlane
@Last Modified time: 01-11-2017 15:08:52
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


class Mosaic(object):
    """docstring for Mosaic"""
    def __init__(self, imgs):
        super(Mosaic, self).__init__()
        self.arg = arg
        



# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   01-11-2017 11:50:27
@Last Modified by:   rlane
@Last Modified time: 01-11-2017 18:33:46
"""

import os
import re
from glob import glob
import numpy as np
import h5py

from skimage.io import imread
from skimage.io import imsave
from skimage.external import tifffile
from skimage.transform import pyramid_reduce

from .odemis_utils import auto_bc


class OdemisMosaicData(object):
    """
    Data class for images obtained by Odemis that serves as input to Mosaic
    """
    def __init__(self, directory=None, filenames=None):
        super(OdemisMosaicData, self).__init__()
        
        if directory is None:
            self.relpath = os.path.dirname(filenames[0])
        else:
            self.relpath = directory
        
        if filenames is None:
            self.fns = self._get_filenames()
        else:
            self.fns = filenames

        self.is_tiff, self.is_h5 = self._get_file_format()
        self.keys = self._get_keys()
        self.x_positions, self.y_positions = self._get_positions()
        self.shape = self._get_shape()
        self.sorted_keys = self._get_sorted_keys()

        self.EM_imgs = self._load_EM_imgs()
        self.FM_imgs = self._load_FM_imgs()

    def _get_filenames(self):
        """
        """
        self.fns = (sorted(glob(self.relpath + '/*ome.tiff')) +
                    sorted(glob(self.relpath + '/*.h5')))
        return self.fns

    def _get_keys(self):
        """
        """
        keys = []
        for fn in self.fns:
            base_fn = os.path.basename(fn)
            keys.append(base_fn.split('.')[0])
        self.keys = keys
        return self.keys

    def _get_shape(self, precision=-1):
        """
        """
        xs = np.array(list(self.x_positions.values()))
        ys = np.array(list(self.y_positions.values()))

        n_cols = np.unique(np.round(xs, decimals=precision)).size
        n_rows = np.unique(np.round(ys, decimals=precision)).size

        self.shape = (n_cols, n_rows)
        return self.shape

    def _get_sorted_keys(self):
        """
        """
        import pandas as pd

        df = pd.DataFrame([self.x_positions, self.y_positions]).T
        df = df.sort_values([1, 0], ascending=[False, True])
        self.sorted_keys = df.index.values.reshape(self.shape[::-1])
        return self.sorted_keys

    def _get_file_format(self):
        """
        """
        exts = []
        for fn in self.fns:
            fn_base = os.path.basename(fn)
            ext = fn_base.split('.')[-1]
            exts.append(ext)

        if len(set(exts)) > 1:
            msg = "Files must be `*.ome.tiff` or `*.h5`, not both."
            raise TypeError(msg)

        if exts[0] == 'tiff':
            self.is_tiff = True
            self.is_h5 = False
        else:
            self.is_tiff = False
            self.is_h5 = True

        return self.is_tiff, self.is_h5

    def _get_positions(self):
        """
        """
        self.x_positions = {}
        self.y_positions = {}

        for fn in self.fns:
            base_fn = os.path.basename(fn)
            k = base_fn.split('.')[0]

            if self.is_tiff:
                tiff = tifffile.TiffFile(fn)
                x = tiff.pages[1].tags['x_position'].value[0] / 1e2
                y = tiff.pages[1].tags['y_position'].value[0] / 1e2
            else:
                h5 = h5py.File(fn)
                x = h5['Acquisition0']['ImageData']['XOffset'].value * 1e6
                y = h5['Acquisition0']['ImageData']['YOffset'].value * 1e6

            self.x_positions[k] = x
            self.y_positions[k] = y

        return self.x_positions, self.y_positions

    def _load_EM_imgs(self):
        """
        """
        self.EM_imgs = {}

        for fn in self.fns:
            base_fn = os.path.basename(fn)
            k = base_fn.split('.')[0]

            if self.is_tiff:
                tiff = tifffile.TiffFile(fn)
                try:
                    img = tiff.pages[2].asarray()
                except IndexError:
                    img = None

            else:
                h5 = h5py.File(fn)
                try:
                    img = h5['SEMimage']['ImageData']['Image'].value
                except KeyError:
                    img = None

            self.EM_imgs[k] = img

        return self.EM_imgs

    def _load_FM_imgs(self):
        """
        """
        self.FM_imgs = {}

        for fn in self.fns:
            base_fn = os.path.basename(fn)
            k = base_fn.split('.')[0]

            if self.is_tiff:
                tiff = tifffile.TiffFile(fn)
                try:
                    img = tiff.pages[1].asarray()
                except IndexError:
                    img = None

            else:
                h5 = h5py.File(fn)
                try:
                    img = h5['Acquisition0']['ImageData']['Image'].value
                except KeyError:
                    img = None

            self.FM_imgs[k] = img

        return self.FM_imgs


class GenericMosaicData(object):
    """
    Data class dedicated to generic (i.e. not from Odemis) images that
    serves as input to Mosaic
    """
    def __init__(self, directory=None, filenames=None):
        super(GenericMosaicData, self).__init__()
        
        if directory is None:
            self.relpath = os.path.dirname(filenames[0])
        else:
            self.relpath = directory
        
        if filenames is None:
            self.fns = self._get_filenames()
        else:
            self.fns = sorted(filenames)

        self.ftype = self._get_file_format()
        self.keys = self._get_keys()
        self.shape = self._get_shape()
        self.sorted_keys = self._get_sorted_keys()
        self.imgs = self._load_imgs()

    def _get_filenames(self):
        """
        """
        self.fns = sorted(glob(self.relpath))
        return self.fns

    def _get_file_format(self):
        """
        """
        exts = []
        for fn in self.fns:
            fn_base = os.path.basename(fn)
            ext = fn_base.split('.')[-1]
            exts.append(ext)

        if len(set(exts)) > 1:
            msg = "Input files must have same extension."
            raise TypeError(msg)

        self.ftype = exts[0]
        return self.ftype

    def _get_keys(self):
        """
        """
        keys = []
        for fn in self.fns:
            base_fn = os.path.basename(fn)
            keys.append(base_fn.split('.')[0])
        self.keys = keys
        return self.keys

    def _get_shape(self):
        """
        """
        try:
            xs = []
            ys = []
            for key in self.keys:
                x = re.findall('\d+x', key)[-1]
                y = re.findall('\d+y', key)[-1]
                xs.append(int(x[:-1]))
                ys.append(int(y[:-1]))

        except IndexError:
            msg = ("Could not get shape of mosaic data. Check that " +
                   "filenames are formatted e.g. `filename_3x_6y.tiff`.")
            print(msg)

        n_cols = len(set(xs))
        n_rows = len(set(ys))

        self.shape = (n_cols, n_rows)
        return self.shape

    def _get_sorted_keys(self):
        """
        """
        self.sorted_keys = np.array(self.keys).reshape(self.shape).T
        return self.sorted_keys

    def _load_imgs(self):
        """
        """
        self.imgs = {}

        for fn in self.fns:
            base_fn = os.path.basename(fn)
            k = base_fn.split('.')[0]
            self.imgs[k] = imread(fn)

        return self.imgs


def unstitch(img, shape=(4, 3), overlap=25):
    """
    Split up a composite image into overlapping sub-images

    Parameters
    ----------
    img : ndarray, shape (nx, ny)
        Composite image for unstitching

    shape : tuple (optional)
        Shape of unstitched image -> (cols, rows)

    overlap : scalar (optional)
        Percentage overlap between tiled sub-images

    Returns
    -------
    sub_imgs : dict
        Collection of tiled images with x, y position as keys
        and image data as values

    Notes
    -----
    For best results, img should have a rather high resolution
    e.g. > 10 Mpx
    """

    Nx, Ny = shape
    nx, ny = img.shape[::-1]

    ext_nx = int(nx * (1 + overlap/100))
    ext_ny = int(ny * (1 + overlap/100))

    img = np.pad(img, [(0, ext_ny), (0, ext_nx)], mode='constant')

    opx = int((nx // Nx) * (overlap / 100))
    opy = int((ny // Ny) * (overlap / 100))

    sub_imgs = {}

    for j in range(Ny):
        iy1 = j * ny // Ny
        iy2 = (j + 1) * ny // Ny + opy

        for i in range(Nx):
            ix1 = i * nx // Nx
            ix2 = (i + 1) * nx // Nx + opx

            sub_imgs[i, j] = img[iy1:iy2, ix1:ix2]

    return sub_imgs


def save_sub_images(sub_imgs, filename):
    """
    Convenience function for saving the output of `unstitch`

    Examples
    --------
    >>> save_sub_images(sub_imgs, 'bar/foo/cell.tiff')
    """
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fn, ext = filename.split('.')
    for i, j in sub_imgs.keys():
        coords = "_{}x_{}y".format(i + 1, j + 1)
        out = fn + coords + '.' + ext
        imsave(out, sub_imgs[i, j])

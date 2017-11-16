# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   01-11-2017 11:50:27
@Last Modified by:   rlane
@Last Modified time: 16-11-2017 15:11:32
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


class MosaicData(object):
    """docstring for MosaicData"""
    def __init__(self, directory=None, filenames=None, **kwargs):

        if directory is None:
            try:
                self.abspath = os.path.dirname(os.path.abspath(filenames[0]))
                self.relpath = os.path.dirname(os.path.relpath(filenames[0]))
            except IndexError as exc:
                msg = ("Failed to find provided `filenames` or `filenames` "
                       "is empty.")
                raise IndexError(msg) from exc
        else:
            self.abspath = os.path.abspath(directory)
            self.relpath = os.path.relpath(directory)

        if filenames is None:
            self.fns = self._get_filenames()
        else:
            self.fns = sorted([os.path.abspath(fn) for fn in filenames])

        self.ftype = self._get_file_format()
        self.keys = self._get_keys()

        if self.ftype == 'tiff':
            self.is_tiff = True
            self.is_h5 = False
        elif self.ftype == 'h5':
            self.is_tiff = False
            self.is_h5 = True
        else:
            self.is_tiff = False
            self.is_h5 = False

    def _get_filenames(self):
        """
        """
        self.fns = sorted(glob(self.abspath))
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


class GenericMosaicData(MosaicData):
    """
    Data class dedicated to generic (i.e. not from Odemis) images that
    serves as input to Mosaic
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.shape = self._get_shape()
        self.sorted_keys = self._get_sorted_keys()
        self.imgs = self._load_imgs()

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

        except IndexError as exc:
            msg = ("Could not get shape of mosaic data. Check that "
                   "filenames are formatted e.g. `filename_3x_6y.tiff`.")
            raise IndexError(msg) from exc

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


class OdemisMosaicData(MosaicData):
    """
    Data class for images obtained by Odemis that serves as input to Mosaic
    """
    def __init__(self, mapping, **kwargs):
        super().__init__(**kwargs)

        self.mapping = mapping

        self.x_positions, self.y_positions = self._get_positions()
        self.shape = self._get_shape()

        if self.shape == (1, 1):
            self.sorted_keys = self.keys
        else:
            self.sorted_keys = self._get_sorted_keys(**kwargs)

        self.imgs = self._load_imgs()

    def _get_shape(self, precision=-1):
        """
        """
        xs = np.array(list(self.x_positions.values()))
        ys = np.array(list(self.y_positions.values()))

        n_cols = np.unique(np.round(xs, decimals=precision)).size
        n_rows = np.unique(np.round(ys, decimals=precision)).size

        self.shape = (n_cols, n_rows)
        return self.shape

    def _get_sorted_keys(self, **kwargs):
        """
        """
        import pandas as pd

        df = pd.DataFrame([self.x_positions, self.y_positions]).T
        df = df.sort_values([1, 0], ascending=[False, True])

        try:
            self.sorted_keys = df.index.values.reshape(self.shape[::-1])
            if kwargs.get('inverse', False):
                self.sorted_keys = self.sorted_keys[::-1, ::-1]
        except ValueError:
            # TODO: make a warning that says failed to sort keys
            self.sorted_keys = self.keys

        return self.sorted_keys

    def _get_positions(self):
        """
        """
        self.x_positions = {}
        self.y_positions = {}

        tag_id = list(self.mapping.values())[0]

        for fn in self.fns:
            base_fn = os.path.basename(fn)
            k = base_fn.split('.')[0]

            if self.is_tiff:
                tiff = tifffile.TiffFile(fn)
                x = tiff.pages[tag_id].tags['x_position'].value[0] / 1e2
                y = tiff.pages[tag_id].tags['y_position'].value[0] / 1e2
            else:
                h5 = h5py.File(fn)
                x = h5[tag_id]['ImageData']['XOffset'].value * 1e6
                y = h5[tag_id]['ImageData']['YOffset'].value * 1e6

            self.x_positions[k] = x
            self.y_positions[k] = y

        return self.x_positions, self.y_positions

    def _load_imgs(self):
        """
        """
        self.imgs = {k: {} for k in self.mapping.keys()}

        for map_key, tag_id in self.mapping.items():

            for fn in self.fns:
                base_fn = os.path.basename(fn)
                k = base_fn.split('.')[0]

                if self.is_tiff:
                    tiff = tifffile.TiffFile(fn)
                    img = tiff.pages[tag_id].asarray()

                else:
                    h5 = h5py.File(fn)
                    img = h5[tag_id]['ImageData']['Image'].value
                    img = np.squeeze(img)

                    from skimage.exposure import equalize_adapthist
                    img = equalize_adapthist(img)
                    # img = auto_bc(img)

                    from skimage import img_as_uint
                    img = img_as_uint(img)

                    from skimage.filters import gaussian
                    img = gaussian(img, 1)

                self.imgs[map_key][k] = img

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
    e.g. >= 10 Mpx
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
    dirname = os.path.dirname(os.path.abspath(filename))
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    fn, ext = filename.split('.')
    for i, j in sub_imgs.keys():
        coords = "_{}x_{}y".format(i + 1, j + 1)
        out = fn + coords + '.' + ext
        imsave(out, sub_imgs[i, j])

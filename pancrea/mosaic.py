# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   01-11-2017 11:42:39
@Last Modified by:   rlane
@Last Modified time: 02-11-2017 16:22:15
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

from .mosaic_data import GenericMosaicData


class Mosaic(object):
    """docstring for Mosaic"""
    def __init__(self, mosaic_data):
        super(Mosaic, self).__init__()

        self.keys = mosaic_data.keys
        self.shape = mosaic_data.shape
        self.sorted_keys = mosaic_data.sorted_keys
        self.imgs = mosaic_data.imgs

        self.Nx, self.Ny = self.shape
        self.nx, self.ny = self.imgs[self.keys[0]].shape[::-1]

        # self.match_kws = 
        # self.ORB_kws = 
        # self.ransac_kws = 
        # self.FFT_kws = 

    def get_translations(self, method='robust', match_kws=None,
                         ORB_kws=None, ransac_kws=None, FFT_kws=None):
        """
        """
        if match_kws is not None:
            match_kws['direction'] = 'horizontal'

        h_shifts = []
        for row in self.sorted_keys:
            h_shifts.append(np.zeros(2))

            for k1, k2 in zip(row, row[1:]):
                img1 = self.imgs[k1]
                img2 = self.imgs[k2]

                if method == 'robust':
                    model = estimate_transform(img1, img2,
                                               match_kws=match_kws,
                                               ORB_kws=ORB_kws,
                                               ransac_kws=ransac_kws)
                    shift = model.translation
                else:
                    shift = estimate_translation(img1, img2,
                                                 FFT_kws=FFT_kws)

                h_shifts.append(shift)

        if match_kws is not None:
            match_kws['direction'] = 'vertical'

        v_shifts = []
        for col in self.sorted_keys.T:
            v_shifts.append(np.zeros(2))

            for k1, k2 in zip(col, col[1:]):
                img1 = self.imgs[k1]
                img2 = self.imgs[k2]

                if method == 'robust':
                    model = estimate_transform(img1, img2,
                                               match_kws=match_kws,
                                               ORB_kws=ORB_kws,
                                               ransac_kws=ransac_kws)
                    shift = model.translation
                else:
                    shift = estimate_translation(img1, img2,
                                                 FFT_kws=FFT_kws)

                v_shifts.append(shift)

        h_shifts = np.array(h_shifts).reshape(self.shape[::-1] + (2,))
        v_shifts = np.array(v_shifts).reshape(self.shape + (2,))

        cum_h_shifts = np.cumsum(h_shifts, axis=1)
        cum_v_shifts = np.cumsum(v_shifts, axis=1)

        translations = cum_h_shifts + np.swapaxes(cum_v_shifts, 0, 1)
        translations = translations.reshape(np.product(self.shape), 2)
        translations[:, 1] = translations[:, 1] - translations[:, 1].min()
        translations = translations.astype(np.int64)
        translations = translations.reshape(self.shape[::-1] + (2,))

        self.translations = translations
        return self.translations

    def get_transforms(self, transform=AffineTransform):
        """
        """
        self.translations = np.array([[[0,       0],
                                       [1122,    3],
                                       [2237,    6]],

                                      [[0,    1041],
                                       [1120, 1045],
                                       [2248, 1049]]])

        if not hasattr(self, 'translations'):
            match_kws = {'overlap': 25}
            self.translations = self.get_translations(match_kws=match_kws)

        self.transforms = {}
        for krow, trow in zip(self.sorted_keys, self.translations):
            for k, tr in zip(krow, trow):
                self.transforms[k] = transform(translation=tr)
        return self.transforms

    def get_warped_images(self):
        """
        """
        if not hasattr(self, 'transforms'):
            self.transforms = self.get_transforms()

        self.mnx = (self.translations[:,:,0].max() - 
                    self.translations[:,:,0].min()) + self.nx
        self.mny = (self.translations[:,:,1].max() -
                    self.translations[:,:,1].min()) + self.ny
        self.mshape = (self.mny, self.mnx)

        self.warpeds = {}
        self.masks = {}

        for i, k in enumerate(self.sorted_keys.flatten()):
            img = self.imgs[k]
            transform = self.transforms[k]

            warped = warp(img, transform.inverse, order=3,
                          output_shape=self.mshape, cval=-1)

            mask = (warped != -1)
            warped[~mask] = 0

            self.warpeds[k] = warped
            self.masks[k] = mask

        return self.warpeds, self.masks

    def get_costs(self):
        """
        """
        if not hasattr(self, 'warpeds') or not hasattr(self, 'masks'):
            self.warpeds, masks = self.get_warped_images()

        self.h_costs = []
        for col in self.sorted_keys.T:
            for i, (k1, k2) in enumerate(zip(col, col[1:])):

                costs = generate_costs(
                    np.abs(self.warpeds[k2] - self.warpeds[k1]),
                           self.masks[k1] & self.masks[k2])
                self.h_costs.append(costs)

        self.v_costs = []
        for row in self.sorted_keys:
            for i, (k1, k2) in enumerate(zip(row, row[1:])):

                costs = generate_costs(
                    np.abs(self.warpeds[k2] - self.warpeds[k1]),
                           self.masks[k1] & self.masks[k2])
                self.v_costs.append(costs)

        self.h_costs = np.array(self.h_costs).reshape(
            self.Nx, self.Ny-1, self.mny, self.mnx)
        self.h_costs = np.swapaxes(self.h_costs, 0, 1)

        self.v_costs = np.array(self.v_costs).reshape(
            self.Nx-1, self.Ny, self.mny, self.mnx)
        self.v_costs = np.swapaxes(self.v_costs, 0, 1)

        return self.h_costs, self.v_costs

    def get_minimum_cost_paths(self):
        """
        """
        if not hasattr(self, 'h_costs') or not hasattr(self, 'v_costs'):
            self.h_costs, self.v_costs = self.get_costs()

        self.h_mcps = []
        h_mcp_masks = []

        for i in range(self.Ny - 1):
            costs = condsum(*self.h_costs[i,:,:,:])
            costs[:, 0] = 0
            costs[:, -1] = 0

            mask_pts = [[self.mny * (i+1) // self.Ny, 0],
                        [self.mny * (i+1) // self.Ny, self.mnx - 1]]

            mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                         fully_connected=True)
            mcp = np.array(mcp)
            self.h_mcps.append(mcp)

            mcp_mask = np.zeros(self.mshape)
            mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
            mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)
            h_mcp_masks.append(mcp_mask)

        self.v_mcps = []
        v_mcp_masks = []

        for i in range(self.Nx - 1):
            costs = condsum(*self.v_costs[i,:,:,:])
            costs[0, :] = 0
            costs[-1, :] = 0

            mask_pts = [[0, self.mnx * (i+1) // self.Nx],
                        [self.mny - 1, self.mnx * (i+1) // self.Nx]]

            mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                         fully_connected=True)
            mcp = np.array(mcp)
            self.v_mcps.append(mcp)

            mcp_mask = np.zeros(self.mshape)
            mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
            mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)
            v_mcp_masks.append(mcp_mask)

        h_mcp_masks = np.array(h_mcp_masks)
        v_mcp_masks = np.array(v_mcp_masks)

        cum_mask = (self.Nx * h_mcp_masks.sum(axis=0) +
                              v_mcp_masks.sum(axis=0))
        self.mcp_masks = {}
        for i, krow in enumerate(self.sorted_keys):
            for j, k in enumerate(krow):
                mask_val = (self.Nx*self.Ny - 1) - (self.Nx*i + j)
                self.mcp_masks[k] = np.where(cum_mask == mask_val, 1, 0)

        return self.mcp_masks

    def tile_images(self):
        """
        """
        if not hasattr(self, 'mcp_masks'):
            self.mcp_masks = self.get_minimum_cost_paths()

        self.patches = {}
        for krow in self.sorted_keys:
            for k in krow:
                self.patches[k] = np.where(
                    self.mcp_masks[k], self.warpeds[k], 0)
                
        self.stitched = np.sum(list(self.patches.values()), axis=0)
        return self.stitched

    def plot_mosaic(self, stitch_lines=False):
        """
        """
        if not hasattr(self, 'stitched'):
            self.stitched = self.tile_images()

        fig, ax = plt.subplots()
        ax.imshow(self.stitched)

        if stitch_lines:
            for mcp in self.h_mcps:
                ax.plot(mcp[:, 1], mcp[:, 0], '#EEEEEE')

            for mcp in self.v_mcps:
                ax.plot(mcp[:, 1], mcp[:, 0], '#EEEEEE')

    def preview(self):
        """
        """
        pass

    def crude_stitch(self):
        """
        """
        pass


def condsum(*arrs, r=1):
    """
    """
    if len(arrs) == 1:
        return arrs[0]
    else:
        a = condsum(*arrs[1:], r=r)
        return np.where(a==r, arrs[0], a)


def detect_features(img, ORB_kws=None):
    """
    Detect features using ORB.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Wrapper for `skimage.feature.ORB`
    http://scikit-image.org/docs/dev/api/skimage.feature.html#skimage.feature.ORB
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


def find_matches(img1, img2, match_kws=None, ORB_kws=None):
    """
    Find matches between images.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Wrapper for `skimage.feature.match_descriptors`
    http://scikit-image.org/docs/dev/api/skimage.feature.html#match-descriptors
    """
    if match_kws is None:
        kps_img1, dps_img1 = detect_features(img1, ORB_kws=ORB_kws)
        kps_img2, dps_img2 = detect_features(img2, ORB_kws=ORB_kws)

    else:
        try:
            m, n = img1.shape
            o1 = 1 / (1 - (match_kws['overlap'] / 100))
            o2 = 1 / (match_kws['overlap'] / 100)

            if match_kws['direction'] == 'horizontal':
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
            msg = "`match_kws` must contain {}.".format(exc)
            raise KeyError(msg)

    matches = match_descriptors(dps_img1, dps_img2, cross_check=True)
    return kps_img1, kps_img2, matches


def estimate_transform(img1, img2, match_kws=None,
                       ORB_kws=None, ransac_kws=None):
    """
    Estimate Affine transformation between two images.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Wrapper for `skimage.measure.ransac` assuming AffineTransform
    http://scikit-image.org/docs/dev/api/skimage.measure.html#skimage.measure.ransac
    """
    kps_img1, kps_img2, matches = find_matches(
        img1, img2, match_kws=match_kws, ORB_kws=ORB_kws)

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
    Estimate lateral translation between two images.

    Parameters
    ----------

    Returns
    -------

    Notes
    -----
    Wrapper for `skimage.feature.register_translation`
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


if __name__ == '__main__':

    fns = glob('sample_data/dartmouth_lungs/*[123]*[12]*.tiff')
    gmd = GenericMosaicData(filenames=fns)
    mosaic = Mosaic(gmd)
    mosaic.plot_mosaic(stitch_lines=True)

    plt.show()

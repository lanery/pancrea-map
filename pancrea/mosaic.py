# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   01-11-2017 11:42:39
@Last Modified by:   rlane
@Last Modified time: 16-11-2017 15:11:51
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import h5py
from tqdm import tqdm

from skimage.transform import AffineTransform, warp
from skimage.graph import route_through_array
from skimage.measure import label

from .algorithms import (condsum, detect_features, find_matches,
                         estimate_transform, estimate_translation,
                         generate_costs)
from .mosaic_data import OdemisMosaicData


class Mosaic(object):
    """docstring for Mosaic"""
    def __init__(self, mosaic_data, tag=None):
        super(Mosaic, self).__init__()

        self.keys = mosaic_data.keys
        self.shape = mosaic_data.shape
        self.sorted_keys = mosaic_data.sorted_keys

        if isinstance(mosaic_data, OdemisMosaicData):
            try:
                self.imgs = mosaic_data.imgs[tag]
            except KeyError as exc:
                msg = ("`tag` must be given when loading from "
                       "OdemisMosaicData`.")
                raise KeyError(msg) from exc
        else:
            self.imgs = mosaic_data.imgs

        self.Nx, self.Ny = self.shape
        print(list(self.imgs.keys()))
        self.nx, self.ny = self.imgs[self.keys[0]].shape[::-1]

        # self.match_kws =
        # self.ORB_kws =
        # self.ransac_kws =
        # self.FFT_kws =

    def get_translations(self, method='FFT', match_kws=None,
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
                    print(shift)
                else:
                    raw_shift = estimate_translation(img1, img2,
                                                     match_kws=match_kws,
                                                     FFT_kws=FFT_kws)
                    # print(raw_shift)
                    shift = np.array([self.nx - raw_shift[0], -raw_shift[1]])
                    print(shift)

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
                    print(shift)
                else:
                    raw_shift = estimate_translation(img1, img2,
                                                     match_kws=match_kws,
                                                     FFT_kws=FFT_kws)
                    # print(raw_shift)
                    shift = np.array([-raw_shift[0], self.ny - raw_shift[1]])
                    print(shift)

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
        if not hasattr(self, 'translations'):
            self.translations = self.get_translations()

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

        warpeds = {}
        masks = {}

        for i, k in enumerate(self.sorted_keys.flatten()):
            img = self.imgs[k]
            transform = self.transforms[k]

            warped = warp(img, transform.inverse, order=3,
                          output_shape=self.mshape, cval=-1)

            mask = (warped != -1)
            warped[~mask] = 0

            warpeds[k] = warped
            masks[k] = mask

        return warpeds, masks

    def get_costs(self):
        """
        """
        warpeds, masks = self.get_warped_images()

        h_costs = []
        for col in self.sorted_keys.T:
            for i, (k1, k2) in enumerate(zip(col, col[1:])):

                costs = generate_costs(
                    np.abs(warpeds[k2] - warpeds[k1]),
                           masks[k1] & masks[k2])
                h_costs.append(costs)

        v_costs = []
        for row in self.sorted_keys:
            for i, (k1, k2) in enumerate(zip(row, row[1:])):

                costs = generate_costs(
                    np.abs(warpeds[k2] - warpeds[k1]),
                           masks[k1] & masks[k2])
                v_costs.append(costs)

        h_costs = np.array(h_costs).reshape(
            self.Nx, self.Ny-1, self.mny, self.mnx)
        h_costs = h_costs.swapaxes(0, 1)

        v_costs = np.array(v_costs).reshape(
            self.Ny, self.Nx-1, self.mny, self.mnx)
        v_costs = v_costs.swapaxes(0, 1)

        return h_costs, v_costs

    def get_minimum_cost_paths(self):
        """
        """
        h_costs, v_costs = self.get_costs()

        self.h_mcps = []
        h_mcp_masks = []

        for i in range(self.Ny - 1):
            costs = condsum(*h_costs[i,:,:,:])
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
            costs = condsum(*v_costs[i,:,:,:])
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
        mcp_masks = {}
        for i, krow in enumerate(self.sorted_keys):
            for j, k in enumerate(krow):
                mask_val = (self.Nx*self.Ny - 1) - (self.Nx*i + j)
                mcp_masks[k] = np.where(cum_mask == mask_val, 1, 0)

        return mcp_masks

    def tile_images(self):
        """
        """
        warpeds, _ = self.get_warped_images()
        mcp_masks = self.get_minimum_cost_paths()

        patches = {}
        for krow in self.sorted_keys:
            for k in krow:
                patches[k] = np.where(
                    mcp_masks[k], warpeds[k], 0)

        self.stitched = np.sum(list(patches.values()), axis=0)
        return self.stitched

    def crude_tile(self):
        """
        """
        warpeds, masks = self.get_warped_images()


        warpeds_stitched = np.sum(list(warpeds.values()), axis=0)
        masks_stitched = np.sum(list(masks.values()), axis=0)

        stitched_norm = np.true_divide(warpeds_stitched, masks_stitched,
                                       out=np.zeros_like(warpeds_stitched),
                                       where=(masks_stitched != 0))
        self.crude_stitch = stitched_norm

        borders = []
        for k, mask in masks.items():
            x, y = np.where(mask == 1)
            x1, y1, x2, y2 = (x.min(), y.min(), x.max(), y.max())
            ledge = np.vstack((np.ones(y2 - y1) * x1, np.arange(y1, y2))).T
            redge = np.vstack((np.ones(y2 - y1) * x2, np.arange(y1, y2))).T
            uedge = np.vstack((np.arange(x1, x2), np.ones(x2 - x1) * y1)).T
            bedge = np.vstack((np.arange(x1, x2), np.ones(x2 - x1) * y2)).T
            borders += ledge, redge, uedge, bedge
        self.crude_borders = borders

        return self.crude_stitch

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
        plt.show()

    def preview(self):
        """
        """
        fig, axes = plt.subplots(*self.shape[::-1])

        for k_row, ax_row in zip(self.sorted_keys, axes):
            for k, ax in zip(k_row, ax_row):
                ax.imshow(self.imgs[k])
                ax.axis('off')
        plt.show()

    def plot_crude_stitch(self, stitch_lines=False):
        """
        """
        fig, ax = plt.subplots()
        ax.imshow(self.crude_stitch)

        if stitch_lines:
            for border in self.crude_borders:
                ax.plot(border[:, 1], border[:, 0], lw=1, color='#FFFFFF')
        plt.show()

    def plot_translations(self):
        """
        """
        X = self.translations[:,:,0].flatten()
        Y = self.translations[:,:,1].flatten()

        fig, ax = plt.subplots()
        ax.plot(X, Y, 'x-', ms=15, mew=3)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        plt.show()

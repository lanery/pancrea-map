# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   31-10-2017 14:22:46
@Last Modified by:   rlane
@Last Modified time: 01-11-2017 11:04:47
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.graph import route_through_array
from skimage.measure import label

def condsum(*arrs, r=1):
    if len(arrs) == 1:
        return arrs[0]
    else:
        a = condsum(*arrs[1:], r=r)
        return np.where(a==r, arrs[0], a)

# Beware of when to use Nx and (Nx - 1) !!!!!!!!!!!!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
im_shp = (800, 600)
gr_shp = (6, 5)
sq_shp = (160, 30)

nx, ny = im_shp
Nx, Ny = gr_shp
sx, sy = sq_shp

costs_img = np.ones(im_shp[::-1])
h_costs_img = np.ones_like(costs_img)
v_costs_img = np.ones_like(costs_img)

h_costs_dict = {}
v_costs_dict = {}

for i in range(Nx - 1):
    ix1 = i*140 + 30
    ix2 = ix1 + sx
    for j in range(Ny - 1):
        iy1 = j*120 + i*3 + 90
        iy2 = iy1 + sy

        h_costs_img[iy1:iy2, ix1:ix2] = np.random.rand(sy, sx) / 2

        h_costs_dict[j, i] = np.ones_like(h_costs_img)
        h_costs_dict[j, i][iy1:iy2, ix1:ix2] = np.random.rand(sy, sx) / 2

for i in range(Nx - 1):
    ix1 = i*140 + 90
    for j in range(Ny - 1):
        ix1 = ix1 + j*6
        ix2 = ix1 + sy

        iy1 = j*120 + 40
        iy2 = iy1 + sx

        v_costs_img[iy1:iy2, ix1:ix2] = np.random.rand(sx, sy) / 2

        v_costs_dict[i, j] = np.ones_like(h_costs_img)
        v_costs_dict[i, j][iy1:iy2, ix1:ix2] = np.random.rand(sx, sy) / 2


# Make cost arrays from cost dictionaries
h_costs = np.array(list(h_costs_dict.values())).reshape(Nx-1, Ny-1, ny, nx)
h_costs = np.swapaxes(h_costs, 0, 1)

v_costs = np.array(list(v_costs_dict.values())).reshape(Nx-1, Ny-1, ny, nx)


# Apply condsum to generate input for `route_through_array`
# and apply route_through_array
# and make mcp masks
h_mcp_masks = []
for i in range(Ny - 1):
    costs = condsum(*h_costs[i,:,:,:])

    costs[:, 0] = 0
    costs[:, -1] = 0

    # THIS IS WHERE Ny AND Nx BEING OFF BY 1 MAKES A HUGE DIFFERENCE
    mask_pts = [[ny * (i+1) // Ny, 0],
                [ny * (i+1) // Ny, nx - 1]]

    mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                 fully_connected=True)
    mcp = np.array(mcp)

    mcp_mask = np.zeros(im_shp[::-1])
    mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
    mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)

    h_mcp_masks.append(mcp_mask)


v_mcp_masks = []
for i in range(Nx - 1):
    costs = condsum(*v_costs[i,:,:,:])

    costs[0, :] = 0
    costs[-1, :] = 0

    # THIS IS WHERE Ny AND Nx BEING OFF BY 1 MAKES A HUGE DIFFERENCE
    mask_pts = [[0, nx * (i+1) // Nx],
                [ny - 1, nx * (i+1) // Nx]]

    mcp, _ = route_through_array(costs, mask_pts[0], mask_pts[1],
                                 fully_connected=True)
    mcp = np.array(mcp)

    mcp_mask = np.zeros(im_shp[::-1])
    mcp_mask[mcp[:, 0], mcp[:, 1]] = 1
    mcp_mask = (label(mcp_mask, connectivity=1, background=-1) == 1)

    v_mcp_masks.append(mcp_mask)


h_mcp_masks = np.array(h_mcp_masks)
v_mcp_masks = np.array(v_mcp_masks)

# ALSO HERE
mcp_masks = {}
for i in range(Ny):
    cum_mask = Nx*h_mcp_masks.sum(axis=0) + v_mcp_masks.sum(axis=0)

    for j in range(Nx):

        mcp_masks[i, j] = np.where(cum_mask == (Nx*i + j), 1, 0)

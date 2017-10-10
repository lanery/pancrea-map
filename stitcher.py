# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   27-09-2017 09:39:33
@Last Modified by:   rlane
@Last Modified time: 10-10-2017 11:46:30
"""

import os
import numpy as np
from glob import glob
import matplotlib.pyplot as plt; plt.rcParams['image.cmap'] = 'magma'
import matplotlib.gridspec as gridspec

import stitching
import stitching_utils
import odemis_utils


#-----------------+
# Load image data |
#-----------------+
ome_files = sorted(glob('test_images/test*.ome.tiff'))
im_dict = {}

for ome_file in ome_files:
    k = os.path.basename(ome_file).split('.')[0]
    im_dict[k] = stitching_utils.load_tiff(ome_file)

EM_imgs = {}
FM_imgs = {}

for k, d in im_dict.items():
    EM_imgs[k] = d['EM'].asarray()
    FM_imgs[k] = d['FM'].asarray()
    im_dict[k]['x_pos'] = d['EM'].tags['x_position'].value[0]
    im_dict[k]['y_pos'] = d['EM'].tags['y_position'].value[0]


# img1 = odemis_utils.auto_bc(FM_imgs['test2'])
# img2 = odemis_utils.auto_bc(FM_imgs['test4'])

# model_robust = stitching._estimate_transform(
#     img1, img2, ORB_kws={'downscale': 2})

# stitched = stitching._stitch(img1, img2, model_robust)

base_img = odemis_utils.auto_bc(FM_imgs['test2'])

for label, img in FM_imgs.items():
    if label != 'test2':
        print(label)

        stitched = base_img

        img_bc = odemis_utils.auto_bc(img)
        model_robust = stitching._estimate_transform(
            stitched, img_bc, ORB_kws={'downscale': 2})

        stitched = stitching._stitch(stitched, img_bc, model_robust)

        plt.figure()
        plt.imshow(stitched)

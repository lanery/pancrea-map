# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   27-09-2017 09:39:33
@Last Modified by:   rlane
@Last Modified time: 05-10-2017 18:24:45
"""

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
ome_files = sorted(glob('test*.ome.tiff'))
im_dict = {}

for ome_file in ome_files:
    k = ome_file.split('.')[0]
    im_dict[k] = stitching_utils.load_tiff(ome_file)

EM_imgs = {}
FM_imgs = {}

for k, d in im_dict.items():
    EM_imgs[k] = d['EM'].asarray()
    FM_imgs[k] = d['FM'].asarray()
    im_dict[k]['x_pos'] = d['EM'].tags['x_position'].value[0]
    im_dict[k]['y_pos'] = d['EM'].tags['y_position'].value[0]

# compare(EM_imgs, figsize=(14, 4))
# compare(FM_imgs, figsize=(14, 4))

img1 = odemis_utils.auto_bc(FM_imgs['test3'])
img2 = odemis_utils.auto_bc(FM_imgs['test4'])


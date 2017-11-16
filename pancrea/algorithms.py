# -*- coding: utf-8 -*-
"""
@Author: rlane
@Date:   08-11-2017 11:47:55
@Last Modified by:   rlane
@Last Modified time: 16-11-2017 15:51:04
"""

import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import (ORB, match_descriptors, register_translation,
                             plot_matches)
from skimage.transform import AffineTransform
from skimage.measure import ransac, label


def round_to_base(n, base):
    """
    """
    return int(base * round(float(n) / base))


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

            if 'h' in match_kws['direction']:
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


def estimate_translation(img1, img2, match_kws=None, FFT_kws=None):
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
        'space': 'real'}

    FFT_kws = {**default_FFT_kws, **FFT_kws}

    if match_kws is None:
        shifts, error, phase_difference = register_translation(
            img1, img2, **FFT_kws)

    else:
        try:
            m, n = img1.shape
            o1 = 1 / (1 - (match_kws['overlap'] / 100))
            o2 = 1 / (match_kws['overlap'] / 100)

            if 'h' in match_kws['direction']:
                img1 = img1[:, n - int(n/o2):]
                img2 = img2[:, :int(n/o2)]

                shifts, error, phase_difference = register_translation(
                    img1, img2, **FFT_kws)

            else:  # assume images are stacked vertically
                img1 = img1[m - int(m/o2):, :]
                img2 = img2[:int(m/o2), :]

                shifts, error, phase_difference = register_translation(
                    img1, img2, **FFT_kws)

        except KeyError as exc:
            msg = "`match_kws` must contain {}.".format(exc)
            raise KeyError(msg)

    return -shifts[::-1]


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


def display_matches(img1, img2, remove_outliers=True, match_kws=None,
                    ORB_kws=None, ransac_kws=None):
    """
    """
    ORB_kws = {} if ORB_kws is None else ORB_kws
    ransac_kws = {} if ransac_kws is None else ransac_kws

    kps1, kps2, matches = find_matches(img1, img2, match_kws=match_kws,
                                       ORB_kws=ORB_kws)

    src = kps2[matches[:, 1]][:, ::-1]
    dst = kps1[matches[:, 0]][:, ::-1]

    default_ransac_kws = {
        'min_samples': 5,
        'residual_threshold': 10,
        'max_trials': 5000}

    ransac_kws = {**default_ransac_kws, **ransac_kws}
    model, inliers = ransac((src, dst), AffineTransform, **ransac_kws)

    fig, ax = plt.subplots()

    if remove_outliers:
        plot_matches(ax, img1, img2, kps1, kps2,
                     matches[inliers], '#09BB62', '#00F67A')
    else:
        plot_matches(ax, img1, img2, kps1, kps2,
                     matches, '#09BB62', '#00F67A')

    return model.translation


def display_FFT_match(img1, img2, match_kws=None, FFT_kws=None):
    """
    """
    from matplotlib.patches import Rectangle

    shift = estimate_translation(
        img1, img2, match_kws=match_kws, FFT_kws=FFT_kws)

    y, x = img1.shape
    dx, dy = shift

    img = np.concatenate((img1, img2), axis=1)

    fig, ax = plt.subplots()
    ax.imshow(img)

    x_overlap = Rectangle((x-dx, 0), 2*dx, y, fc='red', alpha=0.2)
    y_overlap1 = Rectangle((0, y-dy), x, dy, fc='red', alpha=0.2)
    y_overlap2 = Rectangle((x, 0), x, dy, fc='red', alpha=0.2)

    ax.add_patch(x_overlap)
    ax.add_patch(y_overlap1)
    ax.add_patch(y_overlap2)

    return shift


def display_cross_correlation(img1, img2, match_kws=None, FFT_kws=None):
    """
    """
    FFT_kws = {} if FFT_kws is None else FFT_kws
    default_FFT_kws = {
        'upsample_factor': 1,
        'space': 'real'}

    ny, nx = img1.shape

    if match_kws is None:
        shifts, error, phase_difference = register_translation(
            img1, img2, **FFT_kws)
    else:
        try:
            m, n = img1.shape
            o1 = 1 / (1 - (match_kws['overlap'] / 100))
            o2 = 1 / (match_kws['overlap'] / 100)

            if 'h' in match_kws['direction']:
                img1 = img1[:, n - int(n/o2):]
                img2 = img2[:, :int(n/o2)]

                shifts, error, phase_difference = register_translation(
                    img1, img2, **FFT_kws)

            else:  # assume images are stacked vertically
                img1 = img1[m - int(m/o2):, :]
                img2 = img2[:int(m/o2), :]

                shifts, error, phase_difference = register_translation(
                    img1, img2, **FFT_kws)

        except KeyError as exc:
            msg = "`match_kws` must contain {}.".format(exc)
            raise KeyError(msg)

    fig = plt.figure(figsize=(11, 4))
    ax1 = plt.subplot(1, 3, 1, adjustable='box-forced')
    ax2 = plt.subplot(1, 3, 2, sharex=ax1, sharey=ax1, adjustable='box-forced')
    ax3 = plt.subplot(1, 3, 3)

    ax1.imshow(img1)
    ax1.set_title('Reference')

    ax2.imshow(img2)
    ax2.set_title('Target')

    # Show the output of a cross-correlation to show what the algorithm is
    # doing behind the scenes
    image_product = np.fft.fft2(img1) * np.fft.fft2(img2).conj()
    cc_image = np.fft.fftshift(np.fft.ifft2(image_product))
    ax3.imshow(cc_image.real)
    ax3.set_title("Cross-correlation")

    shift = -shifts[::-1]
    return shift

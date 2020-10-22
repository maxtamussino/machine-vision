#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    edges_in /= np.amax(edges_in)
    edges_sorted = np.sort(edges_in[edges_in != 0].ravel())
    pixel_num = len(edges_sorted)

    low = edges_sorted[round(pixel_num * (1 - low_prop))]
    high = edges_sorted[round(pixel_num * (1 - high_prop))]
    print("Hysteresis calculated with...\n   High: {:.2f}\n   Low: {:.2f}".format(high, low))

    hyst_out = hyst_thresh(edges_in, low, high)
    ######################################################
    return hyst_out

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################

    # Normalise and remove pixels under low threshold
    edges_in = edges_in / np.amax(edges_in)
    edges_in = np.where(edges_in > low, edges_in, 0)

    # Find connected edges
    labels_num, labels = cv2.connectedComponents((edges_in * 255).astype(np.uint8))
    bitwise_img = np.zeros(edges_in.shape).astype(np.float32)

    # For every edge, check if one pixel is above higher threshold
    for i in range(1, labels_num):
        edge = labels == i
        if np.logical_and(edge, edges_in > high).any():
            bitwise_img += np.where(edge, 1., 0.)
    ######################################################
    return bitwise_img

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    width, height = gradients.shape
    edges = np.zeros(gradients.shape).astype("float32")

    for idx, orientation in np.ndenumerate(orientations):
        pi = np.pi

        # Check orientation and set relative index
        if abs(orientation) <= pi * 1/8 or abs(orientation) >= pi * 7/8:
            idx_rel = np.array([1, 0])  # West or East
        elif pi * 5/8 >= abs(orientation) >= pi * 3/8:
            idx_rel = np.array([0, 1])  # North or South
        else:  # Diagonal
            if orientation < 0:
                orientation += pi
            if orientation <= pi * 3/8:
                idx_rel = np.array([1, 1])  # Northeast or Southwest
            else:
                idx_rel = np.array([1, -1])  # Northwest or Southeast

        # Add and subtract relative index (both directions!)
        idx1 = tuple(np.array(idx) + idx_rel)
        idx2 = tuple(np.array(idx) - idx_rel)

        # Check for image boundaries and skip, if neighbour has higher value
        if not idx1[0] == -1 and not idx1[1] == -1:
            if not idx1[0] == width and not idx1[1] == height:
                if gradients[idx1] > gradients[idx]:
                    continue
        if not idx2[0] == -1 and not idx2[1] == -1:
            if not idx2[0] == width and not idx2[1] == height:
                if gradients[idx2] > gradients[idx]:
                    continue
        edges[idx] = gradients[idx]
    ######################################################

    return edges

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
    width,height = edges_in.shape
    bitwise_img = np.zeros(edges_in.shape).astype("float32")
    edges_in /= np.amax(edges_in)

    queue = []
    for idx,edge_prop in np.ndenumerate(edges_in):
        if edge_prop > high:
            queue.append(idx)
            bitwise_img[idx] = 1

    for idx in queue:
        for x in range(idx[0]-1,idx[0]+2):
            for y in range(idx[1]-1,idx[1]+2):
                if x == -1 or x == width:
                    continue
                if y == -1 or y == height:
                    continue
                if not (x,y) in queue and edges_in[x][y] >= low:
                    queue.append((x,y))
                    bitwise_img[x][y] = 1

    ######################################################
    return bitwise_img

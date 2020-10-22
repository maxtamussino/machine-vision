#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np
from numpy.core._multiarray_umath import ndarray


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
    pi = np.pi
    """
    Each pixel has eight neighbours. The neighbours are in the directions:
    pi * 1/4 * n, 0 <= n <= 7
    Because there is only one neighbour considered, no interpolation between
    two neighbours is performed. The correct neighbour is chosen by looking 
    at the orientation. Each neighbour is assigned one range of orientations:
    pi * 1/4 * n +- pi * 1/8, 0 <= n <= 7
    Because each value has to be compared to two neighbour values, and these are
    always in opposite directions, only half (four
    whole range from -pi to pi is divided into eight sections, each 
    """

    # Create shifted images in every of the eight neighbour's directions
    north = np.roll(gradients, 1, 0)
    south = np.roll(gradients, -1, 0)
    west = np.roll(gradients, -1, 1)
    east = np.roll(gradients, 1, 1)
    northwest = np.roll(gradients, (1, -1), (0, 1))
    northeast = np.roll(gradients, (1, 1), (0, 1))
    southwest = np.roll(gradients, (-1, -1), (0, 1))
    southeast = np.roll(gradients, (-1, 1), (0, 1))

    # Create absolute and shifted orientations
    orientations_abs = np.abs(orientations)
    orientations_shift = np.where(orientations < 0, orientations + pi, orientations)

    # Case north/south
    north_south = np.logical_and(orientations_abs >= pi * 3 / 8, orientations_abs <= pi * 5 / 8)
    north_south = np.logical_and(np.logical_and(north_south, gradients > north), gradients > south)
    north_south = np.where(north_south, gradients, 0)

    # Case northeast/southwest
    northeast_southwest = np.logical_and(orientations_shift >= pi * 1 / 8, orientations_shift <= pi * 3 / 8)
    northeast_southwest = np.logical_and(np.logical_and(northeast_southwest, gradients > northeast), gradients > southwest)
    northeast_southwest = np.where(northeast_southwest, gradients, 0)

    # Case east/west
    east_west = np.logical_or(orientations_abs <= pi * 1 / 8, orientations_abs >= pi * 7 / 8)
    east_west = np.logical_and(np.logical_and(east_west, gradients > east), gradients > west)
    east_west = np.where(east_west, gradients, 0)

    # Case southeast/northwest
    southeast_northwest = np.logical_and(orientations_shift >= pi * 5 / 8, orientations_shift <= pi * 7 / 8)
    southeast_northwest = np.logical_and(np.logical_and(southeast_northwest, gradients > southeast), gradients > northwest)
    southeast_northwest = np.where(southeast_northwest, gradients, 0)

    edges = north_south + northeast_southwest + east_west + southeast_northwest
    ######################################################

    return edges

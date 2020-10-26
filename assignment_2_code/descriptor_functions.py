#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Descriptor functions

Author: FILL IN
MatrNr: FILL IN
"""
from typing import Callable

import numpy as np
import cv2

from helper_functions import circle_mask


def patch_basic(patch: np.ndarray) -> np.ndarray:
    """ Return the basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################


def patch_norm(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################


def patch_sort(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################


def patch_sort_circle(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, patch.size))  # Replace this line

    ######################################################


def block_orientations(patch: np.ndarray) -> np.ndarray:
    """ Compute orientation-histogram based descriptor from a patch

    Orientation histograms from 16 4 x 4 blocks of the patch, concatenated in row major order (1 x 128).
    Each orientation histogram should consist of 8 bins in the range [-pi, pi], each bin being weighted by the sum of
    gradient magnitudes of pixel orientations assigned to that bin.

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (16, 16)

    :return: Orientation-histogram based Descriptor
    :rtype: np.ndarray with shape (1, 128)
    """
    ######################################################
    # Write your own code here
    return np.zeros((1, 128))  # Replace this line

    ######################################################


def compute_descriptors(descriptor_func: Callable,
                        img: np.ndarray,
                        locations: np.ndarray,
                        patch_size: int) -> (np.ndarray, np.ndarray):
    """ Calculate the given descriptor using descriptor_func on patches of the image, centred on the locations provided

    :param descriptor_func: Descriptor to compute at each location
    :type descriptor_func: function

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param locations: Locations at which to compute the descriptors (n x 2)
    :type locations: np.ndarray with the shape (n x 2). First column is y (row), second is x (column).

    :param patch_size: Value defining the width and height of the patch around each location
        to pass to the descriptor function.
    :type patch_size: int

    :return: (interest_points, descriptors):
        interest_points: k x 2 array containing the image coordinates [y,x] of the corners.
            Locations too close to the image boundary to cut out the image patch should not be contained.
        descriptors: k x patch_size^2 matrix containing the patch descriptors.
            Each row vector stores the concatenated column vectors of the image patch around each corner.
            Corners too close to the image boundary to cut out the image patch should not be contained.
    :rtype: (np.ndarray, np.ndarray)
    """
    ######################################################
    # Write your own code here
    return np.zeros((10, 2)), np.zeros((10, patch_size*patch_size))  # Replace this line

    ######################################################

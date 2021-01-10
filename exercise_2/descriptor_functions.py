#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Descriptor functions

Author: Max Tamussino
MatrNr: 01611815
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
    # Create simple descriptor only containing patch values
    descriptor = patch.flatten()
    return descriptor
    ######################################################


def patch_norm(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Normalise the basic patch descriptor
    descriptor = patch_basic(patch) / np.max(patch)
    return descriptor
    ######################################################


def patch_sort(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Sort the normalised patch descriptor
    descriptor = np.sort(patch_norm(patch))
    return descriptor
    ######################################################


def patch_sort_circle(patch: np.ndarray) -> np.ndarray:
    """ Return the normalized and sorted basic descriptor as a vector

    :param patch: Patch of the image around a corner
    :type patch: np.ndarray with shape (patch_size, patch_size)

    :return: Descriptor
    :rtype: np.ndarray with shape (1, patch_size^2)
    """
    ######################################################
    # Only consider values inside a circle mask and use sorted descriptor
    descriptor = np.where(circle_mask(patch.shape[0]), patch, 0)
    descriptor = patch_sort(descriptor)
    return descriptor
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
    descriptor = np.zeros((1, 0))

    # Calculate gradients
    grad_x = np.gradient(patch, axis=0)
    grad_y = np.gradient(patch, axis=1)
    grad_mag = np.sqrt(np.square(grad_x) + np.square(grad_y))

    # Calculate orientations
    orient = np.arctan2(grad_y, grad_x)

    # Prepare bins for value characterisation
    bins = np.arange(-np.pi, np.pi, 2*np.pi/8)

    # Iterate over the blocks
    for i in range(0, 4):
        for j in range(0, 4):
            # Create sub patch and characterise content according to bins
            sub_patch_grad_mag = grad_mag[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            sub_patch_orient = orient[i * 4:(i + 1) * 4, j * 4:(j + 1) * 4]
            sub_patch_orient_binned = np.digitize(sub_patch_orient, bins)

            # Create histogram
            histogram = np.zeros((1, 8))
            for k in range(0, 8):
                # Histogram entry is the sum of gradient magnitudes found in that bin
                histogram[0, k] = np.sum(np.multiply(np.where(sub_patch_orient_binned == k, 1, 0), sub_patch_grad_mag))

            # Normalise histogram by gradient magnitude sum
            gradient_sum = np.sum(sub_patch_grad_mag)
            if gradient_sum != 0:
                histogram = histogram / gradient_sum

            # Add histogram to the descriptor
            descriptor = np.hstack((descriptor, histogram))

    return descriptor
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
    interest_points = np.zeros((0, 2))
    if descriptor_func == block_orientations:
        descriptors = np.zeros((0, 128))
    else:
        descriptors = np.zeros((0, patch_size*patch_size))

    patch_rad = int((patch_size - 1) / 2)
    right_top_addon = 0
    if patch_size % 2 == 0:
        right_top_addon = 1

    for loc in locations:
        x = int(loc[0])
        y = int(loc[1])
        if x < patch_rad or x > img.shape[0] - patch_rad - right_top_addon:
            # print("skipped x={} y={}".format(x, y))
            continue
        if y < patch_rad or y > img.shape[1] - patch_rad - right_top_addon:
            # print("skipped x={} y={}".format(x, y))
            continue
        patch = img[x - patch_rad:x + patch_rad + 1 + right_top_addon,
                    y - patch_rad:y + patch_rad + 1 + right_top_addon]
        descriptor = descriptor_func(patch)
        interest_points = np.vstack((interest_points, [x, y]))
        descriptors = np.vstack((descriptors, descriptor))

    return interest_points, descriptors
    ######################################################

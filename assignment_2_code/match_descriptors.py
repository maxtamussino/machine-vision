#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Match descriptors in two images

Author: Max Tamussino
MatrNr: 01611815
"""
import numpy as np
import cv2


def match_descriptors(descriptors_1: np.ndarray, descriptors_2: np.ndarray, best_only: bool) -> np.ndarray:
    """ Find matches for patch descriptors

    :param descriptors_1: Patch descriptors of first image
    :type descriptors_1: np.ndarray with shape (m, n) containing m descriptors of length n

    :param descriptors_2: Patch descriptor of second image
    :type descriptors_2: np.ndarray with shape (m, n) containing m descriptors of length n

    :param best_only: If True, only keep the best match for each descriptor
    :type best_only: Boolean

    :return: Array representing the successful matches. Each row contains the indices of the matches descriptors
    :rtype: np.ndarray with shape (k, 2) with k being the number of matches
    """
    ######################################################
    # Empty array of matches
    matches = np.zeros((0, 2)).astype(int)

    # Keep track of index
    index = -1
    for desc in descriptors_1:
        # Index tracking
        index += 1

        # Calculate distance from the descriptors of image 2
        distances = np.linalg.norm(descriptors_2 - desc, axis=1)

        # Save the best value, its index and the second best value
        best_index = distances.argmin()
        best_val = distances[best_index]
        secondbest_val = min(np.delete(distances, best_index))

        # Only add to matches if best << second best
        if best_val / secondbest_val < 0.8:
            matches = np.vstack((matches, [index, best_index]))

    print("Image one: {} corners".format(descriptors_1.shape[0]))
    print("Image two: {} corners".format(descriptors_2.shape[0]))
    print("Matches: {}".format(matches.shape[0]))
    return matches
    ######################################################

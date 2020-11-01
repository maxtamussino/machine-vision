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
    index1 = -1
    for descr1 in descriptors_1:
        # Index tracking
        index1 += 1
        index2 = -1

        # Save best & second best match (and index for the best)
        best_index = -1
        best_val = 1000
        secondbest_val = 1000

        for descr2 in descriptors_2:
            # Index tracking
            index2 += 1

            # Calculate distance
            new_val = np.linalg.norm(descr1 - descr2)

            # Compare distance to best & second best
            if new_val < best_val:
                secondbest_val = best_val
                best_val = new_val
                best_index = index2
            elif new_val < secondbest_val:
                secondbest_val = new_val

        # Only add to matches if best << second best
        if best_val / secondbest_val < 0.8:
            matches = np.vstack((matches, [index1, best_index]))

    print("Image one: {} corners".format(descriptors_1.shape[0]))
    print("Image two: {} corners".format(descriptors_2.shape[0]))
    print("Matches: {}".format(matches.shape[0]))
    return matches
    ######################################################

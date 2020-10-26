#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Match descriptors in two images

Author: FILL IN
MatrNr: FILL IN
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
    # Write your own code here
    return np.zeros((10, 2))  # Replace this line

    ######################################################

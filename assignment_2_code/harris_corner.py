#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: FILL IN
MatrNr: FILL IN
"""
import numpy as np
import cv2


def harris_corner(img, sigma1, sigma2, k, threshold):
    """ Detect corners using the Harris corner detector

    :param img: Grayscale input image
    :type img: np.ndarray with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma1: Sigma for the first Gaussian filtering
    :type sigma1: float

    :param sigma2: Sigma for the second Gaussian filtering
    :type sigma2: float

    :param k: Coefficient for harris formula
    :type k: float

    :param threshold: corner threshold
    :type threshold: float

    :return: (i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners):
        i_xx: squared input image filtered with derivative of gaussian in x-direction
        i_yy: squared input image filtered with derivative of gaussian in y-direction
        i_xy: Multiplication of input image filtered with derivative of gaussian in x- and y-direction
        g_xx: i_xx filtered by larger gaussian
        g_yy: i_yy filtered by larger gaussian
        g_xy: i_xy filtered by larger gaussian
        h_dense: Result of harris calculation for every pixel. Array of same size as input image.
            Values normalized to 0-1
        h_nonmax: Binary mask of non-maxima suppression. Array of same size as input image.
            1 where values are NOT suppressed, 0 where they are.
        corners: n x 3 array containing all detected corners after thresholding and non-maxima suppression.
            Every row vector represents a corner with the elements [y, x, d]
            (d is the result of the harris calculation)
    :rtype: (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray)

    """

    ######################################################
    # Write your own code here
    output = np.ones(img.shape)  # Replace these lines
    return output, output, output, output, output, output, output, output, output

    ######################################################

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Corner detection with the Harris corner detector

Author: Max Tamussino
MatrNr: 01611815
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
    # Create first gaussian kernel with sigma1
    kernel_width1 = 2 * round(3 * sigma1) + 1
    kernel = cv2.getGaussianKernel(kernel_width1, sigma1)  # (kernel_width x 1)
    gauss1 = np.outer(kernel, kernel.transpose())  # (kernel_width x kernel_width)

    # Apply blurring to input image
    blurred = cv2.filter2D(img, -1, gauss1)

    # Compute derivative and squares/products needed
    i_x = np.gradient(blurred, axis=0)
    i_y = np.gradient(blurred, axis=1)
    i_xy = np.multiply(i_x, i_y)
    i_xx = np.square(i_x)
    i_yy = np.square(i_y)

    # Create second gaussian kernel with sigma2
    kernel_width2 = 2 * round(3 * sigma2) + 1
    kernel = cv2.getGaussianKernel(kernel_width2, sigma2)  # (kernel_width x 1)
    gauss2 = np.outer(kernel, kernel.transpose())  # (kernel_width x kernel_width)

    # Apply blurring to the derivatives
    g_xx = cv2.filter2D(i_xx, -1, gauss2)
    g_yy = cv2.filter2D(i_yy, -1, gauss2)
    g_xy = cv2.filter2D(i_xy, -1, gauss2)

    # Compute value R for all entries: det(M)-k*trace(M)^2
    # Note: M = ((g_xx, g_xy), (g_xy, g_yy))
    # Note: det(M) = g_xx * g_yy - g_xy * g_xy
    # Note: trace(M) = g_xx + g_yy
    h_dense = np.multiply(g_xx, g_yy) - np.square(g_xy) - k * np.square(g_xx + g_yy)
    h_dense /= np.max(h_dense)
    h_dense = np.where(h_dense >= threshold, h_dense, 0)

    # Create neighbour arrays to compare
    north = np.roll(h_dense, 1, 0)
    south = np.roll(h_dense, -1, 0)
    west = np.roll(h_dense, -1, 1)
    east = np.roll(h_dense, 1, 1)
    northwest = np.roll(h_dense, (1, -1), (0, 1))
    northeast = np.roll(h_dense, (1, 1), (0, 1))
    southwest = np.roll(h_dense, (-1, -1), (0, 1))
    southeast = np.roll(h_dense, (-1, 1), (0, 1))

    # Decision tree: where h_dense is greater than all neighbours, h_nonmax is 1
    inter_1 = np.logical_and(h_dense >= north, h_dense > south)
    inter_2 = np.logical_and(h_dense >= east, h_dense > west)
    inter_3 = np.logical_and(h_dense >= northeast, h_dense > southwest)
    inter_4 = np.logical_and(h_dense >= southeast, h_dense > northwest)
    inter_5 = np.logical_and(inter_1, inter_2)
    inter_6 = np.logical_and(inter_3, inter_4)
    inter_7 = np.logical_and(inter_5, inter_6)
    h_nonmax = np.where(inter_7, 1, 0)

    # Filter coordinates where h_nonmax != 0
    coordinates = np.argwhere(h_nonmax)

    # Initialise array with additional column
    corners = np.zeros((coordinates.shape[0], 3))

    # Assign first two columns with coordinates and third with value of R
    corners[:, [0, 1]] = coordinates
    corners[:, 2] = h_dense[coordinates[:, 0], coordinates[:, 1]]

    # Print number of found corners
    print(coordinates.shape[0])

    return i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners
    ######################################################

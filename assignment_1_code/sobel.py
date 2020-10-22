#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    # Initializing sobel kernels (horizontal and vertical)
    kernel_h = np.array([[-1,0,1],
                         [-2,0,2],
                         [-1,0,1]]).astype("float32")
    kernel_v = kernel_h.transpose()

    # Empty x and y results
    sobel_x = np.zeros(img.shape).astype("float32")
    sobel_y = np.zeros(img.shape).astype("float32")

    # Fill in results from horizontal and vertical edges
    cv2.filter2D(img, -1, kernel_h, sobel_x, borderType=cv2.BORDER_REPLICATE)
    cv2.filter2D(img, -1, kernel_v, sobel_y, borderType=cv2.BORDER_REPLICATE)

    # Calculate total gradient and its direction
    gradient = np.sqrt(np.square(sobel_x) + np.square(sobel_y))
    orientation = np.arctan2(sobel_y, sobel_x)
    ######################################################
    return gradient, orientation

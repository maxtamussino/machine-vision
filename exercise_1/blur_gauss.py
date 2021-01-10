#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: Max Tamussino
MatrNr: 01611815
"""

import cv2
import numpy as np

def blur_gauss(img: np.array, sigma: float, kernel_width: int = -1) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :param kernel_width: Optional parameter to manually set the kernel width (odd number)
    :type kernel_width: int

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################
    # Blurred image initialized using original image shape
    img_blur = np.zeros(img.shape).astype("float32")

    # Kernel size is determined
    if kernel_width == -1 or kernel_width % 2 != 1:
        kernel_width = 2 * round(3 * sigma) + 1

    # Offset is used to transform coordinate systems:
    # 0,0 1,0 2,0      -1,-1  0,-1  1,-1
    # 0,1 1,1 2,1  ->  -1, 0  0, 0  1, 0
    # 0,2 1,2 2,2      -1, 1  0, 1  1, 1
    # Array Index      Gauss Calculation
    kernel_coord_offset = int(np.floor((kernel_width-1)/2))

    # The kernel is initialized
    kernel = np.zeros(shape=(kernel_width,kernel_width)).astype("float32")

    # Kernel calculation
    for x, row in enumerate(kernel):
        i = x - kernel_coord_offset
        for y, elem in enumerate(row):
            j = y - kernel_coord_offset
            kernel[x, y] = np.exp(-(i*i+j*j)/(2*sigma*sigma))

    # Kernel normalization
    kernel /= np.sum(kernel.flatten())

    # Kernel application
    cv2.filter2D(img,-1,kernel,img_blur,borderType=cv2.BORDER_REPLICATE)

    """
    # Own loop-based implementation for kernel application (very slow/inefficient)
    # Borders are left out (black)
    for i in range(kernel_coord_offset,width-kernel_coord_offset):
        for j in range(kernel_coord_offset,height-kernel_coord_offset):
            result = 0
            for l in kernel_coord_range:
                for m in kernel_coord_range:
                    result += kernel[l+kernel_coord_offset,m+kernel_coord_offset] * img[i+l,j+m]
            img_blur[i,j] = result
    """
    ######################################################
    return img_blur

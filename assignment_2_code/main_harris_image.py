#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 2a: Harris Corners
Clara Haider, Matthias Hirschmanner 2020
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""
from pathlib import Path

import cv2
import numpy as np

from harris_corner import harris_corner
from helper_functions import show_corners

if __name__ == '__main__':
    np.set_printoptions(precision=2, suppress=True)  # for debugging
    debug_corners = False  # <<< change to reduce output when you're done
    save_image = True
    img_file = 'desk/Image-00.jpg'

    # Parameters
    sigma1 = 0.8      # 0.8  standard
    sigma2 = 1.5      # 1.5  standard
    k = 0.04          # 0.04 standard
    threshold = 0.01  # 0.01 standard

    # Read image
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath(img_file)), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))

    # Convert image from uint8 with range [0,255] to float32 with range [0,1]
    img_gray = img_gray.astype(np.float32) / 255.

    # Harris corner detector
    i_xx, i_yy, i_xy, g_xx, g_yy, g_xy, h_dense, h_nonmax, corners \
        = harris_corner(img_gray,
                        sigma1=sigma1,
                        sigma2=sigma2,
                        k=k,
                        threshold=threshold)

    # Show detected corners
    show_corners(img_gray,
                 i_xx,
                 i_yy,
                 i_xy,
                 g_xx,
                 g_yy,
                 g_xy,
                 h_dense,
                 h_nonmax,
                 corners,
                 debug_corners=debug_corners,
                 save_image=save_image)

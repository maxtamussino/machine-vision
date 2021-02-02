#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 2b: Harris Corners
Clara Haider, Matthias Hirschmanner 2020
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""
from pathlib import Path

import numpy as np
from descriptor_functions import *
from harris_corner import harris_corner
from helper_functions import *
from match_descriptors import *

if __name__ == '__main__':
    save_image = True  # Enables saving of matches image
    imageset = 3

    if imageset == 0:
        # Plant
        img_path_1 = 'plant/Image_0_1.JPG'
        img_path_2 = 'plant/Image_0_2.JPG'
    elif imageset == 1:
        # Desk
        img_path_1 = 'desk/Image-00.JPG'
        img_path_2 = 'desk/Image-01.JPG'
    elif imageset == 2:
        # Desk (exposure)
        img_path_1 = 'desk/Image-00.JPG'
        img_path_2 = 'desk/Image-01-cl.JPG'
    elif imageset == 3:
        # Desk (far)
        img_path_1 = 'desk/Image-00.JPG'
        img_path_2 = 'desk/Image-03.JPG'
    elif imageset == 4:
        # Plant (far)
        img_path_1 = 'plant/Image_0_0.JPG'
        img_path_2 = 'plant/Image_0_3.JPG'

    # Parameters
    sigma1 = 1
    sigma2 = 2
    threshold = 0.01
    k = 0.04
    patch_size = 9
    rotation = 0.0


    current_path = Path(__file__).parent
    img_gray_1 = cv2.imread(str(current_path.joinpath(img_path_1)), cv2.IMREAD_GRAYSCALE)
    if img_gray_1 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_1)))

    img_gray_2 = cv2.imread(str(current_path.joinpath(img_path_2)), cv2.IMREAD_GRAYSCALE)
    if img_gray_2 is None:
        raise FileNotFoundError("Couldn't load image " + str(current_path.joinpath(img_path_2)))

    # Rotation of the image
    if rotation > 0:
        img_gray_2 = rotate_bound(img_gray_2, rotation)

    # Convert images from uint8 with range [0,255] to float32 with range [0,1]
    img_gray_1 = img_gray_1.astype(np.float32) / 255.
    img_gray_2 = img_gray_2.astype(np.float32) / 255.

    # Choose which descriptor to use by indexing into the cell array
    descriptor_func_ind = 4

    # descriptor function names
    descriptor_funcs = [patch_basic, patch_norm, patch_sort, patch_sort_circle, block_orientations]

    # Patch size must be 16 for block_orientations
    if descriptor_func_ind == 4:
        patch_size = 16

    descriptor_func = descriptor_funcs[descriptor_func_ind]

    # Harris corner detector
    _, _, _, _, _, _, _, _, corners = harris_corner(img_gray_1, sigma1=sigma1, sigma2=sigma2, threshold=threshold, k=k)

    # Create descriptors
    interest_points_1, descriptors_1 = compute_descriptors(descriptor_func, img_gray_1, corners[:, 0:2], patch_size)

    # Harris corner detector
    _, _, _, _, _, _, _, _, corners = harris_corner(img_gray_2, sigma1=sigma1, sigma2=sigma2, threshold=threshold, k=k)

    # Create descriptors
    interest_points_2, descriptors_2 = compute_descriptors(descriptor_func, img_gray_2, corners[:, 0:2], patch_size)

    # Match descriptors
    matches = match_descriptors(descriptors_1, descriptors_2, best_only=True)

    # Display results
    show_matches(img_gray_1, img_gray_2, interest_points_1, interest_points_2, matches, save_image=save_image)

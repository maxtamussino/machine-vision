#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Different utility functions

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import open3d as o3d
import numpy as np
import cv2

from camera_params import *


def project_2d(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """ Projects a pointcloud to a 2D numpy image array

    :param pcd: The pointcloud which is to be projected
    :type pcd: o3d.geometry.PointCloud

    :return: 2D image of the pointcloud (np.uint8)
    :rtype: np.ndarray
    """

    # Retrieve point coordinates and their colours
    points = np.asarray(pcd.points)
    colours = np.asarray(pcd.colors)[..., ::-1]

    # Remove origin (z=0 -> calculation invalid)
    origin_idx = np.where(points[:, 2] == 0)[0]
    points = np.delete(points, origin_idx, 0)
    colours = np.delete(colours, origin_idx, 0)

    # Project onto image plane
    u = np.round(fx_rgb * np.divide(points[:, 0], points[:, 2]) + cx_rgb).astype(int)
    v = np.round(fy_rgb * np.divide(points[:, 1], points[:, 2]) + cy_rgb).astype(int)

    # Create image
    image = np.zeros((v.max() + 1, u.max() + 1, 3))

    # Colour each pixel according to pointcloud
    for i in range(points.shape[0]):
        image[v[i], u[i]] = colours[i]

    # Convert to np.uint8
    image = (255 * image).astype(np.uint8)

    return image


def write_hypothesis(objects: list,
                     labels_image: np.ndarray,
                     result_image: np.ndarray) -> np.ndarray:
    """ Writes text object hypothesis onto the scene image

    :param objects: List of found object hypothesis, every entry of the form:
                    [colour, class_best_score, best_score, current_score]
                    [0]    [1]               [2]         [3]
    :type objects: list

    :param labels_image: Image of the coloured object clusters
    :type labels_image: np.ndarray

    :param result_image: Image to write the hypothesis texts on
    :type result_image: np.ndarray

    :return: Image with the object hypothesis written onto it
    :rtype: np.ndarray
    """

    # Font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_colour = (10, 255, 10)
    font_thickness = 1

    for obj in objects:
        # Get area of this colour
        indices = np.where(np.all(labels_image == obj[0], axis=-1))

        # Define description text
        text = obj[1]  # Object hypothesis class name
        text_width, text_height = cv2.getTextSize(text, font, font_size, font_thickness)[0]

        # Location of the text (approx. in the middle of the object)
        loc_x = int(np.round(indices[1].mean() - text_width / 2))
        loc_y = int(np.round(indices[0].mean() - text_height / 2))
        loc = (loc_x, loc_y)

        # Place the text
        cv2.putText(result_image, text, loc, font, font_size, font_colour, font_thickness)

    return result_image


def mask_rgb_image(mask: np.ndarray, image: np.ndarray) -> np.ndarray:
    """ Masks RGB image by using a 2D (!) boolean mask

    :param image: RGB image to check (shape (a, b, 3), np.uint8)
    :type image: np.ndarray

    :param mask: 2D Boolean array masking the image (shape (a, b))
    :type mask: np.ndarray

    :return: Masked image
    :rtype: np.ndarray
    """

    mask = grey_to_rgb_mask(mask)
    masked_image = np.where(mask, image, 0)
    return masked_image


def grey_to_rgb_mask(grey_mask: np.ndarray) -> np.ndarray:
    """ Converts mask with 1 value per pixel to RGB mask with 3 values per pixel

    :param grey_mask: Image mask (shape (a, b), np.uint8)
    :type grey_mask: np.ndarray

    :return: RGB mask (shape (a, b, 3), np.uint8)
    :rtype: np.ndarray
    """

    rgb_mask = np.repeat(grey_mask[:, :, np.newaxis], 3, axis=2)
    return rgb_mask


def colour_bool_mask(image: np.ndarray, colour: np.ndarray) -> np.ndarray:
    """ Returns boolean mask where image matches given color

    :param image: RGB image to check (shape (a, b, 3), np.uint8)
    :type image: np.ndarray

    :param colour: Color to search for
    :type colour: np.ndarray

    :return: Boolean mask
    :rtype: np.ndarray
    """

    mask = np.all(image == colour, axis=-1)
    return mask


def crop_image_black(image: np.ndarray,
                     second_image: np.ndarray = None) -> np.ndarray:
    """ Returns cropped image with removed black around the object(s)

    :param image: If second_image not given: RGB image to crop
                  If second_image given: RGB image to get indices from
                  (shape (a, b, 3), np.uint8)
    :type image: np.ndarray

    :param second_image: Second RGB image to crop from cropping indices of
                         the actual image parameter (to get matching dimensions)
                         (shape (a, b, 3), np.uint8)
    :type second_image: np.ndarray

    :return: Cropped image
    :rtype: np.ndarray
    """

    # Check regions with colour present
    image_notblack = (image > 0).any(axis=2)
    rows_notblack = image_notblack.any(axis=1)
    cols_notblack = image_notblack.any(axis=0)

    # Cropping indices
    row_first = np.argmax(rows_notblack)
    row_last = image_notblack.shape[0] - np.argmax(rows_notblack[::-1])
    col_first = np.argmax(cols_notblack)
    col_last = image_notblack.shape[1] - np.argmax(cols_notblack[::-1])

    # Crop the image
    if second_image is None:
        result = image[row_first:row_last, col_first:col_last, :]
    else:
        result = second_image[row_first:row_last, col_first:col_last, :]

    return result

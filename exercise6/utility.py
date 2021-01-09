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
    """ Projects a pointcloud to a 2D numpy array

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

    # Color each pixel according to pointcloud
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
                    [color, class_best_score, best_score, current_score]
                    [0]    [1]               [2]         [3]
    :type objects: list

    :param labels_image: Image of the colored object clusters
    :type labels_image: np.ndarray

    :param result_image: Image to write the hypothesis texts on
    :type result_image: np.ndarray

    :return: Image with the object hypothesis written onto it
    :rtype: np.ndarray
    """

    # Font parameters
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_size = 0.4
    font_color = (10, 255, 10)
    font_thickness = 1

    for obj in objects:
        # Get area of this color
        indices = np.where(np.all(labels_image == obj[0], axis=-1))

        # Define description text
        text = "{} ({:.1f})".format(obj[1], obj[2])
        text_width, text_height = cv2.getTextSize(text, font, font_size, font_thickness)[0]

        # Location of the text (approx. in the middle of the object)
        loc_x = int(np.round(indices[1].mean() - text_width / 2))
        loc_y = int(np.round(indices[0].mean() - text_height / 2))
        loc = (loc_x, loc_y)

        # Place the text
        cv2.putText(result_image, text, loc, font, font_size, font_color, font_thickness)

    return result_image

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Detect an object in an image and return position, scale and orientation

Author: Max Tamussino
MatrNr: 01611815
"""
from typing import Tuple, List

import numpy as np
import cv2

import sklearn.cluster
import matplotlib.pyplot as plt

from helper_functions import *


def detect_objects(scene_img: np.ndarray,
                   object_img: np.ndarray,
                   scene_keypoints: List[cv2.KeyPoint],
                   object_keypoints: List[cv2.KeyPoint],
                   matches: List[cv2.DMatch],
                   debug_output: bool = False) -> np.ndarray:
    """Return detected configurations of object_img in scene_img given keypoints and matches

    In this function you should implement the whole object detection pipeline. First, filter out bad matches.
    Then extract the position, scale, and orientation of all object hypotheses. Store them in a voting space.
    Cluster the data points of the voting space using one of the clustering algorithms of sklearn.cluster.
    You will need to filter the clusters further to arrive at concrete object hypotheses.
    The object recognition does not have to work perfectly for all provided images,
    but you should be able to explain in the documentation why some errors occur.

    :param scene_img: The color image where the object should be detected.
    :type scene_img: np.ndarray with shape (height, width, 3) with dtype = np.uint8 and values in the range [0, 255]

    :param object_img: An image of the object to be detected
    :type object_img: np.ndarray with shape (height, width, 3) with dtype = np.uint8 and values in the range [0, 255]

    :param scene_keypoints: List of all detected SIFT keypoints in the scene_img
    :type scene_keypoints: list of cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param object_keypoints: List of all detected SIFT keypoints in the object_img
    :type object_keypoints: list of cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param matches: Holds all the possible matches. Each 'row' are matches of one object_keypoint to scene_keypoints
    :type matches: list of lists of cv2.DMatch https://docs.opencv.org/master/d4/de0/classcv_1_1DMatch.html

    :param debug_output: If True enables additional output of intermediate steps.
        [You can use this parameter for your own debugging efforts, but you don't have to use it for the submission.]
    :type debug_output: bool

    :return: An array with shape (n, 4) with each row holding one detected object configuration (x, y, s, o)
        x, y: coordinates of the top-left corner of the detected object in the scene image coordinate frame
        s: relative scale of the object between object coordinate frame and scene image coordinate frame
        o: orientation (clockwise)
    :rtype: np.array with shape (n, 4) with n being the number of detected objects
    """
    ######################################################
    # Find the parameters of all object configuration votes
    configs_all = np.empty((0, 4))
    for match_list in matches:
        for match in match_list:
            # Discard bad matches
            if match.distance > 270:
                continue

            # Extract parameters and save object configuration
            params = match_to_params(scene_keypoints[match.trainIdx], object_keypoints[match.queryIdx])
            configs_all = np.r_[configs_all, [params]]

    # Normalise x, y and scale of the found configurations
    configs_norm = configs_all.copy()[:, :3]
    for k in range(len(configs_norm[0])):
        # Normalise each row separately
        configs_norm[:, k] /= np.max(np.abs(configs_norm[:, k]))

    # Angles will be represented in "complex" to circumvent
    # the transition around -180 degrees. Therefore, sine
    # and cosine of the angle will be saved for clustering,
    # which range from -1 to 1; therefore separately normalised
    # to -0.5 to 0.5.
    angle_scaling = 0.5
    configs_norm = np.c_[configs_norm, angle_scaling * np.cos(configs_all[:, 3])]
    configs_norm = np.c_[configs_norm, angle_scaling * np.sin(configs_all[:, 3])]

    # Apply clustering algorithm DBSCAN to the normalised configurations
    configs_clustered = sklearn.cluster.DBSCAN(min_samples=18, eps=0.1).fit_predict(configs_norm)

    # Calculate final configuration of cluster parameters
    object_configurations = np.empty((0, 4))
    for cluster in range(np.max(configs_clustered) + 1):
        # Save selected x, y and scale of configurations in current cluster
        configs_selected = configs_all[configs_clustered == cluster][:, :3]

        # Save denormalised sine and cosine of the angles in current cluster
        cosine_selected = configs_norm[configs_clustered == cluster][:, 3] / angle_scaling
        sine_selected = configs_norm[configs_clustered == cluster][:, 4] / angle_scaling

        # Save cluster median of x, y and scale to final configuration
        configuration = np.median(configs_selected[:, :3], axis=0)

        # Calculate angle of final configuration from sinus and cosinus medians
        angle = np.arctan2(np.median(sine_selected), np.median(cosine_selected))

        # Append angle to final configuration
        configuration = np.append(configuration, angle)

        # Save configuration
        object_configurations = np.r_[object_configurations, [configuration]]
    ######################################################
    return object_configurations


def match_to_params(scene_keypoint: cv2.KeyPoint, object_keypoint: cv2.KeyPoint) -> Tuple[float, float, float, float]:
    """ Compute the position, rotation and scale of an object implied by a matching pair of descriptors

    This function uses two matching keypoints in the object and scene image to calculate the x and y coordinates, the
    scale and the orientation of the object in the scene image. The scale factor determines the relative size of the
    object image in the scene image.
    The orientation is the rotation of the object in the scene image in clockwise direction in rad.
    A rotation of 0 would be in x-direction and means no relative orientation change between object and scene image.

    :param scene_keypoint: Keypoint in the scene_img where we want to detect the object
    :type scene_keypoint: cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :param object_keypoint: Keypoint in the object_img
    :type object_keypoint: cv2.KeyPoint https://docs.opencv.org/trunk/d2/d29/classcv_1_1KeyPoint.html

    :return: (x, y, scale, orientation)
        x,y: coordinates of top-left corner of detected object in the scene image
        s: relative scale
        o: orientation (clockwise)
    :rtype: (float, float, float, float)
    """
    ######################################################
    # Obtain keypoint coordinates in scene and object
    scene_x, scene_y = scene_keypoint.pt
    object_x, object_y = object_keypoint.pt

    # Obtain scale and orientation of configuration
    scale = scene_keypoint.size / object_keypoint.size
    orientation = np.pi / 180 * (scene_keypoint.angle - object_keypoint.angle)

    # Make sure orientation is within [-pi, pi]
    if orientation < - np.pi:
        orientation += 2 * np.pi
    elif orientation > np.pi:
        orientation -= 2 * np.pi

    # Calculate distance r from keypoint to origin in object image
    r = np.sqrt(np.square(object_x) + np.square(object_y))

    # Calculate angle from horizontal line at keypoint to origin
    beta = np.arctan2(object_y, object_x) + np.pi

    # Calculate configuration coordinates in the scene image
    x = scene_x + scale * r * np.cos(orientation + beta)
    y = scene_y + scale * r * np.sin(orientation + beta)
    ######################################################
    return x, y, scale, orientation

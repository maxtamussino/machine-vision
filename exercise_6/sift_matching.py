#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Calculate sift descriptors and match

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import numpy as np
import cv2

from plot_results import plot_image


def match_sift_grey(scene_img: np.ndarray, object_img: np.ndarray, debug: bool = False) -> np.array:
    """ Matches two greyscale images using SIFT descriptors

    :param scene_img: Scene image in grey
    :type scene_img: np.array

    :param object_img: Object image in grey
    :type object_img: np.array

    :param debug: Show match images
    :type debug: bool

    :return: Array of coordinates of all matches
    :rtype: np.ndarray
    """

    # Get SIFT keypoints and descriptors
    sift = cv2.SIFT_create()
    scene_keypoints, scene_descriptors = sift.detectAndCompute(scene_img, None)
    object_keypoints, object_descriptors = sift.detectAndCompute(object_img, None)

    # FLANN (Fast Library for Approximate Nearest Neighbors) parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # For each keypoint in object_img get the two best matches
    matches = flann.knnMatch(object_descriptors, scene_descriptors, k=2)

    # Show an image of the matches
    if debug:
        draw_params = dict(flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        matches_img = cv2.drawMatchesKnn(object_img,
                                         object_keypoints,
                                         scene_img,
                                         scene_keypoints,
                                         matches,
                                         None,
                                         **draw_params)
        plot_image(matches_img, "Matches")

    # Sort out bad matches
    match_coordinates = np.empty((0, 2))
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            match_coordinates = np.r_[match_coordinates, [scene_keypoints[m.trainIdx].pt]]

    match_coordinates = np.round(match_coordinates).astype(int)

    return match_coordinates


def match_sift_colour(scene_img: np.ndarray, object_img: np.ndarray, debug: bool = False) -> np.array:
    """ Matches two colour images using SIFT descriptors for every colour

    :param scene_img: Scene image in RGB
    :type scene_img: np.array

    :param object_img: Object image in RGB
    :type object_img: np.array

    :param debug: Show match images
    :type debug: bool

    :return: Array of coordinates of all matches
    :rtype: np.ndarray
    """

    match_coordinates = np.empty((0, 2), dtype=int)

    # For every colour channel
    for col in range(3):
        # Pick out one colour
        scene_img_grey = scene_img[:, :, col]
        object_img_grey = object_img[:, :, col]

        # Add new matches
        new_match_coordinates = match_sift_grey(scene_img_grey, object_img_grey, debug=debug)
        match_coordinates = np.r_[match_coordinates, new_match_coordinates]

    return match_coordinates

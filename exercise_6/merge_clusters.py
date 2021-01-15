#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" 2D merging of clusters

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans

from plot_results import plot_image
from utility import colour_bool_mask, mask_rgb_image


def merge_clusters(object_colours: list,
                   labels_image: np.ndarray,
                   scene_image: np.ndarray,
                   debug: bool = False) -> list:

    merge_distance_threshold = 350

    similar_clusters = list()

    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    for colour in object_colours:
        # Find the border area of this colour
        colour_area = np.where(colour_bool_mask(labels_image, colour), 1., 0.)
        gradient = cv2.morphologyEx(colour_area, cv2.MORPH_GRADIENT, morph_kernel)
        border_mask = gradient.astype(bool)
        border_image = mask_rgb_image(border_mask, labels_image)

        if debug:
            plot_image(border_image, "Border")

        # Check the border for other colours
        for check_colour in object_colours:
            # Skip own object colour
            if (colour == check_colour).all():
                continue
            else:
                # Check if this cluster merge was detected already
                already_detected = False
                for detected_col_1, detected_col_2 in similar_clusters:
                    if (colour == detected_col_1).all():
                        already_detected = True
                        break
                    elif (colour == detected_col_2).all():
                        already_detected = True
                        break
                if already_detected:
                    continue

            # Detect if other valid color adjacent (more than 5 pixels, below considered noise)
            if colour_bool_mask(border_image, check_colour).sum() > 5:
                # Find the two adjacent objects
                obj_1_image = mask_rgb_image(colour_bool_mask(labels_image, colour), scene_image)
                obj_2_image = mask_rgb_image(colour_bool_mask(labels_image, check_colour), scene_image)
                if debug:
                    plot_image(obj_1_image, "Object 1")
                    plot_image(obj_2_image, "Object 2")

                # If similar, add to list of similar clusters
                if debug:
                    print()
                dist = colour_cluster_distance(obj_1_image, obj_2_image, debug=debug)
                if dist < merge_distance_threshold:
                    similar_clusters.append((colour, check_colour))
                    if debug:
                        print("distance", dist, "merged")
                else:
                    if debug:
                        print("distance ", dist, "skipped")

    return similar_clusters


def colour_cluster_distance(image1: np.ndarray, image2: np.ndarray, debug: bool = False) -> np.ndarray:
    """ Returns boolean mask where image matches given color

    :param image1: RGB image 1 to check (shape (a, b, 3), np.uint8)
    :type image1: np.ndarray

    :param image2: RGB image 2 to check (shape (a, b, 3), np.uint8)
    :type image2: np.ndarray

    :param debug: Turn on text debugging
    :type debug: bool

    :return: Distance between images
    :rtype: int
    """

    clusters_1, counts_1 = get_colour_clusters(image1)
    clusters_2, counts_2 = get_colour_clusters(image2)

    distance_colours = np.abs(clusters_2 - clusters_1).sum()
    distance_counts = int(np.round(np.abs(counts_2 - counts_1).sum() * 400/np.abs(counts_2 + counts_1).sum()))
    distance = max(distance_colours, distance_counts)
    if debug:
        print("col", distance_colours, ", count", distance_counts, ", max", distance)

    return distance


def get_colour_clusters(image: np.ndarray) -> (np.ndarray, np.ndarray):
    """ Returns boolean mask where image matches given color

    :param image: RGB image to check (shape (a, b, 3), np.uint8)
    :type image: np.ndarray

    :param colour: Color to search for
    :type colour: np.ndarray

    :return: Boolean mask
    :rtype: np.ndarray
    """

    # Reshape
    image = image.reshape(image.shape[0] * image.shape[1], 3)

    # Cluster colors and remove black
    clf = KMeans(n_clusters=4)
    labels = clf.fit_predict(image)
    _, counts = np.unique(labels, return_counts=True)
    counts = counts[1:]
    center_colours = np.round(clf.cluster_centers_).astype(int)[1:]

    # Sort by occurrences
    center_colours = center_colours[np.argsort(counts)[::-1]]

    return center_colours, counts
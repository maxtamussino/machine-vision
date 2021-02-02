#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Categorise match coordinates

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import numpy as np


def categorise_matches(match_coordinates: np.ndarray,
                       labels_image: np.ndarray,
                       objects: list,
                       object_colors: list,
                       curr_object_title: str) -> list:
    """ Takes the coordinates of all matches an object produces,
        and returns updates list of object hypothesis for each color
        (updates a colors hypothesis if the score of the current
        image is higher)

    :param match_coordinates: Array of all coordinates of the matches
    :type match_coordinates: np.array

    :param labels_image: Image of the colored object clusters
    :type labels_image: np.array

    :param objects: List of found object hypothesis, every entry of the form:
                    [color, class_best_score, best_score, current_score]
                    [0]    [1]               [2]         [3]
    :type objects: list

    :param object_colors: List of all possible object colors
    :type object_colors: list

    :param curr_object_title: Class name of the object the matches are obtained from
    :type curr_object_title: str

    :return: List of new object hypothesis, every entry of the form:
                    [color, class_best_score, best_score, current_score]
                    [0]    [1]               [2]         [3]
    :rtype: list

    """

    if match_coordinates.size == 0:
        return objects

    # Match weight reduced if object has many matches
    match_weight = 1 / match_coordinates.size

    for coordinate in match_coordinates:
        # Retrieve color at match coordinates
        color = labels_image[coordinate[1], coordinate[0]]

        # Sort out black
        if (color == np.array([0, 0, 0])).all():
            continue

        # Match to clusters
        matched = False
        for obj in objects:
            if (obj[0] == color).all():
                obj[3] += match_weight
                matched = True
                break

        # Color has not been discovered yet
        if not matched:
            # Check if the color is valid
            valid = False
            for col in object_colors:
                if (color == col).all():
                    valid = True
                    break
            if valid:
                # No "highest" score yet, so add hypothesis as unknown
                objects.append([color, "unknown", 0., match_weight])

    # Update object hypothesis if higher score and more than two matches
    for obj in objects:
        if obj[3] > obj[2] and obj[3] > 2 * match_weight:
            obj[2] = obj[3]
            obj[1] = curr_object_title
        obj[3] = 0

    return objects

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Different error functions for plane fitting

Author: Max Tamussino
MatrNr: 01611815
"""
from typing import Tuple

import numpy as np
import open3d as o3d


def ransac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the RANSAC error which is the number of outliers.

    The RANSAC error is defined as the number of outliers given a specific model.
    A outlier is defined as a point which distance to the plane is larger (or equal ">=") than a threshold.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The distance of a point to the plane below which point is considered an inlier (in meters)
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################
    inliers = distances < threshold
    outliers = np.invert(inliers)
    error = np.sum(outliers)
    ######################################################
    return error, inliers


def msac_error(pcd: o3d.geometry.PointCloud,
               distances: np.ndarray,
               threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the MSAC error as defined in https://pdfs.semanticscholar.org/40d3/0ade023d671c0f4e41e7045f3e59db44edcc.pdf

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The threshold distance at which to change from a quadratic error to a constant error
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################
    inliers = distances < threshold
    error = np.sum(np.square(np.where(inliers, distances, threshold)))
    ######################################################
    return error, inliers


def mlesac_error(pcd: o3d.geometry.PointCloud,
                 distances: np.ndarray,
                 threshold: float) -> Tuple[float, np.ndarray]:
    """ Calculate the MLESAC error as defined in https://pdfs.semanticscholar.org/40d3/0ade023d671c0f4e41e7045f3e59db44edcc.pdf

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param distances: The distances of each point to the proposed plane
    :type distances: np.ndarray with shape (num_of_points,)

    :param threshold: The sigma value needed for MLESAC is calculated as sigma = threshold/2
    :type threshold: float

    :return: (error, inliers)
        error: The calculated error
        inliers: Boolean mask of inliers (shape: (num_of_points,))
    :rtype: (float, np.ndarray)
    """
    ######################################################
    inliers = distances < threshold

    v = np.linalg.norm(pcd.get_max_bound() - pcd.get_min_bound())

    error = np.inf
    ######################################################

    return error, inliers

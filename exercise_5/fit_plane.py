#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Fit Plane in pointcloud

Author: Max Tamussino
MatrNr: 01611815
"""

from typing import List, Tuple, Callable

import copy

import numpy as np
import open3d as o3d


def fit_plane(pcd: o3d.geometry.PointCloud,
              confidence: float,
              inlier_threshold: float,
              min_sample_distance: float,
              error_func: Callable) -> Tuple[np.ndarray, np.ndarray, int]:
    """ Find dominant plane in pointcloud with sample consensus.

    Detect a plane in the input pointcloud using sample consensus. The number of iterations is chosen adaptively.
    The concrete error function is given as an parameter.

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from the plane to be considered an inlier (in meters)
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (best_plane, best_inliers, num_iterations)
        best_plane: Array with the coefficients of the plane equation ax+by+cz+d=0 (shape = (4,))
        best_inliers: Boolean array with the same length as pcd.points. Is True if the point at the index is an inlier
        num_iterations: The number of iterations that were needed for the sample consensus
    :rtype: tuple (np.ndarray[a,b,c,d], np.ndarray, int)
    """
    ######################################################
    points = np.asarray(pcd.points)
    N = points.shape[0]
    best_inliers = np.full(N, False)
    num_iterations = 0
    eps = 3/N
    error_best = np.inf

    while (1 - eps ** 3) ** num_iterations >= (1 - confidence):

        choice_idx = np.random.randint(0, N, 3)
        a = points[choice_idx[0]]
        b = points[choice_idx[1]]
        c = points[choice_idx[2]]

        choice_valid = np.linalg.norm(b - a) > min_sample_distance
        choice_valid = choice_valid and (np.linalg.norm(c - a) > min_sample_distance)
        choice_valid = choice_valid and (np.linalg.norm(c - b) > min_sample_distance)
        if not choice_valid:
            continue

        vec_norm = np.cross(b - a, c - a)
        vec_norm /= np.linalg.norm(vec_norm)

        # Normal vector [A,B,C], point [x0,y0,z0]
        # A(x − x0) + B(y − y0) + C(z − z0) = 0
        # A*x + B*y + C*z - A*x0 - B*y0 - C*z0 = 0
        model = np.array([0., 0., 0., 0.])
        for i in range(3):
            model[i] = vec_norm[i]
            model[3] -= model[i] * a[i]

        distances = abs(np.dot(points, vec_norm).T + model[3])
        error, inliers = error_func(pcd, distances, inlier_threshold)

        if error < error_best:
            error_best = error
            best_inliers = inliers
            eps = np.sum(inliers)/N

        num_iterations = num_iterations + 1

    print("Interations: ", num_iterations)
    new_inliers = points[best_inliers]
    best_plane = np.linalg.lstsq(new_inliers, np.ones(new_inliers.shape[0]), rcond=None)[0]
    best_plane = np.append(best_plane, -1)

    if best_plane[2] < 0:
        best_plane *= -1
    print("Plane: ", best_plane)

    return best_plane, best_inliers, num_iterations


def filter_planes(pcd: o3d.geometry.PointCloud,
                  min_points_prop: float,
                  confidence: float,
                  inlier_threshold: float,
                  min_sample_distance: float,
                  error_func: Callable) -> Tuple[List[np.ndarray],
                                                 List[o3d.geometry.PointCloud],
                                                 o3d.geometry.PointCloud]:
    """ Find multiple planes in the input pointcloud and filter them out.

    Find multiple planes by applying the detect_plane function multiple times. If a plane is found in the pointcloud,
    the inliers of this pointcloud are filtered out and another plane is detected in the remaining pointcloud.
    Stops if a plane is found with a number of inliers < min_points_prop * number of input points.

    :param pcd: The (down-sampled) pointcloud in which to detect planes
    :type pcd: o3d.geometry.PointCloud (http://www.open3d.org/docs/release/python_api/open3d.geometry.PointCloud.html)

    :param min_points_prop: The proportion of points of the input pointcloud which have to be inliers of a plane for it
        to qualify as a valid plane.
    :type min_points_prop: float

    :param confidence: Solution Confidence (in percent): Likelihood of all sampled points being inliers for each plane.
    :type confidence: float

    :param inlier_threshold: Max. distance of a point from a plane to be considered an inlier (in meters).
    :type inlier_threshold: float

    :param min_sample_distance: Minimum distance of all sampled points to each other (in meters). For robustness.
    :type min_sample_distance: float

    :param error_func: Function to use when computing inlier error defined in error_funcs.py
    :type error_func: Callable

    :return: (plane_eqs, plane_pcds, filtered_pcd)
        plane_eqs is a list of np.arrays each holding the coefficient of a plane equation for one of the planes
        plane_pcd is a list of pointclouds with each holding the inliers of one plane
        filtered_pcd is the remaining pointcloud of all points which are not part of any of the planes
    :rtype: (List[np.ndarray], List[o3d.geometry.PointCloud], o3d.geometry.PointCloud)
    """
    ######################################################
    # Write your own code here
    plane_eqs = []
    plane_pcds = []
    filtered_pcd = copy.deepcopy(pcd)

    points_threshold = min_points_prop * np.asarray(pcd.points).shape[0]

    while np.asarray(filtered_pcd.points).shape[0] > points_threshold:
        best_plane, best_inliers, _ = fit_plane(filtered_pcd, confidence, inlier_threshold, min_sample_distance, error_func)

        if np.sum(best_inliers) < points_threshold:
            break

        plane_eqs.append(best_plane)

        filter = np.flatnonzero(best_inliers)
        inlier_pcd = filtered_pcd.select_by_index(filter)
        filtered_pcd = filtered_pcd.select_by_index(filter, invert=True)
        plane_pcds.append(inlier_pcd)



    return plane_eqs, plane_pcds, filtered_pcd

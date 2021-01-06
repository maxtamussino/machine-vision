#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Project pointclouds to 2D image

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import open3d as o3d
import numpy as np
from camera_params import *


def project_2d(pcd: o3d.geometry.PointCloud) -> np.ndarray:
    """ Projects a pointcloud to a 2D numpy array

    :param pcd: The pointcloud which is to be projected
    :type pcd: o3d.geometry.PointCloud

    :return: np.ndarray
    """

    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)[..., ::-1]

    origin_idx = np.where(points[:, 2] == 0)[0]
    points = np.delete(points, origin_idx, 0)
    colors = np.delete(colors, origin_idx, 0)

    u = np.round(fx_rgb * np.divide(points[:, 0], points[:, 2]) + cx_rgb).astype(int)
    v = np.round(fy_rgb * np.divide(points[:, 1], points[:, 2]) + cy_rgb).astype(int)

    image = np.zeros((v.max() + 1, u.max() + 1, 3))

    for i in range(points.shape[0]):
        image[v[i], u[i]] = colors[i]

    return image

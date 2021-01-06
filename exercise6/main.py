#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" TODO Find title

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

from pathlib import Path

import cv2
import copy

from plot_results import *
from project_2d import *

if __name__ == '__main__':
    # Selects which single-plane file to use
    pointcloud_idx = 2
    voxel_size = 0.005

    # Read Pointcloud
    current_path = Path(__file__).parent
    pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00")) + str(pointcloud_idx) + ".pcd")
    if not pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Apply plane-fitting algorithm
    _, inliers = pcd.segment_plane(distance_threshold=0.01,
                                   ransac_n=3,
                                   num_iterations=1000)

    # Remove the plane
    pcd_filtered = pcd.select_by_index(inliers, invert=True)

    # Down-sample the loaded point cloud to reduce computation time
    pcd_sampled = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)

    # Clustering
    labels = np.array(pcd_sampled.cluster_dbscan(eps=0.05, min_points=80))
    max_label = labels.max()
    colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
    colors[labels < 0] = 0
    pcd_labels = copy.deepcopy(pcd_sampled)
    pcd_labels.colors = o3d.utility.Vector3dVector(colors[:, :3])

    # Project original color image to 2D
    color_image = project_2d(pcd_filtered)
    plot_image(color_image, "Original")

    # Project clusters to 2D and fill holes
    labels_image = project_2d(pcd_labels)
    labels_image = cv2.dilate(labels_image, np.ones((3, 3)), iterations=2)
    plot_image(labels_image, "Labels")

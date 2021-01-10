#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Detect and classify objects in 3D pointclouds

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

from pathlib import Path

import cv2
import copy

from plot_results import *
from categorise_matches import *
from utility import *
from matching import *

if __name__ == '__main__':
    # Parameters
    use_color_matching = True
    pointcloud_idx = 5
    voxel_size = 0.003

    # Training pointcloud names
    training_pcds = {
        "book": 17,
        "cookiebox": 22,
        "cup": 17,
        "ketchup": 22,
        "sugar": 22,
        "sweets": 22,
        "tea": 22
    }

    # Read Pointcloud
    current_path = Path(__file__).parent
    obj_pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00")) + str(pointcloud_idx) + ".pcd")
    if not obj_pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Apply plane-fitting algorithm
    _, inliers = obj_pcd.segment_plane(distance_threshold=0.01,
                                       ransac_n=3,
                                       num_iterations=5000)

    # Remove the plane
    pcd_filtered = obj_pcd.select_by_index(inliers, invert=True)

    # Project color image without plane to 2D
    scene_image = project_2d(pcd_filtered)

    # Down-sample the loaded point cloud to reduce computation time
    pcd_sampled = pcd_filtered.voxel_down_sample(voxel_size=voxel_size)

    # Clustering
    labels = np.array(pcd_sampled.cluster_dbscan(eps=0.02, min_points=150))

    # Create the color map, non-clustered noise is black
    num_labels = labels.max() + 1
    colors = plt.get_cmap("gist_rainbow")(labels / num_labels)[:, :3]
    colors[labels < 0] = 0

    # Save possible colors (mixtures are sometimes created by hole-filling)
    object_colors = np.empty((0, 3))
    for label in range(num_labels):
        color = plt.get_cmap("gist_rainbow")(label / num_labels)[:3]
        object_colors = np.vstack((object_colors, color))
    object_colors = (object_colors[..., ::-1] * 255).astype(np.uint8)

    # Create colored pointcloud
    pcd_labels = copy.deepcopy(pcd_sampled)
    pcd_labels.colors = o3d.utility.Vector3dVector(colors)

    # Project colored clusters to 2D
    labels_image = project_2d(pcd_labels)

    # Fill the holes in the colored image
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    labels_image = cv2.morphologyEx(labels_image, cv2.MORPH_CLOSE, kernel)

    # For every color in the labels image, an entry is created
    # Format: [color, class_best_score, best_score, current_score]
    objects = list()

    # Iterate over all training pointclouds
    current_path = Path(__file__).parent
    for object_title, num in training_pcds.items():
        for i in range(num):
            print("Object: {} ({})...".format(object_title, i))

            # Load training pointcloud
            file = str(current_path.joinpath("training/" + object_title)) + str(i).zfill(3) + ".pcd"
            obj_pcd = o3d.io.read_point_cloud(file)
            if not obj_pcd.has_points():
                raise FileNotFoundError("Couldn't load pointcloud in " + file)

            # Project object to 2D
            object_image = project_2d(obj_pcd)

            # Calculate SIFT matching coordinates in scene
            if use_color_matching:
                match_coordinates = match_sift_color(scene_image, object_image)
            else:
                scene_img_grey = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
                object_img_grey = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
                match_coordinates = match_sift_grey(scene_img_grey, object_img_grey)

            # Update the object hypothesis list
            objects = categorise_matches(match_coordinates, labels_image, objects, object_colors, object_title)

    # Write the object class names
    result_image = write_hypothesis(objects, labels_image, scene_image)

    # Plot the result
    plot_image(scene_image, "result_" + str(pointcloud_idx), save_image=True)

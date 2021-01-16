#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Detect and classify objects in 3D pointclouds

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""
import time
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import cv2
import copy

from plot_results import plot_image, plot_pointclouds
from categorise_matches import categorise_matches
from utility import project_2d, crop_image_black, colour_bool_mask, write_hypothesis
from sift_matching import match_sift_grey, match_sift_colour
from merge_clusters import merge_clusters


def detect_objects(pcd_idx: int):
    # Debugging
    debug_text = True    # Show text output
    show_images = False  # Show step result images
    save_images = True   # Save step result images
    timing = False       # Print timing information
    if timing:
        endln = ""
        total_start = time.perf_counter()
    else:
        endln = "\n"

    # Features
    use_colour_matching = True  # Activate RGB SIFT
    use_cluster_merging = True  # Activate cluster merging
    use_cropping = True         # Activate image cropping

    # Training pointcloud names + number of pointclouds
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
    obj_pcd = o3d.io.read_point_cloud(str(current_path.joinpath("test/image00")) + str(pcd_idx) + ".pcd")
    if not obj_pcd.has_points():
        raise FileNotFoundError("Couldn't load pointcloud in " + str(current_path))

    # Apply plane-fitting algorithm
    if debug_text:
        print("plane fitting...", end=endln)
    if timing:
        start = time.perf_counter()
    _, inliers = obj_pcd.segment_plane(distance_threshold=0.015, ransac_n=3, num_iterations=5000)
    if timing:
        end = time.perf_counter()
        time_planefitting = (end - start) * 1000
        if debug_text:
            print(" {:.1f}ms".format(time_planefitting))

    # Remove the plane
    pcd_filtered = obj_pcd.select_by_index(inliers, invert=True)

    # Project colour image without plane to 2D
    scene_image = project_2d(pcd_filtered)
    plot_image(scene_image, "colour_scene_" + str(pcd_idx),
               show_image=show_images, save_image=save_images)

    # Down-sample the loaded point cloud to reduce computation time
    pcd_sampled = pcd_filtered.voxel_down_sample(voxel_size=0.003)

    # Clustering
    if debug_text:
        print("clustering...", end=endln)
    if timing:
        start = time.perf_counter()
    labels = np.array(pcd_sampled.cluster_dbscan(eps=0.025, min_points=150))
    if timing:
        end = time.perf_counter()
        time_clustering = (end - start) * 1000
        if debug_text:
            print(" {:.1f}ms".format(time_clustering))

    # Create the colour map, non-clustered noise is black
    num_labels = labels.max() + 1
    colours = plt.get_cmap("gist_rainbow")(labels / num_labels)[:, :3]
    colours[labels < 0] = 0

    # Save possible colours (mixtures are sometimes created by hole-filling)
    object_colours = np.empty((0, 3))
    for label in range(num_labels):
        colour = plt.get_cmap("gist_rainbow")(label / num_labels)[:3]
        object_colours = np.vstack((object_colours, colour))

    # Convert the colours to RGB and np.uint8
    object_colours = (object_colours[..., ::-1] * 255).astype(np.uint8)

    # Create coloured pointcloud
    pcd_labels = copy.deepcopy(pcd_sampled)
    pcd_labels.colors = o3d.utility.Vector3dVector(colours)

    # Project coloured clusters to 2D
    labels_image = project_2d(pcd_labels)
    plot_image(labels_image, "labels_unfilled_" + str(pcd_idx),
               show_image=show_images, save_image=save_images)

    # Fill the holes in the coloured image
    morph_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    labels_image = cv2.morphologyEx(labels_image, cv2.MORPH_CLOSE, morph_kernel)
    plot_image(labels_image, "labels_filled_" + str(pcd_idx),
               show_image=show_images, save_image=save_images)

    # Crop image according to labels image (first the scene image, see doc)
    if use_cropping:
        scene_image = crop_image_black(labels_image, scene_image)
        labels_image = crop_image_black(labels_image)
        plot_image(labels_image, "labels_cropped_" + str(pcd_idx),
                   show_image=show_images, save_image=save_images)
        plot_image(scene_image, "color_scene_cropped_" + str(pcd_idx),
                   show_image=show_images, save_image=save_images)

    # Re-cluster adjacent clusters with similar colours
    if use_cluster_merging:
        if debug_text:
            print("cluster merging...", end=endln)
        if timing:
            start = time.perf_counter()

        similar_clusters = merge_clusters(object_colours, labels_image, scene_image, debug=False)
        for col1, col2 in similar_clusters:
            mask = colour_bool_mask(labels_image, col1)
            labels_image[mask] = col2

        if timing:
            end = time.perf_counter()
            time_clustermerging = (end - start) * 1000
            if debug_text:
                print(" {:.1f}ms".format(time_clustermerging))
        plot_image(labels_image, "labels_merged_" + str(pcd_idx),
                   show_image=show_images, save_image=save_images)

    # For every colour in the labels image, an entry is created
    # Format: [colour, class_best_score, best_score, current_score]
    objects = list()

    # Iterate over all training pointclouds
    current_path = Path(__file__).parent
    if timing:
        times_per_pcd = list()
    for object_title, num in training_pcds.items():
        for current_pcd in range(num):
            if debug_text:
                print("checking {} nr. {}...".format(object_title, current_pcd), end=endln)
            if timing:
                start = time.perf_counter()

            # Load training pointcloud
            file = str(current_path.joinpath("training/" + object_title)) + str(current_pcd).zfill(3) + ".pcd"
            obj_pcd = o3d.io.read_point_cloud(file)
            if not obj_pcd.has_points():
                raise FileNotFoundError("Couldn't load pointcloud in " + file)

            # Project object to 2D and crop to reduce computational effort
            object_image = project_2d(obj_pcd)
            if use_cropping:
                object_image = crop_image_black(object_image)

            # Calculate SIFT matching coordinates in scene
            if use_colour_matching:
                match_coordinates = match_sift_colour(scene_image, object_image, debug=False)
            else:
                scene_img_grey = cv2.cvtColor(scene_image, cv2.COLOR_BGR2GRAY)
                object_img_grey = cv2.cvtColor(object_image, cv2.COLOR_BGR2GRAY)
                match_coordinates = match_sift_grey(scene_img_grey, object_img_grey)

            if timing:
                end = time.perf_counter()
                current_time = (end - start) * 1000
                times_per_pcd.append(current_time)
                if debug_text:
                    print(" {:.1f}ms".format(current_time))

            # Update the object hypothesis list
            objects = categorise_matches(match_coordinates, labels_image, objects, object_colours, object_title)

    # Finish total timing
    if timing:
        total_end = time.perf_counter()
        time_total = (total_end - total_start) * 1000
        time_pcds_avg = sum(times_per_pcd) / len(times_per_pcd)
        if debug_text:
            print("Average per pcd: {:.1f}ms".format(time_pcds_avg))
            print("Complete! Total time: {:.1f}ms".format(time_total))

    # Write the object class names and plot the result
    result_image = write_hypothesis(objects, labels_image, scene_image)
    plot_image(result_image, "result_" + str(pcd_idx),
               show_image=True, save_image=True)


if __name__ == '__main__':
    all_scenes = False

    if all_scenes:
        for i in range(10):
            print("\npointcloud {}:".format(i))
            detect_objects(i)
    else:
        detect_objects(0)



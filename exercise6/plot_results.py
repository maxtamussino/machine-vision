#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""
from typing import List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import cv2


def plot_pointclouds(pcd: o3d.geometry.PointCloud) -> None:
    """ Plot the inlier points in red and the rest of the pointcloud as is. A coordinate frame is drawn on the plane

    :param pcd: The (down-sampled) pointcloud in which to detect the dominant plane
    :type pcd: o3d.geometry.PointCloud

    :param inliers: Boolean array with the same size as pcd.points. Is True if the point at the index is an inlier
    :type inliers: np.array

    :param plane_eq: An array with the coefficients of the plane equation ax+by+cz+d=0
    :type plane_eq: np.array [a,b,c,d]

    :return: None
    """

    vis = o3d.visualization.Visualizer()
    vis.create_window()
    vis.add_geometry(pcd)
    vc = vis.get_view_control()
    vc.set_front([-0.3, 0.32, -0.9])
    vc.set_lookat([-0.13, -0.15, 0.92])
    vc.set_up([0.22, -0.89, -0.39])
    vc.set_zoom(0.5)
    vis.run()
    vis.destroy_window()


def plot_image(img: np.array, title: str, save_image: bool = False, use_matplotlib: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.array with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param save_image: If this is set to True, an image will be saved to disc as title.png
    :type save_image: bool

    :param use_matplotlib: If this is set to True, Matplotlib will be used for plotting, OpenCV otherwise
    :type use_matplotlib: bool
    """

    # First check if img is color or grayscale. Raise an exception on a wrong type.
    if len(img.shape) == 3:
        is_color = True
    elif len(img.shape) == 2:
        is_color = False
    else:
        raise ValueError(
            'The image does not have a valid shape. Expected either (height, width) or (height, width, channels)')

    if use_matplotlib:
        plt.figure()
        plt.title(title)
        if is_color:
            # OpenCV uses BGR order while Matplotlib uses RGB. Reverse the the channels to plot the correct colors
            plt.imshow(img[..., ::-1])
        else:
            plt.imshow(img, cmap='gray')
        plt.xticks([])
        plt.yticks([])
        plt.show()
    else:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        if is_color:
            png_img = (cv2.cvtColor(img, cv2.COLOR_BGR2BGRA) * 255.).astype(np.uint8)
        else:
            png_img = (cv2.cvtColor(img, cv2.COLOR_GRAY2BGRA) * 255.).astype(np.uint8)
        cv2.imwrite(title.replace(" ", "_") + ".png", png_img)

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

""" Helper functions to plot the results

Machine Vision and Cognitive Robotics (376.054)
Author: Max Tamussino
MatrNr: 01611815
"""

import numpy as np
import open3d as o3d
import cv2


def plot_pointclouds(pcd: o3d.geometry.PointCloud) -> None:
    """ Plot a given pointcloud

    :param pcd: The (down-sampled) pointcloud
    :type pcd: o3d.geometry.PointCloud

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


def plot_image(img: np.array, title: str, show_image: bool = False, save_image: bool = False) -> None:
    """ Plot an image with either OpenCV or Matplotlib.

    :param img: :param img: Input image
    :type img: np.array with shape (height, width) or (height, width, channels)

    :param title: The title of the plot which is also used as a filename if save_image is chosen
    :type title: string

    :param show_image: If this is set to True, the image will be shown
    :type show_image: bool

    :param save_image: If this is set to True, the image will be saved to disc as title.png
    :type save_image: bool

    :return: None
    """

    if show_image:
        cv2.imshow(title, img)
        cv2.waitKey(0)

    if save_image:
        cv2.imwrite("./tex/figures/unused/" + title.replace(" ", "_") + ".png", img)

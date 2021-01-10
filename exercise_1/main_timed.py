#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Machine Vision and Cognitive Robotics (376.054)
Exercise 1: Canny Edge Detector
Matthias Hirschmanner 2020
Automation & Control Institute, TU Wien

Tutors: machinevision@acin.tuwien.ac.at
"""
from pathlib import Path

import cv2
import numpy as np
import timeit

from blur_gauss import blur_gauss
from helper_functions import *
from hyst_auto import hyst_thresh_auto
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel
from helper_functions import show_image

if __name__ == '__main__':

    # Define behavior of the show_image function. You can change these variables if necessary
    save_image = False
    matplotlib_plotting = False

    # Read image
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath("image/circle.jpg")), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))

    # Before we start working with the image, we convert it from uint8 with range [0,255] to float32 with range [0,1]
    img_gray = img_gray.astype(np.float32) / 255.
    show_image(img_gray, "Original Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # 1. Blur Image
    sigma = 3  # Change this value
    start = timeit.default_timer()
    img_blur = blur_gauss(img_gray, sigma)
    print("Blurring: {:.2f}ms".format((timeit.default_timer() - start)*1000))
    show_image(img_blur, "Blurred Image", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # 2. Edge Detection
    start = timeit.default_timer()
    gradients, orientations = sobel(img_blur)
    print("Sobel: {:.2f}ms".format((timeit.default_timer() - start)*1000))
    orientations_color = cv2.applyColorMap(np.uint8((orientations.copy() + np.pi) / (2 * np.pi) * 255),
                                           cv2.COLORMAP_RAINBOW)
    orientations_color = orientations_color.astype(np.float32) / 255.
    gradient_img = np.append(cv2.cvtColor(gradients, cv2.COLOR_GRAY2BGR), orientations_color, axis=1)
    show_image(gradient_img, "Gradients", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # 3. Non-Maxima Suppression
    start = timeit.default_timer()
    edges = non_max(gradients, orientations)
    print("Non Maxima Suppression: {:.2f}ms".format((timeit.default_timer() - start)*1000))
    show_image(edges, "Non Maxima Suppression", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # 4. Hysteresis Thresholding
    hyst_method_auto = True
    start = timeit.default_timer()
    if hyst_method_auto:
        canny_edges = hyst_thresh_auto(edges, 0.25, 0.1)
    else:
        canny_edges = hyst_thresh(edges, 0.3, 0.4)
    print("Canny time: {:.2f}ms".format((timeit.default_timer() - start)*1000))
    show_image(canny_edges, "Canny Edges", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # Overlay the found edges in red over the original image
    img_gray_overlay = cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)
    img_gray_overlay[canny_edges == 1.0] = (0., 0., 1.)
    show_image(img_gray_overlay, "Overlay", save_image=save_image, use_matplotlib=matplotlib_plotting)

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()
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

from blur_gauss import blur_gauss
from helper_functions import *
from hyst_auto import hyst_thresh_auto
from hyst_thresh import hyst_thresh
from non_max import non_max
from sobel import sobel

if __name__ == '__main__':

    # Define behavior
    save_image = False
    experiment = 3.2
    standard_sigma = 3  # From 2 onwards

    high_prop = 0.1  # From 2.3 onwards
    low_prop = 0.25  # From 2.3 onwards

    # Read image
    current_path = Path(__file__).parent
    img_gray = cv2.imread(str(current_path.joinpath("image/rubens.jpg")), cv2.IMREAD_GRAYSCALE)
    if img_gray is None:
        raise FileNotFoundError("Couldn't load image in " + str(current_path))
    img_gray = img_gray.astype(np.float32) / 255.

    if experiment == 1.1:  # 1.1 - Blur Image with different sigma & kernel_width values
        for sigma in range(3, 4):
            for kernel_width in [3, 9, 19]:
                img_blur = blur_gauss(img_gray, sigma, kernel_width)
                name = "1_1_Blurred_sigma_{}_kernelwidth_{}".format(sigma, kernel_width)
                show_image(img_blur, name, save_image=save_image)

    elif experiment == 1.2:  # 1.2 - Row intensities before and after blurring
        plot_row_intensities(img_gray, 300, "1_2_Row_inten_original", save_image)
        for sigma in range(1, 4):
            img_blur = blur_gauss(img_gray, sigma)
            name = "1_2_RowIntensity_sigma_{}".format(sigma)
            plot_row_intensities(img_blur, 300, name, save_image)

    elif experiment == 1.3:  # 1.3 - Effect of sigma on non-maxima suppression
        for sigma in range(1, 4):
            img_blur = blur_gauss(img_gray, sigma)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            name = "1_3_NonMax_sigma_{}".format(sigma)
            show_image(edges, name, save_image=save_image)

    elif experiment == 1.4:  # 1.4 - Effect of sigma on hysteresis-thresholding
        for sigma in range(1, 4):
            img_blur = blur_gauss(img_gray, sigma)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            canny_edges = hyst_thresh(edges, 0.1, 0.3)
            name = "1_4_Canny_sigma_{}".format(sigma)
            show_image(canny_edges, name, save_image=save_image)

    elif experiment == 2.1:  # 2.1 - Effect of different thresholds
        img_blur = blur_gauss(img_gray, standard_sigma)
        gradients, orientations = sobel(img_blur)
        edges = non_max(gradients, orientations)
        threshold_pairs = []
        for t1 in [0.05, 0.1, 0.2, 0.3]:
            threshold_pairs.append((t1, 0.3))
        for t2 in [0.2, 0.3, 0.4, 0.5]:
            threshold_pairs.append((0.2, t2))
        for t1, t2 in threshold_pairs:
            canny_edges = hyst_thresh(edges, t1, t2)
            name = "2_1_Thresholds_low={}_high={}".format(t1, t2)
            show_image(canny_edges, name, save_image=save_image)

    elif experiment == 2.2:  # 2.2 - Effect of same thresholds on different images
        for image in ["rubens", "beardman", "parliament"]:
            img_gray = cv2.imread(str(current_path.joinpath("image/" + image + ".jpg")), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise FileNotFoundError("Couldn't load image in " + str(current_path))
            img_gray = img_gray.astype(np.float32) / 255.
            img_blur = blur_gauss(img_gray, standard_sigma)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            t1 = 0.2
            t2 = 0.4
            canny_edges = hyst_thresh(edges, t1, t2)
            name = "2_2_CannyStatic_{}".format(image)
            show_image(canny_edges, name, save_image=save_image)

    elif experiment == 2.3:  # 2.3 - Effect of adapted thresholds on different images
        for image in ["rubens", "beardman", "parliament"]:
            img_gray = cv2.imread(str(current_path.joinpath("image/" + image + ".jpg")), cv2.IMREAD_GRAYSCALE)
            if img_gray is None:
                raise FileNotFoundError("Couldn't load image in " + str(current_path))
            img_gray = img_gray.astype(np.float32) / 255.
            img_blur = blur_gauss(img_gray, standard_sigma)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            canny_edges = hyst_thresh_auto(edges, low_prop, high_prop)
            name = "2_3_CannyAuto_{}".format(image)
            show_image(canny_edges, name, save_image=save_image)

    elif experiment == 3.1:  # 3.1 - Effect of noise on detected edges
        for noise_level in [0.01, 0.05, 0.1, 0.3, 0.5]:
            img_noise = add_gaussian_noise(img_gray, sigma=noise_level)
            name = "3_1_NoiseOriginal_{}".format(noise_level)
            show_image(img_noise, name, save_image=save_image)
            img_blur = blur_gauss(img_noise, standard_sigma)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            canny_edges = hyst_thresh_auto(edges, low_prop, high_prop)
            name = "3_1_CannyOnNoise_{}".format(noise_level)
            show_image(canny_edges, name, save_image=save_image)

    elif experiment == 3.2:  # 3.2 - Minimising the effect of noise
        noise_level = 0.5
        img_noise = add_gaussian_noise(img_gray, sigma=noise_level)
        name = "3_2_NoiseOriginal_{}".format(noise_level)
        show_image(img_noise, name, save_image=save_image)
        param_pairs = [(low_prop, high_prop, standard_sigma)]
        for sigma_dev in [0.5, 1]:
            param_pairs.append((low_prop, high_prop, standard_sigma + sigma_dev))
        #for low_dev in [-0.05, 0.05]:
            #param_pairs.append((low_prop + low_dev, high_prop, standard_sigma))
        for high_dev in [-0.07, -0.05]:
            param_pairs.append((low_prop, high_prop + high_dev, standard_sigma))
        for high_dev in [-0.07, -0.05]:
            param_pairs.append((low_prop, high_prop + high_dev, standard_sigma+1))
        for lpr, hpr, sig in param_pairs:
            img_blur = blur_gauss(img_noise, sig)
            gradients, orientations = sobel(img_blur)
            edges = non_max(gradients, orientations)
            canny_edges = hyst_thresh_auto(edges, lpr, hpr)
            name = "3_2_MinimiseNoise_sig{}_low{:.2f}_high{:.2f}".format(sig, lpr, hpr)
            show_image(canny_edges, name, save_image=True)

    # Destroy all OpenCV windows in case we have any open
    cv2.destroyAllWindows()

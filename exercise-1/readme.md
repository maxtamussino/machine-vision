# Exercise 1

This exercise implements the Canny edge detection algorithm.

## Structure

The project is separated into several files:

- **main.py**: Contains main function to display detected corners in single image
- **main_timed.py**: To be executed for timing experiment
- **main_experiments.py**: To be executed for automated experiments using different parameters
- **blur_gauss.py**: Gaussian image blurring
- **sobel.py**: Application of sobel filters getting gradient and its orientation
- **non_max.py**: Non-maximum suppression
- **hyst_thresh.py**: Hysteresis thresholding of edges using fixed threshold
- **hyst_auto.py**: Application of hysteresis thresholding using dynamic threshold

## Running the code

When running the file **main.py**, there are multiple adjustable parameters throughout the code:

- `save_image`: Save result images
- `sigma`: Gaussian blurring parameter
- `hyst_method_auto`: Determines whether dynamic threshold selection is used

Additionally, the input image path may be edited in line 33.
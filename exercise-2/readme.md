# Exercise 2

This exercise implements Harris corner detection and the matching of corners using patch descriptors.

## Structure

The project is separated into several files:

- **main_harris_image.py**: Contains main function to display detected corners in single image
- **main_descriptors_image.py**: Contains main function to display matched corners of two images
- **harris_corner.py**: Harris corner detection
- **match_descriptors.py**: Matching of corners in two images using descriptors defined in **descriptor_functions.py**
- **descriptor_functions.py**: Different implementation of patch descriptor calculation
- **helper_functions.py**: Helper functions to display images, show corners and matched corners 

## Running the code

The file **main_harris_image.py** is executed to obtain the corners for one image. It contains the option to set several parameters:

- `sigma1`: First blurring sigma, before derivation
- `sigma2`: Second blurring sigma, after derivation
- `k`: Parameter for calculating corners, see line 75 in **harris_corner.py**
- `threshold`: Corner strength threshold, see line 81 in **harris_corner.py**

For detailed information in Harris corner detection, see https://docs.opencv.org/master/dc/d0d/tutorial_py_features_harris.html.

The file **main_descriptors_image.py** is executed to obtain the corner matching results for two images. It also contains the option to set several parameters:

- `imageset`: Switches between pre-defined sets of images
- `patch_size`: The size of descriptor patches used
- `rotation`: The applied rotation on the second image
- `descriptor_func_ind`: Switches between different descriptor functions (see **descriptor_functions.py**)

Additionally, the parameters described for **main_harris_image.py** may also be set separately in **main_descriptors_image.py**.
# Exercise 3

This exercise implements two-dimensional object recognition and pose estimation using SIFT and Generalized Hough Transform.

## Structure

The project is separated into three files:

- **main.py**: Contains main function, to be executed
- **detect_objects.py**: The implemented function searches the input image for instances of the object and returns their position, rotation and scale
- **helper_functions.py**: Helper functions to display images, draw rectangles around object hypothesis and transform 2D points 

## Running the code

The file **main.py** is executed to obtain the result for one of the images. It contains the option to set several parameters:

- `image_nr`: Choose an image out of `./data`
- `save_image`: Save the result
- `debug_output`: Show intermediate steps

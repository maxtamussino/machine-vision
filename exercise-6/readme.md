# Exercise 6

This exercise implements object detection out of 3D pointcloud data.

## Structure

The project is seperated into the following files:

- **main.py**: Contains main function, to be executed
- **camera_params.py**: Contains values for different parameters of the camera used to take the 3D pointcloud images
- **merge_clusters.py**: Contains the function to merge object clusters using colour histograms
- **sift_matching.py**: Contains functions to use SIFT matching (grey or RGB) for one object image to the scene image
- **categorise_matches.py**: Contains creating and updating the object hypothesis given all matches resulting from SIFT
- **plot_results.py**: Contains functions to plot (and save) images and to plot 3D pointclouds
- **utility.py**: Contains several different functions (projection 3D to 2D, writing object hypothesis text on images, various image masking functions and image cropping)

## Running the code

The file **main.py** contains a function to detect objects out of a specific scene pointcloud. At the bottom, this function is called for either one pointcloud index or for all available pointcloud indices. At the top of the function, multiple debug and feature parameters may be set:

- `debug_text`: Activates simple console status output
- `show_images`: Show interim result images of single steps
- `save_images` Save those images to ./doc/figures/unused
- `timing`: Debug output shows timing information

Additionally, some special features may be enabled and disabled to compare their results:

- `use_colour_matching`: Activates the use of all three RGB colour channels for SIFT matching, which will increase execution time but also detection performance
- `use_cluster_merging`: Activates using of the implemented cluster merging, which will merge pointclouds like the cup, however also introduce additional execution time
- `use_cropping`: Activates cropping of scene and object images to reduce computation time

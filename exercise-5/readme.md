# Exercise 5

This exercise implements plane-fitting in 3D pointcloud data using the RANSAC algorithm and some extensions to it.

## Structure

The project is separated into several files:

- **main.py**: Contains main function, to be executed
- **fit_plane.py**: Contains the implementation of RANSAC and the application of it to obtain multiple planes out of one pointcloud
- **error_funcs.py**: Contains different outlier error functions corresponding to RANSAC and its extensions (MSAC, MLESAC)
- **sac_comparison.py**: Experiments file (to be executed), compares results from RANSAC, MSAC and MLESAC
- **plot_results.py**: Helper functions to display pointclouds

## Running the code

The file **main.py** is executed to obtain the result for one of the pointclouds.  It contains the option to set several RANSAC parameters:

- `pointcloud_idx`: Choose a pointcloud out of `./pointclouds`
- `confidence`: RANSAC confidence
- `inlier_threshold`: Maximum distance d from model to inlier
- `min_sample_distance`: Minimum distance between randomly selected points
- `error_function_idx`: Choose between RANSAC, MSAC and MLESAC
- `voxel_size`: Downsampling grid size

Additionally, the file **sac_comparison.py** may be executed to compare RANSAC, MSAC and MLESAC accuracy.
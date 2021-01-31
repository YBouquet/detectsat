# detectsat
Two approaches for detecting streaks in astronomical images. First approach is based on Hough Transform and Morphological Mathematics, the second one is based on Neural Networks.

## Hough processes
### hough_debug.py
Apply the detection processing on a single block at the address (--subcrop_i, --subcrop_j) in the 4x8 matrix describing the mosaic.

It saves lines detected by Hough and an image diplaying different intermediate results.

Example : 

`python hough_debug.py --i "your_image_path.fits" --o "output_path.png" --subcrop_i 0 --subcrop_j 0 --hough 200`

* --i : fits file location;
* --o : output location, we want to save an image here;
* --subcrop_i : row address in the 4x8 matrix;
* --subcrop_j : columns address in the 4x8 matrix;
* --hough : minimum length of line, line segments shorter than this are rejected.

### hough_full_process.py
Apply the detection processing on the full mosaic by processing blocks in parallel.

It saves lines detected by Hough in one file per block and an image displaying the full mosaic with the final results of the process.

Example:
* If you want to save the lines in a directory : 
`python hough_full_process.py --i "your_image_path.fits" --o "output_path.png" --hough 200`
* If you do not want to save the lines in a directory :
`python hough_full_process.py --i "your_image_path.fits" --o "output_path.png" --no-save_lines --hough 200`
* If you already saved lines in a directory for this output : 
`python hough_full_process.py --i "your_image_path.fits" --o "output_path.png" --load_lines --hough 200`

* --load_lines : load lines detected by Hough instead of running the hough detection process
* --no-save_lines : the lines detected by Hough will not be saved in a directory

## Data Set Generator for NN Training
Generate 64x64 patches with synthetic streaks in it. 

`python synthetic_generator.py --i "your_image_path.fits" --o "label" --n 2`

* --i : the patches will be generated from this original fits image;
* --o : the generator will generate three .npy file, the first one corresponds to the features, the others are the targets. The name of this files will be named after this argument. 
With the example above, the files will be named "label_samples.npy", "label_targets.npy", "label_patch_targets.npy";
* --n : the number of samples we want to generate with a single 64x64 block from the original image. 




# Image Ellipsis Extraction

This script extracts round objects from an image based on their shape similarity to an ellipse with predefined dimensions. It utilizes color segmentation and contour analysis to identify and extract objects that resemble the predefined shape.

## Usage

To use this script, follow the steps below:

1. Make sure you have Python and the required libraries installed (OpenCV, numpy).  
 ```shell
 pip install -r requirements.txt
 ```
2. Place the image you want to process in the same directory as the script.
3. Open a terminal or command prompt in the directory containing the script and image.
4. Run the following command:

```shell
python images_ellipsis.py <image_file>
```

Replace `<image_file>` with the filename of the image you want to process (e.g., `image.jpg`).

## Output

The script will extract round objects from the image that resemble the predefined elliptical shape. The extracted objects will be saved as separate image files in a directory called `round_objects_testing`, which will be created in the same directory as the script.

## Customization

If you want to customize the predefined elliptical shape or color range for segmentation, you can modify the code in the `extract_round_objects` function. Adjust the values in the `lower_silver` and `upper_silver` arrays to define the color range, and modify the dimensions of the ellipse by adjusting the `target_area`, `semi_major_axis`, and `semi_minor_axis` variables.

Please note that this code provides a basic example and might require adjustments or further enhancements to suit your specific use case.

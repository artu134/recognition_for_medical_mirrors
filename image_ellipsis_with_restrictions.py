import math
import cv2
import numpy as np
import sys
import os

def extract_round_objects(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define range for silver color in HSV
    lower_silver = np.array([0, 0, 80])   # Adjust these values
    upper_silver = np.array([255, 80, 255]) # Adjust these values

    # Threshold the HSV image to get only silver colors
    mask = cv2.inRange(hsv, lower_silver, upper_silver)

    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(image, image, mask=mask)

    # Convert the result to grayscale
    gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
    
    # Find contours in the thresholded image    
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    round_objects = []

    # Calculate the area of the target ellipse
    target_area = math.pi * 193 * 127.5

    # Iterate over the contours
    for contour in contours:
        # Approximate the contour to an ellipse
        if len(contour) >= 5:
            ellipse = cv2.fitEllipse(contour)

            # Check that ellipse is valid and calculate its area
            if ellipse[1][0] >= 0 and ellipse[1][1] >= 0:
                ellipse_area = math.pi * (ellipse[1][0] / 2) * (ellipse[1][1] / 2)

                # Compare the ellipse's area with the target area (with some tolerance)
                if 0.7 * target_area <= ellipse_area <= 1.3 * target_area:
                    # Create a new mask for the ellipse
                    ellipse_mask = np.zeros_like(gray)
                    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

                    # Bitwise-and the ellipse mask and the original image
                    round_object = cv2.bitwise_and(image, image, mask=ellipse_mask)

                    round_objects.append(round_object)

    return round_objects

# ... rest of the code ...
def main():
    # Get the image file name from command-line argument
    if len(sys.argv) < 2:
        print("Usage: python images_ellipsis.py <image_file>")
        return

    image_path = sys.argv[1]

    # Extract round objects from the image
    round_objects = extract_round_objects(image_path)

    # Create a directory to save the round objects if it doesn't exist
    output_dir = 'round_objects_testing'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the round objects separately in files
    for i, obj in enumerate(round_objects):
        output_path = os.path.join(output_dir, f"round_object_{i + 1}.jpg")
        cv2.imwrite(output_path, obj)
        print(f"Round Object {i + 1} saved as {output_path}")

    # Display the extracted round objects
    # for i, obj in enumerate(round_objects):
    #     cv2.imshow(f"Round Object {i + 1}", obj)
    #     cv2.waitKey(0)
    #     cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

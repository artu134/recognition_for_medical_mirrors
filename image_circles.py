import cv2
import numpy as np
import sys
import os

#Basic algo for the circles detection 

def extract_round_objects(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Perform adaptive thresholding to extract black and white image
    _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    round_objects = []

    # Iterate over the contours
    for contour in contours:
        # Approximate the contour to a circle
        approx = cv2.approxPolyDP(contour, 0.03 * cv2.arcLength(contour, True), True)

        # If the contour has a circular shape (close to 4 vertices)
        if len(approx) >= 8:
            # Get the center and radius of the minimal enclosing circle
            (x, y), radius = cv2.minEnclosingCircle(contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Create an empty mask to match the image size
            mask = np.zeros_like(image)

            # Draw a white circle on the mask
            cv2.circle(mask, center, radius, (255,255,255), -1)

            # Bitwise-and the mask and the original image
            round_object = cv2.bitwise_and(image, mask)

            # Append the round object to the list
            round_objects.append(round_object)

    return round_objects

def main():
    # Get the image file name from command-line argument
    if len(sys.argv) < 2:
        print("Usage: python round_object_extraction.py <image_file>")
        return

    image_path = sys.argv[1]

    # Extract round objects from the image
    round_objects = extract_round_objects(image_path)

    # Create a directory to save the round objects if it doesn't exist
    output_dir = 'round_objects'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the round objects separately in files
    for i, obj in enumerate(round_objects):
        output_path = os.path.join(output_dir, f"round_object_{i + 1}.jpg")
        cv2.imwrite(output_path, obj)
        print(f"Round Object {i + 1} saved as {output_path}")

    # Display the extracted round objects
    for i, obj in enumerate(round_objects):
        cv2.imshow(f"Round Object {i + 1}", obj)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

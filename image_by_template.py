import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import os

def create_ellipse_mask(image):
    # Create a black image with same dimensions as the input image
    mask = np.zeros_like(image)
    
    # Get the center, width and height of the image
    center = (mask.shape[1]//2, mask.shape[0]//2)
    axes = (mask.shape[1]//2, mask.shape[0]//2)

    # Draw an ellipse onto the mask
    cv2.ellipse(mask, center, axes, 0, 0, 360, (255, 255, 255), -1)
    
    return mask

def extract_ellipse_template(image_path, template_path):
    # Load the image and template
    image = cv2.imread(image_path)
    template = cv2.imread(template_path)

    # Create an ellipse mask for the template
    mask = create_ellipse_mask(template)

    # Apply the mask to the template
    template = cv2.bitwise_and(template, mask)

    # Get width and height of template
    w, h = template.shape[:-1]

    # Perform template matching
    res = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED, mask=mask)
    
    # Threshold for matching
    threshold = 0.8
    
    # Find where the matches are
    loc = np.where(res >= threshold)

    # Create a list of [x, y, x + w, y + h] rectangles
    rectangles = []
    for pt in zip(*loc[::-1]):
        rectangles.append([int(pt[0]), int(pt[1]), int(pt[0] + w), int(pt[1] + h)])

    # Apply non-maxima suppression to the bounding boxes
    rectangles = np.array(rectangles)
    pick = non_max_suppression(rectangles, probs=None, overlapThresh=0.5)
    
    # Create directory to save the matched images if it doesn't exist
    output_dir = 'matched_images'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Extract matched images and save them
    for i, (startX, startY, endX, endY) in enumerate(pick):
        matched = image[startY:endY, startX:endX]
        cv2.imwrite(os.path.join(output_dir, f'matched_{i + 1}.jpg'), matched)



extract_ellipse_template('S__43024404.jpg', 'S__43024399_only_interest_part.jpg')

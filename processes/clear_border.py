"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class ClearBorder(Process):
    def __init__(self, radius):
        self._radius = radius

    def process(self, image, previous_images):
        # Create a copy of the original image
        img = image.copy()

        # Get image dimensions
        w = img.shape[1]
        h = img.shape[0]

        # Find contours of image
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Check to see if any point in the contour is inside the border
            for point in contour:
                # Get the x and y of the point
                p_x = point[0][0]
                p_y = point[0][1]

                # Check if the point is inside the radius of the border
                if ((p_x >= 0 and p_x < self._radius) or (p_x >= w - self._radius - 1 and p_x < w) # X
                    or (p_y >= 0 and p_y < self._radius) or (p_y >= h - self._radius - 1 and p_y < h)): # Y
                    # Draw over the entire contour
                    cv2.drawContours(img, [ contour ], 0, (0, 0, 0), -1)
                    break

        return img
"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class ClearArea(Process):
    def __init__(self, area_threshold):
        self._area_threshold = area_threshold

    def process(self, image, previous_images):
         # Create a copy of the original image
        img = image.copy()

        # Get image dimensions
        w = img.shape[1]
        h = img.shape[0]
        a = w * h

        # Find contours of image
        contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            # Get contour area
            area = cv2.contourArea(contour)

            # Remove contour if the area is less than the threshold
            if area >= 0 and area <= self._area_threshold:
                cv2.drawContours(img, [ contour ], 0, (0, 0, 0), -1)

        return img
"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class Canny(Process):
    def __init__(self, min_threshold=0, max_threshold=255):
        self._min_threshold = min_threshold
        self._max_threshold = max_threshold

    def process(self, image, previous_images):
        return cv2.Canny(image, self._min_threshold, self._max_threshold)

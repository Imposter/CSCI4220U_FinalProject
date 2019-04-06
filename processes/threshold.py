"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
from enum import Enum
import cv2


class ThresholdType(Enum):
    BINARY = cv2.THRESH_BINARY
    BINARY_INV = cv2.THRESH_BINARY_INV
    TRUNC = cv2.THRESH_TRUNC
    TO_ZERO = cv2.THRESH_TOZERO
    TO_ZERO_INV = cv2.THRESH_TOZERO_INV


class ThresholdMethod(Enum):
    DEFAULT = 0
    OTSU = cv2.THRESH_OTSU
    TRIANGLE = cv2.THRESH_TRIANGLE


class Threshold(Process):
    def __init__(self, threshold, max_value=255, threshold_type=ThresholdType.BINARY, method=ThresholdMethod.DEFAULT):
        self._threshold = threshold
        self._max_value = max_value
        self._threshold_type = threshold_type
        self._method = method

    def process(self, image, previous_images):
        r, return_image = cv2.threshold(image, self._threshold, self._max_value,
                                        self._threshold_type.value + self._method.value)
        return return_image

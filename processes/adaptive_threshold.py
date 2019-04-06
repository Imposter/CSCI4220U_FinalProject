"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
from .threshold import ThresholdType
from enum import Enum
import cv2


class AdaptiveThresholdMethod(Enum):
    MEAN_C = cv2.ADAPTIVE_THRESH_MEAN_C
    GAUSSIAN_C = cv2.ADAPTIVE_THRESH_GAUSSIAN_C


class AdaptiveThreshold(Process):
    def __init__(self, block_size, constant, max_value=255, threshold_type=ThresholdType.BINARY,
                 method=AdaptiveThresholdMethod.GAUSSIAN_C):
        self._block_size = block_size
        self._constant = constant
        self._max_value = max_value
        self._threshold_type = threshold_type
        self._method = method

    def process(self, image, previous_images):
        return cv2.adaptiveThreshold(image, self._max_value, self._method.value, self._threshold_type.value,
                                     self._block_size, self._constant)

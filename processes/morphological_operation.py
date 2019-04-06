"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
from enum import Enum
import cv2


class OperationType(Enum):
    OPEN = cv2.MORPH_OPEN


class MorphologicalOperation(Process):
    def __init__(self, operation, kernel):
        self._operation = operation
        self._kernel = kernel

    def process(self, image, previous_images):
        return cv2.morphologyEx(image, self._operation.value, self._kernel)

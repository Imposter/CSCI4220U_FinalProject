"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from abc import ABC, abstractmethod
import cv2

class Process(ABC):
    @abstractmethod
    def process(self, image, previous_images):
        pass

class ImageProcessor:
    def __init__(self, show_stages=False):
        self._show_stages = show_stages
        self._process = []
    
    def add_process(self, f):
        self._process.append(f)

    def process(self, img):
        images = list()
        current_image = img
        if self._show_stages:
            cv2.imshow("Before processing", current_image)
        for f in self._process:
            current_image = f.process(current_image, images)
            images.append(current_image)
            if self._show_stages:
                cv2.imshow(f.__class__.__name__, current_image)
        if self._show_stages:
            cv2.imshow("After processing", current_image)
        return current_image, images

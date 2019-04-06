"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import imutils as im


class Resize(Process):
    def __init__(self, width=None, height=None):
        self._width = width
        self._height = height

        if width is None and height is None:
            raise ValueError("width or height must be specified")

    def process(self, image, previous_images):
        return im.resize(image, width=self._width, height=self._height)

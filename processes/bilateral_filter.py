"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class BilateralFilter(Process):
    def __init__(self, diameter, sigma_color, sigma_space):
        """
        A bilateral filter is a non-linear, edge-preserving, and noise-reducing smoothing filter for images.
        :param diameter: Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, 
        it is computed from sigma_space. 
        :param sigma_color: Filter sigma in the color space. A larger value of the parameter means that farther colors 
        within the pixel neighborhood (see sigma_space) will be mixed together, resulting in larger areas of semi-equal 
        color.
        :param sigma_space: Filter sigma in the coordinate space. A larger value of the parameter means that farther 
        pixels will influence each other as long as their colors are close enough (see sigma_color). When d > 0, it 
        specifies the neighborhood size regardless of sigma_space. Otherwise, d is proportional to sigma_space.
        """
        self.v = 0
        self._diameter = diameter
        self._sigma_color = sigma_color
        self._sigma_space = sigma_space

    def process(self, image, previous_images):
        return cv2.bilateralFilter(image, self._diameter, self._sigma_color, self._sigma_space)

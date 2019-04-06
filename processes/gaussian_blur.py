"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class GaussianBlur(Process):
    def __init__(self, kernel_size, sigma_x=0, sigma_y=0):
        self._kernel_size = kernel_size
        self._sigma_x = sigma_x
        self._sigma_y = sigma_y

    def process(self, image, previous_images):
        return cv2.GaussianBlur(image, self._kernel_size, sigmaX=self._sigma_x, sigmaY=self._sigma_y)
"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2


class HistogramEqualization(Process):
    def process(self, image, previous_images):
        return cv2.equalizeHist(image)

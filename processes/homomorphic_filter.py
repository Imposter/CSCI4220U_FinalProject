"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from processor import Process
import cv2
import numpy as np
import scipy as sc
import scipy.fftpack as fft

# Homomorphic Filtering implementation by Raymond Phan
# GitHub: https://github.com/rayryeng
# From: https://stackoverflow.com/questions/24731810/segmenting-license-plate-characters
class HomomorphicFilter(Process):
    def __init__(self, sigma, gamma_low, gamma_high):
        self._sigma = sigma
        self._gamma_low = gamma_low
        self._gamma_high = gamma_high

    def process(self, image, previous_images):
        # Get number of rows and columns
        rows = image.shape[0]
        columns = image.shape[1]

        # Convert image to float values
        img_float = np.array(image, dtype=float) / 255

        # Perform log(1 + I)
        img_log = np.log1p(img_float)

        # Create gaussian mask
        gaussian_m = 2 * rows + 1
        gaussian_n = 2 * columns + 1
        gaussian_x, gaussian_y = np.meshgrid(np.linspace(0, gaussian_n - 1, gaussian_n), np.linspace(0, gaussian_m - 1, gaussian_m))
        center_x = np.ceil(gaussian_n / 2)
        center_y = np.ceil(gaussian_m / 2)
        gaussian_numerator = (gaussian_x - center_x)**2 + (gaussian_y - center_y)**2
        
        # Low pass and high pass filter
        low_pass = np.exp(-gaussian_numerator / (2 * self._sigma * self._sigma))
        high_pass = 1 - low_pass

        # Move the origin of the filters so that it's at the 
        # top left corner to match with the input image
        low_shift = fft.ifftshift(low_pass.copy())
        high_shift = fft.ifftshift(high_pass.copy())

        # Filter and crop the image
        filtered_image = fft.fft2(img_log.copy(), (gaussian_m, gaussian_n))
        low_image = sc.real(fft.ifft2(filtered_image.copy() * low_shift, (gaussian_m, gaussian_n)))
        high_image = sc.real(fft.ifft2(filtered_image.copy() * high_shift, (gaussian_m, gaussian_n)))

        # Use low pass and high pass image with gamma values to compose output image
        output_image = self._gamma_low*low_image[0:rows, 0:columns] + self._gamma_high*high_image[0:rows, 0:columns]

        # Anti-log then rescale to [0, 1]
        h_image = np.expm1(output_image)
        h_image = (h_image - np.min(h_image)) / (np.max(h_image) - np.min(h_image))
        h_image = np.array(255 * h_image, dtype="uint8")

        return h_image
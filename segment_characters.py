"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import cv2

from processor import ImageProcessor
from processes.grayscale import Grayscale
from processes.resize import Resize
from processes.threshold import Threshold, ThresholdType
from processes.clear_border import ClearBorder
from processes.clear_area import ClearArea
from processes.homomorphic_filter import HomomorphicFilter

# Find contours where vertical height is greater than horizontal height
def find_characters(img, height=64, debug=False):
	# Calculate border and area ratio depending on height constant of 64
	ratio = height / 64

	# Create image processor with processes
	image_processor = ImageProcessor(show_stages=debug)
	image_processor.add_process(Grayscale())
	image_processor.add_process(HomomorphicFilter(sigma=10, gamma_low=0.3, gamma_high=1.5))
	image_processor.add_process(Threshold(threshold=60, threshold_type=ThresholdType.BINARY_INV))
	image_processor.add_process(Resize(height=height))
	image_processor.add_process(ClearBorder(radius=5 * ratio))
	image_processor.add_process(ClearArea(area_threshold=60 * ratio))

	# Preprocess image
	pimg, images = image_processor.process(img)

	results = list() # Characters
	contours, _ = cv2.findContours(pimg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		# Find bounding box around contour
		rect = cv2.boundingRect(contour)

		# Check if height is larger than width
		x, y, w, h = rect
		if h > w:
			# Crop preprocessed image
			char = pimg[y:y+h, x:x+w]

			# Store character bounding box in results
			results.append((rect, char))

	# Sort characters from left to right
	results.sort(key=lambda v: v[0][0]) 
		
	return results
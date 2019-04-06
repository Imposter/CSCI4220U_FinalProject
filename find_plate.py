"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import cv2
from imutils import resize

from processes.adaptive_threshold import AdaptiveThreshold, ThresholdType
from processes.bilateral_filter import BilateralFilter
from processes.grayscale import Grayscale
from processes.histogram_equalization import HistogramEqualization
from processes.invert import Invert
from processor import ImageProcessor

def pyramid(img, factor, min_size):
	img_width = img.shape[1]
	img_height = img.shape[0]
	result = [ (img_width, img_height) ]
	while True:
		n_width = int(img_width * factor)
		n_height = int(img_height * factor)
		if n_width < min_size[0] or n_height < min_size[1]:
			break
		result.append((n_width, n_height))
		img_width = n_width
		img_height = n_height
	return result

def preprocess_image(img, debug=False):
	# Create image processor with processes
	image_processor = ImageProcessor(show_stages=debug)
	image_processor.add_process(Grayscale())
	image_processor.add_process(Invert())
	image_processor.add_process(BilateralFilter(diameter=11, sigma_color=24, sigma_space=24))
	image_processor.add_process(HistogramEqualization())
	image_processor.add_process(AdaptiveThreshold(block_size=11, constant=12, threshold_type=ThresholdType.BINARY_INV))
	
	# Process image
	final_image, images = image_processor.process(img)

	return final_image


def find_rectangles(img, dimensions, dimension_error):
	# Find contours
	results = list() # plates
	contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
	for contour in contours:
		# Approximate the line around the contour
		perimeter = cv2.arcLength(contour, True)
		approximation = cv2.approxPolyDP(contour, 0.05 * perimeter, True)

		if len(approximation) >= 4:
			# Find bounding box around contour
			rect = cv2.boundingRect(contour)
			x, y, w, h = rect

			# Determine aspect ratio
			aspect_ratio = w / h

			# Check if the contour matches a specification
			for spec in dimensions:
				s_aspect_ratio = spec[0] / spec[1]

				# Ensure the error in the aspect ratio is less than the threshold
				if abs(aspect_ratio - s_aspect_ratio) < dimension_error:
					# Store the object
					results.append(rect)
					
	return results

def find_plates(img, plate_dimensions, dimension_error, nearby_threshold, size_threshold, scale=0.5, min_size=(128, 128), debug=False):
	# Get width, height and area
	w = img.shape[1]
	h = img.shape[0]
	a = w * h

	# Get resized image dimensions
	sizes = pyramid(img, scale, min_size)

	# Find rectangles in scaled images
	results = list()
	for sz in sizes:
		# Resize image
		rimg = resize(img, sz[0], sz[1], inter=cv2.INTER_LINEAR)

		# Preprocess image
		pimg = preprocess_image(rimg, debug=debug)

		# Find rectangles in image
		rectangles = find_rectangles(pimg, plate_dimensions, dimension_error)

		# Check to see if rectangles exist in results already
		for r in rectangles:
			# Get rectangle dimensions
			r_x, r_y, r_w, r_h = r

			# Find scale for current image size against original image
			scale = w / sz[0]

			# Scale dimensions
			r_x *= scale
			r_y *= scale
			r_w *= scale
			r_h *= scale

			# Check if it already exists in the results
			exists = False
			for res in results:
				res_x, res_y, res_w, res_h = res
				if (abs(res_x - r_x) / w < nearby_threshold 
					and abs(res_y - r_y) / h < nearby_threshold 
					and abs(res_w - r_w) / w < nearby_threshold 
					and abs(res_h - r_h) / h < nearby_threshold):
					exists = True
			if not exists:
				# Check if the area of the object matches the size threshold
				area = r_w * r_h
				if area / a > size_threshold:
					results.append((r_x, r_y, r_w, r_h))

	return results

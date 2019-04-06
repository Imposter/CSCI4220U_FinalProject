"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import cv2
import numpy as np
from find_plate import find_plates
from segment_characters import find_characters
from ocr import OCR
from imutils import resize

from time import time

def pad(img, pad_x=0.0, pad_y=0.0, color=(255, 255, 255)):
	# Add padding around image
	top = int(pad_y * img.shape[0]) # rows
	bottom = top
	left = int(pad_x * img.shape[1]) # columns
	right = left

	return cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, None, color)

def read_plate(ocr, img, plate_dimensions, dimension_error, nearby_threshold, size_threshold, plate_height=64, debug=False):
	results = list()

	# Locate plates
	plates = find_plates(img, plate_dimensions, dimension_error, nearby_threshold, size_threshold, debug=False)
	for plate in plates:
		plate = tuple(np.array(plate).astype(int))
		p_x, p_y, p_w, p_h = plate

		# Crop image
		pimg = img[p_y:p_y+p_h, p_x:p_x+p_w]
		
		# Resize image
		rimg = resize(pimg, height=plate_height)

		# Get characters in plate
		characters = find_characters(pimg, height=plate_height, debug=debug)

		# Collect characters
		plate_chars = list()
		for c_box, cimg_bw in characters:
			# Convert float tuple to int tuple
			c_box = tuple(np.array(c_box).astype(int))
			c_x, c_y, c_w, c_h = c_box

			# Crop image
			cimg = rimg[c_y:c_y+c_h, c_x:c_x+c_w]

			# Resize images
			cimg = pad(cimg, pad_x=0.05, pad_y=0.0)
			cimg = resize(cimg, height=32)

			plate_chars.append(cimg)
		
		# If there are no characters in the plate, skip
		if not len(plate_chars):
			continue

		# Merge plate characters
		plate_img = np.concatenate(tuple(plate_chars), axis=1)

		# Debug
		if debug:
			cv2.imshow("Plate Characters", plate_img)
			cv2.waitKey()

		# Perform OCR
		predictions = ocr.predict(plate_img)
		
		# Store result
		results.append(predictions)

	return results
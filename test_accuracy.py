"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import cv2
import operator
import numpy as np
from segment_characters import find_characters
from read_plate import pad
from ocr import OCR
from imutils import resize
from os.path import join, splitext, basename
from glob import glob
from argparse import ArgumentParser

from itertools import chain

def find_by_extension(path, extension):
	return glob(join(path, "**." + extension))

def main(args):
	# Get arguments
	dataset_path = args.dataset
	classes = args.classes
	dir_path = args.directory
	height = args.height
	extensions = args.extensions

	# Create OCR instance
	ocr = OCR(dataset_path, classes, cuda=False)

	# Find images in directory
	images = set(chain(*[ find_by_extension(dir_path, e) for e in extensions ]))

	# Counts
	c_empty = 0
	c_no_prediction = 0
	c_length_mismatch = 0
	c_incorrect = list()
	c_correct = list()
	c_char_map = dict()
	c_char_total = 0

	for path in images:
		# Read image and resize
		img = cv2.imread(path)
		img = resize(img, height=height)

		# Plate name
		plate_name = basename(splitext(path)[0]).replace(" ", "")
		print(plate_name)

		# Resize image
		rimg = resize(img, height=64)

		# Get characters in plate
		characters = find_characters(img, height=height, debug=False)

		# If no characters were found, add that to the error
		if not len(characters):
			c_empty += 1
			continue

		# Collect characters
		plate_chars = list()
		for c_box, cimg_bw in characters:
			# Convert float tuple to int tuple
			c_box = tuple(np.array(c_box).astype(int))
			c_x, c_y, c_w, c_h = c_box

			# Crop image
			cimg = rimg[c_y:c_y+c_h, c_x:c_x+c_w]

			# Resize images
			cimg = pad(cimg, pad_x=0.2, pad_y=0.2)
			cimg = resize(cimg, height=32)

			plate_chars.append(cimg)

		# Merge plate characters
		plate_img = np.concatenate(tuple(plate_chars), axis=1)

		# Perform OCR
		predictions = ocr.predict(plate_img)

		# If there are no predictions, skip
		if not len(predictions):
			c_no_prediction += 1
			continue

		# If the prediction size does not match the actual name, record
		pred = "".join([ pred[0] for pred in predictions ])
		if len(pred) != len(plate_name):
			c_length_mismatch += 1
			continue
		
		c_char_total += len(plate_name)
		if pred != plate_name:
			c_incorrect.append((plate_name, pred))

			# Map invalid characters
			for c in pred:
				if c not in c_char_map:
					c_char_map[c] = 1
				else:
					c_char_map[c] += 1
		else:
			c_correct.append((plate_name, pred))
	
	# Print total error
	c_total = len(images)
	print("Empty plates (characters not found):", c_empty, "/", c_total)
	print("No prediction (error):", c_no_prediction, "/", c_total)
	print("Prediction length mismatch (error):", c_length_mismatch, "/", c_total)
	print("Incorrect match (error):", len(c_incorrect), "/", c_total)
	print("Correct match:", len(c_correct), "/", c_total)
	print("")

	# Print N correct plates
	n = min(5, len(c_correct))
	print(n, "correct plate predictions")
	for p in c_correct:
		print("Plate:", p[0], "Prediction:", p[1])
	print("")

	# Print N incorrect plates
	n = n = min(10, len(c_incorrect))
	print(n, "incorrect plate predictions")
	for p in c_incorrect:
		print("Plate:", p[0], "Prediction:", p[1])
	print("")
	
	c_char_map = sorted(c_char_map.items(), key=operator.itemgetter(1), reverse=True)
	
	# Print the top N worst characters
	n = min(len(args.classes), len(c_char_map))
	print("Top", n, "incorrect characters")
	for i in range(0, n):
		p = c_char_map[i]
		print(p[0], "appeared incorrectly", p[1], "/", c_char_total, "=>", str(p[1] / c_char_total * 100) + "%")
	print("")

if __name__ == "__main__":
	# Parse args
	parser = ArgumentParser(description="Plate Generator")
	parser.add_argument("-p", "--directory", type=str, required=True, help="Image directory")
	parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset")
	parser.add_argument("-c", "--classes", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", required=False, help="Classes for dataset")
	parser.add_argument("-s", "--height", type=int, default=64, required=False, help="Image height")
	parser.add_argument("-e", "--extensions", type=str, nargs="+", default=[ "jpg", "png" ], required=False, help="Image extensions")
	args = parser.parse_args()

	main(args)
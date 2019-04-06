"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import cv2
from read_plate import read_plate
from ocr import OCR
from imutils import resize
from os.path import join, basename
from glob import glob
from argparse import ArgumentParser

from time import time
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

	for path in images:
		# Read image and resize
		print("Reading image", path)
		img = cv2.imread(path)
		img = resize(img, height=height)

		start = time()

		# Find plate and perform OCR
		plates = read_plate(ocr, img, plate_dimensions=[
				(300, 150), # Canada/US (mm)
				(520, 114), # European (mm)
				(500, 110), # Netherlands (px)
				(240, 90) # Brazil (px)
			], dimension_error=0.5, # 50%
			nearby_threshold=0.05, # 5%
			size_threshold=0.01, # 1%
			debug=False)
		
		# Skip if there are no predictions
		if not len(plates):
			continue

		# Output predictions
		for p in plates:
			if len(p):
				prediction = "".join([ pred[0] for pred in p ])
				print("Prediction", prediction)
		
		elapsed = time() - start
		print("Took", elapsed, "seconds")

if __name__ == "__main__":
	# Parse args
	parser = ArgumentParser(description="Plate Generator")
	parser.add_argument("-p", "--directory", type=str, required=True, help="Image directory")
	parser.add_argument("-d", "--dataset", type=str, required=True, help="Path to dataset")
	parser.add_argument("-c", "--classes", type=str, default="0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ", required=False, help="Classes for dataset")
	parser.add_argument("-s", "--height", type=int, default=512, required=False, help="Image height")
	parser.add_argument("-e", "--extensions", type=str, nargs="+", default=[ "jpg", "png" ], required=False, help="Image extensions")
	args = parser.parse_args()

	main(args)
"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

from PIL import Image, ImageFont, ImageDraw
from argparse import ArgumentParser
from time import time
import itertools as it
import random as rnd
import numpy as np
import os

class Plate:
    def __init__(self, label, size, border_offset, font_path, font_color=(0, 0, 0), border_color=(0, 0, 0), color=(255, 255, 255)):
        self._label = label
        self._size = size
        self._border_offset = border_offset # percent
        self._font_path = font_path
        self._font_color = font_color
        self._border_color = border_color
        self._color = color

    def render(self):
        # Get plate size
        plate_w, plate_h = self._size

        # Create surface to draw on
        img = Image.new("RGB", (int(plate_w), int(plate_h)), color=self._color)

        # Determine font size using label
        t_font = ImageFont.truetype(font=self._font_path, size=int(plate_h))
        t_size = t_font.getsize(self._label)

        f_w_ratio = plate_w / t_size[0]
        f_h_ratio = plate_h / t_size[1]
        f_h = plate_h * f_h_ratio * f_w_ratio

        # Get font
        font = ImageFont.truetype(font=self._font_path, size=int(f_h))

        # Determine new label location
        l_s = font.getsize(self._label)
        l_x = (plate_w - l_s[0]) / 2
        l_y = (plate_h - l_s[1]) / 2

        # Draw on source image
        draw = ImageDraw.Draw(img)

        # Draw label
        draw.text((l_x, l_y), self._label, fill=self._font_color, font=font)

        # Annotate characters
        c_tx = l_x
        char_boxes = list()
        for c in self._label:
            # Skip spaces
            if c == ' ':
                continue

            # Get character size and offset
            c_size = font.getsize(c)
            c_offset = font.getoffset(c)

            # Draw rectangle around character
            c_x = c_tx + c_offset[0]
            c_y = l_y + c_offset[1]
            c_w = c_x + c_size[0]
            c_h = c_offset[1] + c_size[1]
            char_boxes.append((c, c_x, c_y, c_w, c_h))

            # Increase current character x location
            c_tx += c_size[0]

        # Draw border around image
        b_o_x = plate_w * self._border_offset
        b_o_y = plate_h * self._border_offset
        draw.rectangle([ b_o_x, b_o_y, plate_w - b_o_x, plate_h - b_o_y ], outline=self._border_color)

        return img, char_boxes

def dict_val_reached(d, val):
    result = True
    for k in d.keys():
        if d[k] < val:
            result = False
            break
    return result

def invert_list(l):
    m = max(l)
    for i in range(len(l)):
        l[i] = m - l[i]

def file_write(p, lines):
    with open(p, "w") as f:
        for line in lines:
            f.write(line + "\n")

def get_yolov2_annotations(classes, char_boxes):
    result = list()
    for c_box in char_boxes:
        # Unpack tuple
        c, c_x, c_y, c_w, c_h = c_box

        # Convert to YOLO format
        c_x += c_w / 2.0
        c_y += c_h / 2.0

        result.append("{} {} {} {} {}".format(classes.index(c), c_x, c_y, c_w, c_h))
    return result

# Currently only supports generation of Ontario-like license plates
def main(args):
    # Calculate time
    start_time = time()

    # Arguments
    output_path = args.directory
    num_images = args.images

    # Create output directory if it does not exist
    if not os.path.isdir(output_path):
        os.mkdir(output_path, 0o755)

    # List of characters to permutate
    letters_avail = [ chr(o) for o in range(ord('A'), ord('Z') + 1) ]
    numbers_avail = [ chr(o) for o in range(ord('0'), ord('9') + 1) ]
    classes = list(it.chain(letters_avail, numbers_avail))

    # Set of generated plates
    generated_plates = set()

    # Weight maps
    letter_freq = dict.fromkeys(letters_avail, 0)
    number_freq = dict.fromkeys(numbers_avail, 0)

    # Keep running as long as we haven't met the requirements for letter and number counts
    while not (dict_val_reached(letter_freq, num_images) and dict_val_reached(number_freq, num_images)):
        # Select weights from frequency map
        letter_weights = list(letter_freq.values())
        number_weights = list(number_freq.values())

        # Invert weights
        invert_list(letter_weights)
        invert_list(number_weights)

        # Generate a random combination of letters and numbers (ie. "BFXD 842")
        letters = rnd.choices(letters_avail, weights=letter_weights, k=4)
        numbers = rnd.choices(numbers_avail, weights=number_weights, k=3)

        # Concatenate plate
        label = "".join(letters) + " " + "".join(numbers)

        # Check if plate is already used
        if label in generated_plates:
            continue
        generated_plates.add(label)
        print(label)

        # Increase count for each letter and number used in the label
        for c in label:
            if c != ' ':
                if c in letter_freq:
                    letter_freq[c] += 1
                elif c in number_freq:
                    number_freq[c] += 1

        # Generate image
        p = Plate(label, np.array((30, 15)) * 5.0, border_offset=0.01, font_path=args.font, font_color=(0, 75, 175), border_color=(0, 0, 0))
        p_img, p_char_boxes = p.render()

        # Save image in output directory
        p_img.save(output_path + "/" + label + ".jpg")

        # Write annotated information to a file
        annotations = get_yolov2_annotations(classes, p_char_boxes)
        file_write(output_path + "/" + label + ".txt", annotations)
        
    # Write classes to file
    file_write(output_path + "/classes.txt", classes)
        
    elapsed_time = time() - start_time

    print("Generated", len(generated_plates), "plates")
    print("Completed in " + str(elapsed_time) + "s")

if __name__ == "__main__":
    # Parse args
    parser = ArgumentParser(description="Plate Generator")
    parser.add_argument("-f", "--font", type=str, required=True, help="Font path")
    parser.add_argument("-i", "--images", type=int, required=True, help="Amount of images for each character")
    parser.add_argument("-d", "--directory", type=str, required=True, help="Image output directory")
    args = parser.parse_args()

    main(args)
"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import torch
import torch.cuda
import cv2
import numpy as np
from torch.autograd import Variable
from torch.nn import Softmax
from torchvision.transforms import ToTensor
from PIL import Image
from imutils import resize
from model import CRNN

class OCR():
    def __init__(self, path, classes, image_width=120, image_height=32, cuda=True):
        # Initialize model
        model = CRNN(image_height=image_height, num_channels=1, num_classes=len(classes) + 1, rnn_hidden_size=256)

        # Use CUDA if available
        if cuda and torch.cuda.is_available():
            model = model.cuda()
        
        # Load states from model path
        model.load_state_dict(torch.load(path))

        # Set model to evaluation mode
        model.eval()

        # Set instance vars
        self._model = model
        self._classes = classes
        self._to_tensor = ToTensor()
        self._image_width = image_width
        self._image_height = image_height
        self._cuda = cuda

    def predict(self, img):
        # Convert to pillow image
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img).convert("L")

        # Resize image
        img = img.resize((self._image_width, self._image_height), Image.BILINEAR)
        
        # Transform image to tensor
        t = self._to_tensor(img)
        t.sub_(0.5).div_(0.5)

        # Use CUDA if available
        if self._cuda and torch.cuda.is_available():
            t = t.cuda()
        
        # Convert tensor to variable and evaluate predictions
        t = t.view(1, *t.size())
        v = Variable(t)
        preds = self._model(v)

        # Perform softmax and get confidence
        softmax = Softmax(dim=1)       

        # Get confidence for each character
        char_confidence = list()
        for i in range(len(preds)):
            prob = torch.max(softmax(preds[i]))
            char_confidence.append(prob)

        # Get predictions
        _, preds = preds.max(2)
        preds = preds.transpose(1, 0).contiguous().view(-1)

        # Get prediction data from source device
        preds_data = None
        if self._cuda and torch.cuda.is_available():
            preds_data = preds.data.cpu().numpy()
        else:
            preds_data = preds.data.numpy()

        # Get raw predictions
        predictions = list()
        for i in range(len(preds_data)):
            c = preds_data[i]
            if c != 0:
                predictions.append((self._classes[c - 1], char_confidence[i].item()))
        
        return predictions
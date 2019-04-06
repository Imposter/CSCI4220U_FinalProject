"""
License Plate Recognition
CSCI4220U: Computer Vision - Final Project
Authors: Eyaz Rehman (100584735), Rishabh Patel (100583380)
Date: April 5th, 2019
""" 

import torch.nn as nn

# Model is based on research by Baoguang Shi, Xiang Bai, and Cong Yao
# Original Paper: https://arxiv.org/abs/1507.05717
# From: https://github.com/meijieru/crnn.pytorch/blob/master/models/crnn.py
class BidirectionalLSTM(nn.Module):
    def __init__(self, channels_in, rnn_hidden_size, channels_out):
        super(BidirectionalLSTM, self).__init__()

        self.rnn = nn.LSTM(channels_in, rnn_hidden_size, bidirectional=True)
        self.embedding = nn.Linear(rnn_hidden_size * 2, channels_out)

    def forward(self, input):
        recurrent, _ = self.rnn(input)
        T, b, h = recurrent.size()
        t_rec = recurrent.view(T * b, h)

        output = self.embedding(t_rec)  # [T * b, channels_out]
        output = output.view(T, b, -1)

        return output

class CRNN(nn.Module):
    def conv_relu(self, cnn, i, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, batch_normalize):
            channels_in = output_size[i - 1]
            if i == 0:
                channels_in = num_channels
            channels_out = output_size[i]
            cnn.add_module("conv{0}".format(i), nn.Conv2d(channels_in, channels_out, kernel_size[i], stride_size[i], padding_size[i]))
            if batch_normalize:
                cnn.add_module("batchnorm{0}".format(i), nn.BatchNorm2d(channels_out))
            if leaky_relu:
                cnn.add_module("relu{0}".format(i), nn.LeakyReLU(0.2, inplace=True))
            else:
                cnn.add_module("relu{0}".format(i), nn.ReLU(True))


    def __init__(self, image_height, num_channels, num_classes, rnn_hidden_size, leaky_relu=False):
        super(CRNN, self).__init__()
        assert image_height % 16 == 0, "image_height has to be a multiple of 16"

        kernel_size = [3, 3, 3, 3, 3, 3, 2]
        padding_size = [1, 1, 1, 1, 1, 1, 0]
        stride_size = [1, 1, 1, 1, 1, 1, 1]
        output_size = [64, 128, 256, 256, 512, 512, 512]

        cnn = nn.Sequential()

        self.conv_relu(cnn, 0, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, False)
        cnn.add_module("pooling0", nn.MaxPool2d(2, 2)) # 64x16x64
        self.conv_relu(cnn, 1, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, False)
        cnn.add_module("pooling1", nn.MaxPool2d(2, 2)) # 128x8x32
        self.conv_relu(cnn, 2, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, True)
        self.conv_relu(cnn, 3, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, False)
        cnn.add_module("pooling2", nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 256x4x16
        self.conv_relu(cnn, 4, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, True)
        self.conv_relu(cnn, 5, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, False)
        cnn.add_module("pooling3", nn.MaxPool2d((2, 2), (2, 1), (0, 1))) # 512x2x16
        self.conv_relu(cnn, 6, num_channels, kernel_size, padding_size, stride_size, output_size, leaky_relu, True)  # 512x1x16

        self.cnn = cnn
        self.rnn = nn.Sequential(
            BidirectionalLSTM(512, rnn_hidden_size, rnn_hidden_size),
            BidirectionalLSTM(rnn_hidden_size, rnn_hidden_size, num_classes)
        )

    def forward(self, x):
        # CNN Features
        conv = self.cnn(x)
        b, c, h, w = conv.size()
        assert h == 1, "the height of conv must be 1"
        conv = conv.squeeze(2)
        conv = conv.permute(2, 0, 1)  # [w, b, c]

        # RNN Features
        output = self.rnn(conv)

        return output
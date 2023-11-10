#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Edge detection with the Sobel filter

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np


def sobel(img: np.array) -> (np.array, np.array):
    """ Apply the Sobel filter to the input image and return the gradient and the orientation.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    :return: (gradient, orientation): gradient: edge strength of the image in range [0.,1.],
                                      orientation: angle of gradient in range [-np.pi, np.pi]
    :rtype: (np.array, np.array)
    """
    ######################################################
    # Write your own code here

    g_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    g_y = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])

    # Initialize result arrays
    gradient = np.zeros_like(img, dtype=np.float32)
    orientation = np.zeros_like(img, dtype=np.float32)

    # Apply Sobel filters
    height, width = np.shape(img)
    for i in range(height-2):
        for j in range(width-2):
            x = np.sum(np.multiply(g_x, img[i:i + 3, j:j + 3])) 
            y = np.sum(np.multiply(g_y, img[i:i + 3, j:j + 3]))
            gradient[i, j] = np.sqrt(x**2 + y**2)
            orientation[i, j] = np.arctan(y/x)

    
    ######################################################
    # Sources:
    # https://github.com/adamiao/sobel-filter-tutorial/blob/master/sobel_from_scratch.py
    # https://www.adeveloperdiary.com/data-science/computer-vision/how-to-implement-sobel-edge-detection-using-python-from-scratch/
    ######################################################
    return gradient, orientation

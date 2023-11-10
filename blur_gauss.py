#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Blur the input image with Gaussian filter kernel

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np
import math

def blur_gauss(img: np.array, sigma: float) -> np.array:
    """ Blur the input image with a Gaussian filter with standard deviation of sigma.

    :param img: Grayscale input image
    :type img: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param sigma: The standard deviation of the Gaussian kernel
    :type sigma: float

    :return: Blurred image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0.,1.]
    """
    ######################################################

    # Given kernel width
    kernel_width = 2*math.ceil(3*sigma)+1        # ceil: rounding

    # Initialize 2D Numpy array filled with ones
    kernel = np.ones((kernel_width, kernel_width), dtype = np.float32)/9
    
    # Find center of kernel
    center = kernel_width // 2

    # Compute filter
    for i in range(kernel_width):
        for j in range(kernel_width):

            x, y = i - center, j - center  
            base = 1/ (2 * np.pi * sigma**2)
            exponent = np.exp( - (x**2 + y**2) / (2* sigma**2))

            kernel[i, j] = base * exponent
    
    # Initialize blurred_image
    img_blur = np.zeros_like(img, dtype=np.float32)

    # Apply gaussian filter
    img_blur = cv2.filter2D(img, -1, kernel)

    ######################################################
    # Sources:
    # https://medium.com/@rohit-krishna/coding-gaussian-blur-operation-from-scratch-in-python-f5a9af0a0c0f
    # https://medium.com/@akumar5/computer-vision-gaussian-filter-from-scratch-b485837b6e09
    # https://medium.com/spinor/a-straightforward-introduction-to-image-blurring-smoothing-using-python-f8870cf1096
    ######################################################
    return img_blur

#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Automatic hysteresis thresholding

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np

from hyst_thresh import hyst_thresh


def hyst_thresh_auto(edges_in: np.array, low_prop: float, high_prop: float) -> np.array:
    """ Apply automatic hysteresis thresholding.

    Apply automatic hysteresis thresholding by automatically choosing the high and low thresholds of standard
    hysteresis threshold. low_prop is the proportion of edge pixels which are above the low threshold and high_prop is
    the proportion of pixels above the high threshold.

    :param edges_in: Edge strength of the image in range [0., 1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low_prop: Proportion of pixels which should lie above the low threshold
    :type low_prop: float in range [0., 1.]

    :param high_prop: Proportion of pixels which should lie above the high threshold
    :type high_prop: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
   # Compute histogramm
    edges = edges_in * 255

    # Filter out zeros as they are background pixels
    hist, bins = np.histogram(edges, bins=255, range=(1, 255))
    
    # Calculate total number of pixels
    total_pixels = sum(hist)


    # Loop over histogram until the accumulate number of pixels reaches the high threshold
    accumulated = 0
    i = 0
    while accumulated < (total_pixels * high_prop):
            accumulated += hist[i]
            i += 1
    else:    
        high_threshold = bins[i]/255 
    
     # Loop over histogram until the accumulate number of pixels reaches the lower threshold
    accumulated = 0
    i = 0
    while accumulated < (total_pixels * low_prop):
            accumulated += hist[i]
            i += 1
    else:    
        low_threshold = bins[i]/255
    
    print(low_threshold, high_threshold)
    hyst_out = hyst_thresh(edges_in, low_threshold, high_threshold)

    ######################################################
    return hyst_out

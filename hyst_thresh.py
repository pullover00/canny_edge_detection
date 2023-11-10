#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Hysteresis thresholding

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np


def hyst_thresh(edges_in: np.array, low: float, high: float) -> np.array:
    """ Apply hysteresis thresholding.

    Apply hysteresis thresholding to return the edges as a binary image. All connected pixels with value > low are
    considered a valid edge if at least one pixel has a value > high.

    :param edges_in: Edge strength of the image in range [0.,1.]
    :type edges_in: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param low: Value below all edges are filtered out
    :type low: float in range [0., 1.]

    :param high: Value which a connected element has to contain to not be filtered out
    :type high: float in range [0., 1.]

    :return: Binary edge image
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values either 0 or 1
    """
    ######################################################
    
    # Create a binary mask for pixels above the high threshold
    high_mask = (edges_in > high).astype(np.float32)
    
    # Use OpenCV's connectedComponents to find connected edge segments
    num_labels, labels = cv2.connectedComponents(high_mask.astype(np.uint8), connectivity=8)

    # Initialize the binary edge image
    bitwise_img = np.zeros_like(edges_in)
    
    # Create a queue for BFS
    queue = []
    
    for label in range(1, num_labels):
        # Create a mask for the current label
        label_mask = (labels == label).astype(np.float32)
        
        # Mark the ridge pixels as visited edges
        if np.max(label_mask * edges_in) >= high:
            bitwise_img = np.maximum(bitwise_img, label_mask)
        
        # Find the coordinates of the ridge pixels for BFS
        ridge_pixels = np.argwhere(label_mask * edges_in >= low)
        
        # Add the ridge pixels to the queue
        queue.extend(ridge_pixels.tolist())

    # Run BFS
    while queue:
        i, j = queue.pop(0)
        
        # Check adjacent pixels
        for x, y in [(i-1, j), (i+1, j), (i, j-1), (i, j+1)]:
            if 0 <= x < edges_in.shape[0] and 0 <= y < edges_in.shape[1]:
                if bitwise_img[x, y] == 0 and edges_in[x, y] >= low:
                    bitwise_img[x, y] = 1
                    queue.append((x, y))
    
    ######################################################
    # Sources:
    # https://stackoverflow.com/questions/64469190/hysteresis-thresholding-in-python
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    ######################################################

    return bitwise_img

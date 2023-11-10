#!/usr/bin/env python
# -*- coding: utf-8 -*-

""" Non-Maxima Suppression

Author: FILL IN
MatrNr: FILL IN
"""

import cv2
import numpy as np


def non_max(gradients: np.array, orientations: np.array) -> np.array:
    """ Apply Non-Maxima Suppression and return an edge image.

    Filter out all the values of the gradients array which are not local maxima.
    The orientations are used to check for larger pixel values in the direction of orientation.

    :param gradients: Edge strength of the image in range [0.,1.]
    :type gradients: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]

    :param orientations: angle of gradient in range [-np.pi, np.pi]
    :type orientations: np.array with shape (height, width) with dtype = np.float32 and values in the range [-pi, pi]

    :return: Non-Maxima suppressed gradients
    :rtype: np.array with shape (height, width) with dtype = np.float32 and values in the range [0., 1.]
    """
    ######################################################
    edges = np.zeros_like(gradients, dtype=np.float32)

    # Transformation into degrees
    angles = orientations * 180. / np.pi
    angles[angles < 0] += 180

    # 4 different angle cases
    height, width = np.shape(gradients)
    for i in range(1, height - 1):
        for j in range(1, width - 1):
            # Angles 0 deg or 180 deg
            if (0 <= angles[i,j] < 22.5) or (157.5 <= angles[i,j] < 202.5) or (337.5 <= angles[i,j] < 360):
                    local_max = max(gradients[i, j-1], gradients[i, j+1])      
            # Angles 45 deg or 225 deg
            elif (22.5 <= angles[i,j] < 67.5) or (202.5 <= angles[i,j] < 247.5 ):
                local_max = max(gradients[i+1, j-1], gradients[i-1, j+1])
            # Angles 90 deg or 270 deg
            elif (67.5 <= angles[i,j] < 112.5) or (247.5 <= angles[i,j] < 292.5 ):
                local_max = max(gradients[i-1, j], gradients[i+1, j])
            # Angles 135 deg or 315
            else:
                local_max = max(gradients[i-1, j-1], gradients[i+1, j+1])

            if gradients[i, j] >= local_max:
                edges[i, j] = gradients[i, j]
       
   
       ######################################################
    # Sources:
    # https://github.com/StefanPitur/Edge-detection---Canny-detector/blob/master/canny.py
    # https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123
    ######################################################

    return edges

from helper_functions import *
from pathlib import Path
import cv2
import numpy as np
from blur_gauss import blur_gauss

# Define behavior of the show_image function. You can change these variables if necessary
save_image = False
matplotlib_plotting = False

# Read image. You can change the image you want to process here.
current_path = Path(__file__).parent
img_gray = cv2.imread(str(current_path.joinpath("image/beardman.jpg")), cv2.IMREAD_GRAYSCALE)
if img_gray is None:
    raise FileNotFoundError("Couldn't load image in " + str(current_path))

 # Before we start working with the image, we convert it from uint8 with range [0,255] to float32 with range [0,1]
img_gray = img_gray.astype(np.float32) / 255.

sigmas = [1, 5, 15]  # Change this value

for sigma in sigmas:
    img_blur = blur_gauss(img_gray, sigma)
    plot_row_intensities(img_blur, 500)


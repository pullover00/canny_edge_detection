import cv2
import numpy as np

# Load an image
image = cv2.imread("image/rubens.jpg", cv2.IMREAD_GRAYSCALE)

# Define a range of kernel sizes and sigma values
kernel_sizes = [3, 9, 15]  # Try different kernel sizes
sigma_values = [1, 10 , 20]  # Try different sigma values

for kernel_size in kernel_sizes:
    for sigma in sigma_values:
        # Calculate the standard deviation from sigma
        kernel_width = kernel_size
        sigma = sigma

        # Create a Gaussian kernel
        kernel = cv2.getGaussianKernel(kernel_width, sigma)

        # Apply Gaussian smoothing using cv2.filter2D
        smoothed_image = cv2.filter2D(image, -1, kernel)

        # Display or save the smoothed images for analysis
        cv2.imshow(f'Kernel Size {kernel_size} Sigma {sigma}', smoothed_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
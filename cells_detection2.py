import os
import numpy as np
import scipy.ndimage



def gaussian_filter(image, sigma):
    
    H, W = image.shape
    kernel_size = int(2 * np.ceil(2 * sigma) + 1)
    kernel_size = min(kernel_size, min(H, W) // 2)
    if kernel_size % 2 == 0:
        kernel_size = kernel_size + 1

    # Create a 2D Gaussian kernel using np.meshgrid and np.exp
    x = np.arange(-kernel_size // 2, kernel_size // 2 + 1)
    xx, yy = np.meshgrid(x, x)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= np.sum(kernel)

    # Apply the Gaussian filter using scipy.ndimage.convolve
    filtered_image = scipy.ndimage.convolve(image, kernel, mode='reflect')

    return filtered_image


def main():

if __name__ == '__main__':
    main()
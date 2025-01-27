import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use same padding (mode = 'reflect'). Refer docs for further info.

from common import (find_maxima, read_img, visualize_maxima,
                    visualize_scale_space)


def gaussian_filter(image, sigma):
    """
    Given an image, apply a Gaussian filter with the input kernel size
    and standard deviation

    Input
      image: image of size HxW
      sigma: scalar standard deviation of Gaussian Kernel

    Output
      Gaussian filtered image of size HxW
    """
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
    image = read_img('polka.png')

    # Single-scale Blob Detection 
    # Detecting Polka Dots

    # First, complete gaussian_filter()
    print("Detecting small polka dots")
    sigma_1, sigma_2 = 2, 4  # Choose appropriate sigma values for small circles
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # Calculate difference of Gaussians
    DoG_small = gauss_2 - gauss_1

    # Visualize maxima
    maxima = find_maxima(DoG_small, k_xy=10)
    visualize_scale_space(DoG_small, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_small_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_small.png')

    #  Detect Large Circles
    print("Detecting large polka dots")
    sigma_1, sigma_2 = 10, 20  # Choose appropriate sigma values for large circles
    gauss_1 = gaussian_filter(image, sigma_1)
    gauss_2 = gaussian_filter(image, sigma_2)

    # Calculate difference of Gaussians
    DoG_large = gauss_2 - gauss_1

    # Visualize maxima
    maxima = find_maxima(DoG_large, k_xy=10)
    visualize_scale_space(DoG_large, sigma_1, sigma_2 / sigma_1,
                          './polka_detections/polka_large_DoG.png')
    visualize_maxima(image, maxima, sigma_1, sigma_2 / sigma_1,
                     './polka_detections/polka_large.png')


if __name__ == '__main__':
    main()

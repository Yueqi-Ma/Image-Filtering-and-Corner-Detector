import os

import numpy as np
import scipy.ndimage
# Use scipy.ndimage.convolve() for convolution.
# Use zero padding (Set mode = 'constant'). Refer docs for further info.

from common import read_img, save_img



def corner_score(image, u=5, v=5, window_size=(5, 5)):
    """
    Given an input image, x_offset, y_offset, and window_size,
    return the function E(u,v) for window size W
    corner detector score for that pixel.
    Use zero-padding to handle window values outside of the image.

    Input- image: H x W
           u: a scalar for x offset
           v: a scalar for y offset
           window_size: a tuple for window size

    Output- results: an image of size H x W
    """
    # Offset the image by (u, v)
    shifted_image = np.roll(image, (u, v), axis=(0, 1))

    # Calculate the squared difference between the original image and the shifted image
    diff = (image - shifted_image) ** 2

    # Create a window of ones with the specified size
    window = np.ones(window_size)

    # Perform convolution with the window to compute the sum of squared differences within the window
    corner_score = scipy.ndimage.convolve(diff, window, mode='constant', cval=0.0)

    return corner_score


def harris_detector(image, window_size=(5, 5)):
    """
    Given an input image, calculate the Harris Detector score for all pixels
    You can use same-padding for intensity (or 0-padding for derivatives)
    to handle window values outside of the image.

    Input- image: H x W
    Output- results: an image of size H x W
    """
    # Compute the derivatives using Sobel operators
    Ix = scipy.ndimage.convolve(image, np.array([[-1, 0, 1]]), mode='constant', cval=0.0)
    Iy = scipy.ndimage.convolve(image, np.array([[-1], [0], [1]]), mode='constant', cval=0.0)

    # Calculate the squared derivatives
    Ixx = Ix ** 2
    Iyy = Iy ** 2
    Ixy = Ix * Iy

    # Create the structure tensor components by convolving with a window
    window = np.ones(window_size)
    Sxx = scipy.ndimage.convolve(Ixx, window, mode='constant', cval=0.0)
    Syy = scipy.ndimage.convolve(Iyy, window, mode='constant', cval=0.0)
    Sxy = scipy.ndimage.convolve(Ixy, window, mode='constant', cval=0.0)

    # Calculate the Harris response
    k = 0.04  # Harris detector constant
    det = Sxx * Syy - Sxy ** 2
    trace = Sxx + Syy
    response = det - k * trace ** 2

    return response


def main():
    img = read_img('./grace_hopper.png')

    # Feature Detection
    if not os.path.exists("./feature_detection"):
        os.makedirs("./feature_detection")

    # -- TODO Task 6: Corner Score --
    # (a): Complete corner_score()

    # (b)
    # Define offsets and window size and calculate corner score
    u, v, W = 0, 5, (5, 5)

    score = corner_score(img, u, v, W)
    save_img(score, "./feature_detection/corner_score.png")

    score = corner_score(img, 0, 5, W)
    save_img(score, "./feature_detection/corner_score05.png")

    score = corner_score(img, 0, -5, W)
    save_img(score, "./feature_detection/corner_score0-5.png")

    score = corner_score(img, 5, 0, W)
    save_img(score, "./feature_detection/corner_score50.png")

    score = corner_score(img, -5, 0, W)
    save_img(score, "./feature_detection/corner_score-50.png")

    # (c): No Code

    # -- TODO Task 7: Harris Corner Detector --
    # (a): Complete harris_detector()

    # (b)
    harris_corners = harris_detector(img)
    save_img(harris_corners, "./feature_detection/harris_response.png")


if __name__ == "__main__":
    main()
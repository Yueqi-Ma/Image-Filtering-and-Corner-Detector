import os

import numpy as np

from common import read_img, save_img


def image_patches(image, patch_size=(16, 16)):
    """
    Given an input image and patch_size,
    return the corresponding image patches made
    by dividing up the image into patch_size sections.

    Input- image: H x W
           patch_size: a scalar tuple M, N
    Output- results: a list of images of size M x N
    """
    output = []

    patch_height, patch_width = patch_size

    height, width = image.shape[:2]

    for i in range(0, height, patch_height):
        for j in range(0, width, patch_width):
            patch = image[i:i+patch_height, j:j+patch_width]
            output.append(patch)

    return output



def convolve(image, kernel):
    """
    Return the convolution result: image * kernel.
    Reminder to implement convolution and not cross-correlation!
    Caution: Please use zero-padding.

    Input- image: H x W
           kernel: h x w
    Output- convolve: H x W
    """
    kernel_height, kernel_width = kernel.shape
    image_height, image_width = image.shape

    # Calculate the padding size
    pad_height = kernel_height // 2
    pad_width = kernel_width // 2

    # Pad the image with zeros
    padded_image = np.pad(image, ((pad_height, pad_height), (pad_width, pad_width)), mode='constant')

    # Initialize the output image
    output = np.zeros_like(image)

    # Perform convolution
    for i in range(image_height):
        for j in range(image_width):
            patch = padded_image[i:i + kernel_height, j:j + kernel_width]
            output[i, j] = np.sum(patch * kernel)

    return output



def edge_detection(image):
    """
    Return Ix, Iy and the gradient magnitude of the input image

    Input- image: H x W
    Output- Ix, Iy, grad_magnitude: H x W
    """
    # Define the Sobel kernels for gradient calculation
    kx = np.array([[-1, 0, 1]])
    ky = np.array([[-1], [0], [1]])

    # Compute horizontal and vertical gradients
    Ix = convolve(image, kx)
    Iy = convolve(image, ky)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(Ix**2 + Iy**2)

    return Ix, Iy, grad_magnitude

def gaussian_kernel(size, sigma):
    """
    Generate a Gaussian kernel of the specified size and standard deviation (sigma).

    Input:
    - size: An integer specifying the size of the kernel (should be odd).
    - sigma: A float specifying the standard deviation of the Gaussian distribution.

    Output:
    - kernel: A 2D NumPy array representing the Gaussian kernel.
    """
    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma**2)) * np.exp(
        -((x - size//2)**2 + (y - size//2)**2) / (2 * sigma**2)), (size, size))
    kernel = kernel / np.sum(kernel)  # Normalize the kernel
    return kernel


    
def sobel_operator(image):
    """
    Return Gx, Gy, and the gradient magnitude.

    Input- image: H x W
    Output- Gx, Gy, grad_magnitude: H x W
    """
    # Define the Sobel kernels
    kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    ky = np.array([[-1, -2, -1],
                   [0, 0, 0],
                   [1, 2, 1]])

    # Compute horizontal and vertical gradients using convolution
    Gx = convolve(image, kx)
    Gy = convolve(image, ky)

    # Compute gradient magnitude
    grad_magnitude = np.sqrt(Gx**2 + Gy**2)

    return Gx, Gy, grad_magnitude


def main():
    # The main function
    img = read_img('./grace_hopper.png')
    """ Image Patches """
    if not os.path.exists("./image_patches"):
        os.makedirs("./image_patches")

    # -- TODO Task 1: Image Patches --
    # (a)
    # First complete image_patches()
    patches = image_patches(img)
    # Now choose any three patches and save them
    # chosen_patches should have those patches stacked vertically/horizontally
    chosen_patches = np.vstack((patches[0], patches[1], patches[2]))
    save_img(chosen_patches, "./image_patches/q1_patch.png")

    # (b), (c): No code

    """ Convolution and Gaussian Filter """
    if not os.path.exists("./gaussian_filter"):
        os.makedirs("./gaussian_filter")

    # -- TODO Task 2: Convolution and Gaussian Filter --
    # (a): No code

    # (b): Complete convolve()

    # (c)
    # Calculate the Gaussian kernel described in the question.
    # There is tolerance for the kernel.
    kernel_gaussian = gaussian_kernel(3, 1)
    filtered_gaussian = convolve(img, kernel_gaussian)
    save_img(filtered_gaussian, "./gaussian_filter/q2_gaussian.png")

    # (d), (e): No code

    # (f): Complete edge_detection()

    # (g)
    # Use edge_detection() to detect edges
    # for the original and Gaussian filtered images.
    _, _, edge_detect = edge_detection(img)
    save_img(edge_detect, "./gaussian_filter/q3_edge.png")
    _, _, edge_with_gaussian = edge_detection(filtered_gaussian)
    save_img(edge_with_gaussian, "./gaussian_filter/q3_edge_gaussian.png")

    print("Gaussian Filter is done. ")

    # -- TODO Task 3: Sobel Operator --
    if not os.path.exists("./sobel_operator"):
        os.makedirs("./sobel_operator")

    # (a): No code

    # (b): Complete sobel_operator()

    # (c)
    Gx, Gy, edge_sobel = sobel_operator(img)
    save_img(Gx, "./sobel_operator/q2_Gx.png")
    save_img(Gy, "./sobel_operator/q2_Gy.png")
    save_img(edge_sobel, "./sobel_operator/q2_edge_sobel.png")

    print("Sobel Operator is done. ")

    # -- TODO Task 4: LoG Filter --
    if not os.path.exists("./log_filter"):
        os.makedirs("./log_filter")

    # (a)
    kernel_LoG1 = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    kernel_LoG2 = np.array([[0, 0, 3, 2, 2, 2, 3, 0, 0],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [2, 5, 0, -23, -40, -23, 0, 5, 2],
                            [2, 5, 3, -12, -23, -12, 3, 5, 2],
                            [3, 3, 5, 3, 0, 3, 5, 3, 3],
                            [0, 2, 3, 5, 5, 5, 3, 2, 0],
                            [0, 0, 3, 2, 2, 2, 3, 0, 0]])
    filtered_LoG1 = convolve(img, kernel_LoG1)
    filtered_LoG2 = convolve(img, kernel_LoG2)
    # Use convolve() to convolve img with kernel_LOG1 and kernel_LOG2
    save_img(filtered_LoG1, "./log_filter/q1_LoG1.png")
    save_img(filtered_LoG2, "./log_filter/q1_LoG2.png")

    # (b)
    # Follow instructions in pdf to approximate LoG with a DoG
    print("LoG Filter is done. ")


if __name__ == "__main__":
    main()
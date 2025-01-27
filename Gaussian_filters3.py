import cv2
import numpy as np
import matplotlib.pyplot as plt

def read_img(file_path):
    img = cv2.imread(file_path, 0)
    return img

def save_img(img, file_path):
    cv2.imwrite(file_path, img)

def edge_detection(img):
    sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    grad_magnitude = np.sqrt(sobelx**2 + sobely**2)
    grad_magnitude = (grad_magnitude / np.max(grad_magnitude)) * 255
    grad_magnitude = grad_magnitude.astype(np.uint8)
    return sobelx, sobely, grad_magnitude

def gaussian_filter(img, kernel_size=5, sigma=1.4):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), sigma)

def main():
    # Load the original image
    original_image = read_img('./grace_hopper.png')

    # Apply Gaussian filter
    filtered_gaussian = gaussian_filter(original_image)

    # Perform edge detection on the original image and filtered image
    _, _, grad_magnitude_original = edge_detection(original_image)
    _, _, grad_magnitude_filtered = edge_detection(filtered_gaussian)

    # Save the filtered image
    save_img(filtered_gaussian, "./gaussian_filter/q3_gaussian_filtered.png")

    # Plot the results
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 2, 1)
    plt.imshow(original_image, cmap='gray')
    plt.title('Original Image')

    plt.subplot(2, 2, 2)
    plt.imshow(grad_magnitude_original, cmap='gray')
    plt.title('Gradient Magnitude (Original)')

    plt.subplot(2, 2, 3)
    plt.imshow(filtered_gaussian, cmap='gray')
    plt.title('Gaussian Filtered Image')

    plt.subplot(2, 2, 4)
    plt.imshow(grad_magnitude_filtered, cmap='gray')
    plt.title('Gradient Magnitude (Filtered)')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()
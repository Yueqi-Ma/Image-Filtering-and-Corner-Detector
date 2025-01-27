import matplotlib.pyplot as plt

def main():
    image = read_img('polka.png')

    # ...

    # -- Task 8: Single-scale Blob Detection --

    # (a), (b): Detecting Polka Dots

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

    # Plot DoG response
    plt.figure()
    plt.imshow(DoG_small, cmap='gray')
    plt.title('DoG Response (Small Polka Dots)')
    plt.colorbar()
    plt.show()

    # -- Detect Large Circles
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

    # Plot DoG response
    plt.figure()
    plt.imshow(DoG_large, cmap='gray')
    plt.title('DoG Response (Large Polka Dots)')
    plt.colorbar()
    plt.show()

if __name__ == '__main__':
    main()

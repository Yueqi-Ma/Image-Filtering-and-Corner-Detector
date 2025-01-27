import matplotlib.pyplot as plt
from filters import gaussian_kernel, convolve
from common import read_img, save_img




# Load the input image
img = read_img('grace_hopper.png')

# Apply Gaussian filtering
kernel_size = 3
sigma = 0.572
kernel = gaussian_kernel(kernel_size, sigma)
filtered_img = convolve(img, kernel)




# Plot the original and filtered images side by side
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(img, cmap='gray')
axs[0].set_title('Original Image')
axs[1].imshow(filtered_img, cmap='gray')
axs[1].set_title('Gaussian Filtered Image')
plt.tight_layout()
plt.show()




# Save the filtered image
save_img(filtered_img, 'gaussian_filtered.png')
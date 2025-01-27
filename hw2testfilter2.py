import numpy as np
import matplotlib.pyplot as plt

sigma = 1.0  # Standard deviation
k = 1.6  # Scaling factor

# Create Gaussian filter
size = 9  # Filter size
x = np.linspace(-size // 2, size // 2, size)
gaussian_filter_k = np.exp(-(x ** 2) / (2 * (k * sigma) ** 2)) / ((k * sigma) * np.sqrt(2 * np.pi))

# Plot the Gaussian filter
plt.plot(x, gaussian_filter_k)
plt.title('Gaussian Filter (kσ = 1.6σ)')
plt.xlabel('Position')
plt.ylabel('Filter Value')
plt.grid(True)
plt.show()
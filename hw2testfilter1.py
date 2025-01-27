import numpy as np
import matplotlib.pyplot as plt

sigma = 1.0  # Standard deviation

# Create Gaussian filter
size = 9  # Filter size
x = np.linspace(-size // 2, size // 2, size)
gaussian_filter = np.exp(-(x ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

# Plot the Gaussian filter
plt.plot(x, gaussian_filter)
plt.title('Gaussian Filter (σ = 1.0)')
plt.xlabel('Position')
plt.ylabel('Filter Value')
plt.grid(True)
plt.show()


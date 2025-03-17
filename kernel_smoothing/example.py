import numpy as np
import matplotlib.pyplot as plt
from kernel_smoother import KernelSmoother

# Generate sample data
np.random.seed(42)
x = np.linspace(0, 2*np.pi, 100)
y_true = np.sin(x)
y_noisy = y_true + np.random.normal(0, 0.2, 100)

# Create smoothers with different kernels
kernels = ['gaussian', 'epanechnikov', 'uniform']
smoothers = [KernelSmoother(bandwidth=0.5, kernel_type=k) for k in kernels]

# Apply smoothing
y_smoothed = [smoother.smooth(x, y_noisy) for smoother in smoothers]

# Plotting
plt.figure(figsize=(12, 6))

# Plot original data
plt.scatter(x, y_noisy, color='gray', alpha=0.5, label='Noisy data')
plt.plot(x, y_true, 'k--', label='True function')

# Plot smoothed data
colors = ['red', 'blue', 'green']
for y_smooth, kernel, color in zip(y_smoothed, kernels, colors):
    plt.plot(x, y_smooth, color=color, label=f'{kernel} kernel')

plt.title('Kernel Smoothing with Different Kernels')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Print MSE for each kernel
print("\nMean Squared Error for each kernel type:")
for kernel, y_smooth in zip(kernels, y_smoothed):
    mse = np.mean((y_smooth - y_true)**2)
    print(f"{kernel.capitalize():12} : {mse:.6f}") 
import numpy as np
import pytest
import matplotlib.pyplot as plt
from kernel_smoother import KernelSmoother

def test_kernel_smoothing():
    # Test data: sine wave with noise
    x = np.linspace(0, 2*np.pi, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    # Initialize model
    smoother = KernelSmoother(bandwidth=0.5, kernel_type='gaussian')
    
    # Apply smoothing
    y_smoothed = smoother.smooth(x, y)
    
    # Check noise reduction
    error_original = np.mean((y - np.sin(x))**2)
    error_smoothed = np.mean((y_smoothed - np.sin(x))**2)
    assert error_smoothed < error_original
    
    # Check boundary conditions handling
    assert not np.isnan(y_smoothed).any()
    
def test_kernel_types():
    x = np.array([0, 1, 2])
    y = np.array([0, 1, 0])
    
    for kernel in ['gaussian', 'epanechnikov', 'uniform']:
        smoother = KernelSmoother(bandwidth=1.0, kernel_type=kernel)
        y_smooth = smoother.smooth(x, y)
        assert np.allclose(y_smooth[1], 1.0, atol=0.1)

def test_invalid_kernel():
    with pytest.raises(ValueError):
        smoother = KernelSmoother(bandwidth=1.0, kernel_type='invalid')
        x = np.array([0, 1, 2])
        y = np.array([0, 1, 0])
        smoother.smooth(x, y)

def test_bandwidth_effect():
    x = np.linspace(0, 10, 100)
    y = np.sin(x) + np.random.normal(0, 0.1, 100)
    
    smoother_narrow = KernelSmoother(bandwidth=0.1)
    smoother_wide = KernelSmoother(bandwidth=2.0)
    
    y_smooth_narrow = smoother_narrow.smooth(x, y)
    y_smooth_wide = smoother_wide.smooth(x, y)
    
    # Wide bandwidth should produce smoother result
    narrow_variation = np.var(np.diff(y_smooth_narrow))
    wide_variation = np.var(np.diff(y_smooth_wide))
    assert wide_variation < narrow_variation 

def plot_smoothing_results(x, y_true, y_noisy, y_smoothed, kernel_type):
    """
    Visualize the smoothing results
    
    Parameters:
    x: array of x values
    y_true: true function values (if known)
    y_noisy: noisy data points
    y_smoothed: smoothed values
    kernel_type: type of kernel used
    """
    plt.figure(figsize=(10, 6))
    if y_true is not None:
        plt.plot(x, y_true, 'k--', label='True function')
    plt.scatter(x, y_noisy, color='gray', alpha=0.5, label='Noisy data')
    plt.plot(x, y_smoothed, 'r-', label=f'Smoothed ({kernel_type})')
    plt.title(f'Kernel Smoothing Results ({kernel_type} kernel)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    plt.show()

def test_visualization():
    # Generate test data
    x = np.linspace(0, 2*np.pi, 100)
    y_true = np.sin(x)
    y_noisy = y_true + np.random.normal(0, 0.2, 100)
    
    # Apply smoothing
    smoother = KernelSmoother(bandwidth=0.5, kernel_type='gaussian')
    y_smoothed = smoother.smooth(x, y_noisy)
    
    # Plot results
    plot_smoothing_results(x, y_true, y_noisy, y_smoothed, 'gaussian') 
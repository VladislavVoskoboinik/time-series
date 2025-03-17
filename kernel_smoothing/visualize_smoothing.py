import numpy as np
import matplotlib.pyplot as plt
from kernel_smoother import KernelSmoother
from datetime import datetime
import os

# Create directory for graphs if it doesn't exist
GRAPHS_DIR = "time-series\kernel_smoothing\kernel_smoothing_graphs"
os.makedirs(GRAPHS_DIR, exist_ok=True)

def get_timestamp():
    """Generate timestamp string for filenames"""
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def save_plot(plt, plot_type, kernel_type):
    """Save plot to file with timestamp"""
    timestamp = get_timestamp()
    filename = f"{plot_type}_{kernel_type}_{timestamp}.png"
    filepath = os.path.join(GRAPHS_DIR, filename)
    plt.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to: {filepath}")

def plot_smoothing_results(x, y_true, y_noisy, y_smoothed, kernel_type):
    plt.figure(figsize=(10, 6))
    if y_true is not None:
        plt.plot(x, y_true, 'k--', label='True function')
    
    # Plot non-missing values in gray
    mask_valid = ~np.isnan(y_noisy)
    plt.scatter(x[mask_valid], y_noisy[mask_valid], color='gray', alpha=0.5, label='Observed data')
    
    # Plot missing values in red
    mask_missing = np.isnan(y_noisy)
    if np.any(mask_missing):
        plt.scatter(x[mask_missing], np.full_like(x[mask_missing], np.min(y_noisy[mask_valid])), 
                   color='red', marker='x', label='Missing values')
    
    plt.plot(x, y_smoothed, 'b-', label=f'Smoothed ({kernel_type})')
    plt.title(f'Kernel Smoothing with Missing Values ({kernel_type} kernel)')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.grid(True)
    save_plot(plt, "smoothing_result", kernel_type)

def plot_rmse_scores(rmse_scores, kernel_type):
    """
    Plot RMSE scores for different bandwidths
    
    Parameters:
    rmse_scores: dict with bandwidths and their corresponding RMSE scores
    kernel_type: str, type of kernel used for smoothing
    """
    bandwidths = list(rmse_scores.keys())
    scores = list(rmse_scores.values())
    
    plt.figure(figsize=(10, 6))
    plt.semilogx(bandwidths, scores, 'bo-')
    plt.xlabel('Bandwidth')
    plt.ylabel('RMSE')
    plt.title(f'RMSE Scores for Different Bandwidths ({kernel_type} kernel)')
    plt.grid(True)
    save_plot(plt, "rmse_scores", kernel_type)

if __name__ == "__main__":
    print(f"Graphs will be saved to: {os.path.abspath(GRAPHS_DIR)}")
    
    # Generate sample data
    np.random.seed(42)  # For reproducibility
    x = np.linspace(0, 2*np.pi, 100)
    y_true = np.sin(x)
    y_noisy = y_true + np.random.normal(0, 0.2, 100)
    
    # Randomly introduce missing values (20% of data)
    missing_mask = np.random.choice([True, False], size=len(x), p=[0.4, 0.6])
    y_noisy[missing_mask] = np.nan
    
    # Try different kernel types with optimal bandwidth
    kernels = ['gaussian', 'epanechnikov', 'uniform']
    
    for kernel_type in kernels:
        # Initialize smoother
        smoother = KernelSmoother(kernel_type=kernel_type)
        
        # Find optimal bandwidth using RMSE
        optimal_bandwidth, rmse_scores = smoother.find_optimal_bandwidth_rmse(x, y_noisy)
        print(f"\nOptimal bandwidth for {kernel_type} kernel: {optimal_bandwidth:.3f}")
        print(f"Best RMSE: {rmse_scores[optimal_bandwidth]:.3f}")
        
        # Plot RMSE scores
        plot_rmse_scores(rmse_scores, kernel_type)
        
        # Apply smoothing with optimal bandwidth
        y_smoothed = smoother.smooth(x, y_noisy)
        plot_smoothing_results(x, y_true, y_noisy, y_smoothed, 
                             f"{kernel_type} (bw={optimal_bandwidth:.3f})") 
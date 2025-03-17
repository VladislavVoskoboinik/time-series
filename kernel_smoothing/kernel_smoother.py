import numpy as np

class KernelSmoother:
    """
    Implementation of kernel smoothing with support for different kernel types
    
    Parameters:
    bandwidth: float - smoothing window width
    kernel_type: str - kernel type ('gaussian', 'epanechnikov', 'uniform')
    """
    
    def __init__(self, bandwidth=1.0, kernel_type='gaussian'):
        self.bandwidth = bandwidth
        self.kernel_type = kernel_type
        
    def _kernel(self, distances):
        """Calculate kernel weights"""
        if self.kernel_type == 'gaussian':
            return np.exp(-0.5 * (distances/self.bandwidth)**2)
        elif self.kernel_type == 'epanechnikov':
            return np.maximum(1 - (distances/self.bandwidth)**2, 0)
        elif self.kernel_type == 'uniform':
            return (np.abs(distances) <= self.bandwidth).astype(float)
        else:
            raise ValueError("Unsupported kernel type")

    def smooth(self, x, y, x_new=None):
        """Apply kernel smoothing"""
        x = np.asarray(x)
        y = np.asarray(y)
        x_new = x if x_new is None else np.asarray(x_new)
        
        y_smoothed = np.zeros_like(x_new)
        
        # Get mask of non-missing values
        valid_mask = ~np.isnan(y)
        
        if not np.any(valid_mask):
            return np.full_like(x_new, np.nan)
        
        # Use only non-missing values for smoothing
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        for i, xi in enumerate(x_new):
            distances = np.abs(x_valid - xi)
            weights = self._kernel(distances)
            
            # Only smooth if we have enough valid points in the neighborhood
            if np.sum(weights > 0) >= 2:  # Require at least 2 points for smoothing
                weights /= np.sum(weights)  # Normalization
                y_smoothed[i] = np.dot(weights, y_valid)
            else:
                y_smoothed[i] = np.nan
            
        return y_smoothed

    def find_optimal_bandwidth_rmse(self, x, y, bandwidths=None):
        """
        Find optimal bandwidth using RMSE
        
        Parameters:
        x: array of x values
        y: array of y values (can contain NaN)
        bandwidths: array of bandwidth values to try (default: np.logspace(-1, 1, 20))
        
        Returns:
        optimal_bandwidth: float
        rmse_scores: dict with bandwidths and their corresponding RMSE scores
        """
        if bandwidths is None:
            bandwidths = np.logspace(-1, 1, 20)
        
        # Get mask of non-missing values
        valid_mask = ~np.isnan(y)
        x_valid = x[valid_mask]
        y_valid = y[valid_mask]
        
        rmse_scores = {}
        
        # Calculate RMSE for each bandwidth
        for bw in bandwidths:
            self.bandwidth = bw
            y_pred = self.smooth(x_valid, y_valid)
            
            # Calculate RMSE only for valid predictions
            valid_pred = ~np.isnan(y_pred)
            if np.any(valid_pred):
                rmse = np.sqrt(np.mean((y_valid[valid_pred] - y_pred[valid_pred])**2))
                rmse_scores[bw] = rmse
        
        if not rmse_scores:
            raise ValueError("No valid RMSE scores obtained")
        
        # Find optimal bandwidth
        optimal_bandwidth = min(rmse_scores.items(), key=lambda x: x[1])[0]
        self.bandwidth = optimal_bandwidth
        
        return optimal_bandwidth, rmse_scores 
import numpy as np

def generate_test_series(n_points=100, trend_coef=0.1, seasonal_amp=5, noise_std=1, seasonal_period=12):
    """
    Generate synthetic time series data with trend, seasonality, and noise.
    
    Parameters:
    -----------
    n_points : int
        Number of data points to generate
    trend_coef : float
        Coefficient for linear trend
    seasonal_amp : float
        Amplitude of seasonal component
    noise_std : float
        Standard deviation of Gaussian noise
    seasonal_period : int
        Number of periods in one seasonal cycle
        
    Returns:
    --------
    tuple
        (time series data, trend component, seasonal component, noise component)
    """
    t = np.arange(n_points)
    
    # Generate components
    trend = trend_coef * t
    season = seasonal_amp * np.sin(2 * np.pi * t / seasonal_period)
    noise = np.random.normal(0, noise_std, n_points)
    
    # Combine components
    series = trend + season + noise
    
    return series, trend, season, noise

def split_data(data, train_ratio=0.8):
    """
    Split data into training and testing sets.
    
    Parameters:
    -----------
    data : array-like
        Input time series data
    train_ratio : float
        Ratio of training data (0 to 1)
        
    Returns:
    --------
    tuple
        (training data, testing data)
    """
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    test_data = data[train_size:]
    return train_data, test_data

if __name__ == "__main__":
    # Generate example series
    series, trend, season, noise = generate_test_series()
    
    # Plot components
    import matplotlib.pyplot as plt
    from datetime import datetime
    
    plt.figure(figsize=(12, 8))
    t = np.arange(len(series))
    
    plt.subplot(4, 1, 1)
    plt.plot(t, series, label='Full Series')
    plt.title('Generated Time Series Components')
    plt.legend()
    
    plt.subplot(4, 1, 2)
    plt.plot(t, trend, label='Trend', color='red')
    plt.legend()
    
    plt.subplot(4, 1, 3)
    plt.plot(t, season, label='Seasonality', color='green')
    plt.legend()
    
    plt.subplot(4, 1, 4)
    plt.plot(t, noise, label='Noise', color='purple')
    plt.legend()
    
    plt.tight_layout()
    
    # Save plot with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'generated_series_{timestamp}.png')
    plt.close() 
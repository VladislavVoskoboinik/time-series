import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from tqdm import tqdm

from generate_series import generate_test_series, split_data
from sarima_model import SARIMA

def rmse(y_true, y_pred):
    """
    Calculate Root Mean Square Error.
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
        
    Returns:
    --------
    float
        RMSE value
    """
    return np.sqrt(np.mean((np.array(y_true) - np.array(y_pred))**2))

def grid_search_parameters(data, train_ratio=0.8):
    """
    Perform grid search to find optimal SARIMA parameters.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    train_ratio : float
        Ratio of training data to use
        
    Returns:
    --------
    dict
        Dictionary containing optimal parameters and their RMSE score
    """
    # Split data for validation
    train_size = int(len(data) * train_ratio)
    train_data = data[:train_size]
    val_data = data[train_size:]
    
    # Define parameter grid
    p_params = range(0, 3)  # AR parameters
    d_params = range(0, 3)  # Differencing
    q_params = range(0, 3)  # MA parameters
    P_params = range(0, 3)  # Seasonal AR
    D_params = range(0, 3)  # Seasonal differencing
    Q_params = range(0, 3)  # Seasonal MA
    m_params = [12]        # Fixed seasonal period (monthly)
    
    # Create all possible combinations
    param_combinations = list(itertools.product(
        p_params, d_params, q_params,
        P_params, D_params, Q_params,
        m_params
    ))
    
    best_rmse = float('inf')
    best_params = None
    results = []
    
    print("Performing grid search for optimal parameters...")
    for params in tqdm(param_combinations):
        p, d, q, P, D, Q, m = params
        
        try:
            # Skip invalid combinations
            if p == 0 and q == 0 and P == 0 and Q == 0:
                continue
                
            # Create and train model
            model = SARIMA(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m)
            model.fit(train_data)
            
            # Generate predictions
            predictions = model.predict(len(val_data))
            
            # Calculate RMSE
            score = rmse(val_data, predictions)
            
            # Store results
            results.append({
                'params': params,
                'rmse': score
            })
            
            # Update best parameters if needed
            if score < best_rmse:
                best_rmse = score
                best_params = params
                
        except Exception as e:
            continue
    
    # Sort results by RMSE
    results.sort(key=lambda x: x['rmse'])
    
    # Print top 5 parameter combinations
    print("\nTop 5 parameter combinations:")
    for i, result in enumerate(results[:5]):
        p, d, q, P, D, Q, m = result['params']
        print(f"{i+1}. SARIMA({p},{d},{q})({P},{D},{Q}){m} - RMSE: {result['rmse']:.4f}")
    
    return {
        'best_params': best_params,
        'best_rmse': best_rmse,
        'all_results': results
    }

def train_and_evaluate(data, p=1, d=1, q=1, P=1, D=1, Q=1, m=12, train_ratio=0.8):
    """
    Train SARIMA model and evaluate its performance.
    
    Parameters:
    -----------
    data : array-like
        Time series data
    p, d, q : int
        Non-seasonal ARIMA parameters
    P, D, Q : int
        Seasonal ARIMA parameters
    m : int
        Seasonal period
    train_ratio : float
        Ratio of training data
        
    Returns:
    --------
    tuple
        (model, train_data, test_data, predictions, rmse_value)
    """
    # Split data
    train_data, test_data = split_data(data, train_ratio)
    
    # Create and train model
    model = SARIMA(p=p, d=d, q=q, P=P, D=D, Q=Q, m=m)
    model.fit(train_data)
    
    # Generate predictions
    predictions = model.predict(len(test_data))
    
    # Calculate RMSE
    rmse_value = rmse(test_data, predictions)
    
    return model, train_data, test_data, predictions, rmse_value

def plot_results(train_data, test_data, predictions, save=True):
    """
    Plot training data, test data, and predictions.
    
    Parameters:
    -----------
    train_data : array-like
        Training data
    test_data : array-like
        Test data
    predictions : array-like
        Model predictions
    save : bool
        Whether to save the plot
    """
    plt.figure(figsize=(12, 6))
    
    # Plot training data
    plt.plot(range(len(train_data)), train_data, 
             label='Training Data', color='blue')
    
    # Plot test data
    test_start = len(train_data)
    plt.plot(range(test_start, test_start + len(test_data)), 
             test_data, label='Test Data', color='green')
    
    # Plot predictions
    plt.plot(range(test_start, test_start + len(predictions)), 
             predictions, label='Predictions', color='red', linestyle='--')
    
    plt.title('SARIMA Model: Actual vs Predicted Values')
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.legend()
    plt.grid(True)
    
    if save:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(f'graph_sarima_{timestamp}.png')
    
    plt.close()

if __name__ == "__main__":
    # Generate example data
    n_points = 200
    series, _, _, _ = generate_test_series(
        n_points=n_points,
        trend_coef=0.2,
        seasonal_amp=10,
        noise_std=0.5,
        seasonal_period=12
    )
    
    # Find optimal parameters
    print("Finding optimal parameters...")
    grid_results = grid_search_parameters(series)
    best_p, best_d, best_q, best_P, best_D, best_Q, best_m = grid_results['best_params']
    
    print(f"\nBest parameters found:")
    print(f"SARIMA({best_p},{best_d},{best_q})({best_P},{best_D},{best_Q}){best_m}")
    print(f"Best RMSE: {grid_results['best_rmse']:.4f}")
    
    # Train and evaluate with best parameters
    model, train_data, test_data, predictions, rmse_value = train_and_evaluate(
        series,
        p=best_p,
        d=best_d,
        q=best_q,
        P=best_P,
        D=best_D,
        Q=best_Q,
        m=best_m
    )
    
    print(f"\nFinal test RMSE: {rmse_value:.4f}")
    
    # Plot results
    plot_results(train_data, test_data, predictions) 
# SARIMA Time Series Analysis

This project implements a SARIMA (Seasonal ARIMA) model from scratch for time series analysis.

## Project Structure

- `sarima_model.py`: Core SARIMA model implementation
- `generate_series.py`: Synthetic time series data generation
- `train_test.py`: Model training and evaluation utilities
- `requirements.txt`: Project dependencies

## Installation

```bash
pip install -r requirements.txt
```

## Usage

1. Generate synthetic data:
```python
from generate_series import generate_test_series
data = generate_test_series(n_points=100)
```

2. Train and test the SARIMA model:
```python
from sarima_model import SARIMA
from train_test import train_test_split, rmse

# Create and train model
model = SARIMA(p=1, d=1, q=1, P=1, D=1, Q=1, m=12)
model.fit(train_data)

# Make predictions
forecast = model.predict(len(test_data))
print(f"RMSE: {rmse(test_data, forecast)}")
```

## Parameters

- p, d, q: Non-seasonal components (AR order, differencing, MA order)
- P, D, Q: Seasonal components (seasonal AR order, seasonal differencing, seasonal MA order)
- m: Seasonal period 
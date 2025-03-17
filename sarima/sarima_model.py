import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

class SARIMA:
    """
    Seasonal ARIMA implementation from scratch.
    """
    
    def __init__(self, p=1, d=1, q=1, P=1, D=1, Q=1, m=12):
        """
        Initialize SARIMA model.
        
        Parameters:
        -----------
        p : int
            Order of AR term
        d : int
            Order of differencing
        q : int
            Order of MA term
        P : int
            Seasonal order of AR term
        D : int
            Seasonal order of differencing
        Q : int
            Seasonal order of MA term
        m : int
            Number of periods in a seasonal cycle
        """
        self.p = p
        self.d = d
        self.q = q
        self.P = P
        self.D = D
        self.Q = Q
        self.m = m
        
        # Initialize parameters
        self.ar_params = np.zeros(p)
        self.ma_params = np.zeros(q)
        self.sar_params = np.zeros(P)
        self.sma_params = np.zeros(Q)
        
        self.fitted = False
        self.residuals = None
        self.data = None
    
    def difference(self, data, k=1):
        """Apply k-th order differencing."""
        return np.diff(data, n=k)
    
    def seasonal_difference(self, data, D=1):
        """Apply seasonal differencing."""
        if D == 0:
            return data
        result = data.copy()
        for _ in range(D):
            result = result[self.m:] - result[:-self.m]
        return result
    
    def prepare_data(self, data):
        """Apply both regular and seasonal differencing."""
        data = np.array(data)
        # Regular differencing
        for _ in range(self.d):
            data = self.difference(data)
        # Seasonal differencing
        data = self.seasonal_difference(data, self.D)
        return data
    
    def _compute_residuals(self, params, data):
        """Compute residuals for given parameters."""
        n = len(data)
        residuals = np.zeros(n)
        
        # Split parameters
        ar_params = params[:self.p]
        ma_params = params[self.p:self.p+self.q]
        sar_params = params[self.p+self.q:self.p+self.q+self.P]
        sma_params = params[self.p+self.q+self.P:]
        
        # Initialize with differenced data
        residuals[:] = data[:]
        
        # Apply AR and SAR terms
        for t in range(max(self.p, self.m), n):
            if self.p > 0:
                ar_terms = np.sum(ar_params * data[t-self.p:t][::-1])
                residuals[t] -= ar_terms
            
            if self.P > 0:
                sar_terms = np.sum(sar_params * data[t-self.m*self.P:t:self.m][::-1])
                residuals[t] -= sar_terms
        
        # Apply MA and SMA terms
        for t in range(max(self.q, self.m), n):
            if self.q > 0:
                ma_terms = np.sum(ma_params * residuals[t-self.q:t][::-1])
                residuals[t] += ma_terms
            
            if self.Q > 0:
                sma_terms = np.sum(sma_params * residuals[t-self.m*self.Q:t:self.m][::-1])
                residuals[t] += sma_terms
        
        return residuals
    
    def _log_likelihood(self, params, data):
        """Compute negative log likelihood."""
        residuals = self._compute_residuals(params, data)
        sigma2 = np.var(residuals)
        return -np.sum(norm.logpdf(residuals, loc=0, scale=np.sqrt(sigma2)))
    
    def fit(self, data):
        """
        Fit SARIMA model to the data.
        
        Parameters:
        -----------
        data : array-like
            Time series data
        """
        self.data = np.array(data)
        self.diff_data = self.prepare_data(self.data)
        
        # Initial parameter guesses with small values
        initial_params = np.zeros(self.p + self.q + self.P + self.Q) * 0.1
        
        # Parameter bounds to ensure stationarity and prevent numerical issues
        bounds = []
        # AR bounds (более строгие ограничения)
        bounds.extend([(-0.95, 0.95)] * self.p)
        # MA bounds (более строгие ограничения)
        bounds.extend([(-0.95, 0.95)] * self.q)
        # Seasonal AR bounds (более строгие ограничения)
        bounds.extend([(-0.95, 0.95)] * self.P)
        # Seasonal MA bounds (более строгие ограничения)
        bounds.extend([(-0.95, 0.95)] * self.Q)
        
        # Optimize parameters with improved settings
        result = minimize(
            self._log_likelihood,
            initial_params,
            args=(self.diff_data,),
            method='L-BFGS-B',
            bounds=bounds,
            options={
                'maxiter': 1000,  # Увеличиваем максимальное число итераций
                'ftol': 1e-6,     # Увеличиваем точность
                'gtol': 1e-6      # Увеличиваем точность градиента
            }
        )
        
        # Store optimized parameters
        params = result.x
        self.ar_params = params[:self.p]
        self.ma_params = params[self.p:self.p+self.q]
        self.sar_params = params[self.p+self.q:self.p+self.q+self.P]
        self.sma_params = params[self.p+self.q+self.P:]
        
        self.residuals = self._compute_residuals(params, self.diff_data)
        self.sigma2 = np.var(self.residuals)
        self.fitted = True
        
        return self
    
    def predict(self, steps):
        """
        Generate predictions.
        
        Parameters:
        -----------
        steps : int
            Number of steps to forecast
            
        Returns:
        --------
        array-like
            Predicted values
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        # Initialize predictions
        last_values = self.data[-max(self.p, self.m):]
        predictions = np.zeros(steps)
        
        # Generate predictions one step at a time
        for t in range(steps):
            pred = 0
            
            # Add AR terms
            if self.p > 0:
                ar_terms = np.sum(self.ar_params * last_values[-self.p:][::-1])
                pred += ar_terms
            
            # Add SAR terms
            if self.P > 0:
                sar_terms = np.sum(self.sar_params * last_values[-self.m*self.P::self.m][::-1])
                pred += sar_terms
            
            # Store prediction and update last_values
            predictions[t] = pred
            last_values = np.append(last_values[1:], pred)
        
        # Reverse differencing
        result = predictions.copy()
        
        # Reverse seasonal differencing
        if self.D > 0:
            for _ in range(self.D):
                result = np.cumsum(result) + self.data[-self.m]
        
        # Reverse regular differencing
        if self.d > 0:
            for _ in range(self.d):
                result = np.cumsum(result) + self.data[-1]
        
        return result 
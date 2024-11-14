import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# Set seed for reproducibility
np.random.seed(0)

# Parameters for the AR(1) process
n = 100         # Number of time steps
phi = 0.9       # Autocorrelation coefficient (close to 1 for high autocorrelation)
sigma = 1       # Standard deviation of the noise

# Generate AR(1) process
time_series = [0]  # Start with initial value
for i in range(1, n):
    #new_value = phi * time_series[-1] + np.random.normal(0, sigma)
    new_value = np.sin(1/n*6*np.pi) + np.random.normal(0, sigma)
    time_series.append(new_value)
time_series = np.array(time_series)

# Plot the time series
plt.figure(figsize=(12, 5))
plt.plot(time_series, label="AR(1) Process with High Autocorrelation")
plt.title("AR(1) Time Series with High Autocorrelation (phi = 0.9)")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# Plot the Autocorrelation Function (ACF)
plot_acf(time_series, lags=20)
plt.title("Autocorrelation Function (ACF) of AR(1) Process with High Autocorrelation")
plt.xlabel("Lags")
plt.ylabel("Autocorrelation")
plt.show()

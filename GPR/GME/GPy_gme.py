import GPy
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel, WhiteKernel, StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter
from scipy.spatial.distance import cdist, pdist, squareform

start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2021-01-01", utc=True)

def noise_function(X):
    size = len(X)
    arr = np.ones(size)*0.1
    # Define peak parameters
    peak_value = 15      # The peak value at the centers
    center1, center2 = size, 60  # Peak centers (peak at the end)
    width = 3           # Width of the peaks (standard deviation of the Gaussian)

    # Add Gaussian peaks centered at 30 and 60
    arr += (peak_value - 1) * np.exp(-((np.arange(size) - center1) ** 2) / (2 * width ** 2))
    return arr

def process_range(start_date, end_date):
    data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()
    data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime
    data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]

    date_range = data_history['Date']

    X_ordinal = data_history['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    Y = data_history['Open'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ordinal)

    # Generate synthetic data with heteroscedastic noise
    #X = np.linspace(0, 10, 100)[:, None]
    #Y = np.sin(X) + np.random.randn(100, 1) * (0.1 + 0.5 * (X / 10)**2)

    # Create an initial noise variance array
    initial_noise_var = noise_function(X_scaled).reshape(-1, 1)  # Heteroscedastic noise

    # Create the heteroscedastic GP model with the RBF kernel
    kernel = GPy.kern.RBF(input_dim=1)
    model = GPy.models.GPHeteroscedasticRegression(X_scaled, Y, kernel=kernel)

    # Set the initial noise variance in the model's likelihood
    model['.*het_Gauss.variance'] = initial_noise_var

    # Optimize the model
    model.optimize()
    model.plot()
    plt.show()  # Ensure the plot is displayed

process_range(start_date, end_date)
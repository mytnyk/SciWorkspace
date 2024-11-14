import numpy as np
import pandas as pd
import heapq
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel, WhiteKernel, ExpSineSquared, StationaryKernelMixin, NormalizedKernelMixin, Hyperparameter
from scipy.spatial.distance import cdist, pdist, squareform
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf
from rbf_kernels import RBF_WITH_CALLBACK

#good example, big period
start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2022-02-01", utc=True)
num_forecast = 70

def process_range(start_date, end_date):

    data_history = pd.read_csv('_data_/gme_stock_history.csv')
    data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime
    data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]

    date_range = data_history['Date']

    date_to_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(date_range)}

    X_ordinal = data_history['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    Y = data_history['Open'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ordinal)


    #-------------------------

    # Now warping input space:

    yrange = max(Y)-min(Y)
    X_warped = np.copy(X_scaled)
    for i in range(1,len(X_scaled)):
        xn = X_scaled[i]#probably should be taken into account
        xp = X_scaled[i-1]# so far we can go with it because all input is dense and full
        yn = Y[i]
        yp = Y[i-1]
        d = abs(yn-yp)/yrange
        X_warped[i] = X_warped[i-1]+d

    #------------------------------------

    y = Y.flatten()

    number_of_training_samples = len(X_ordinal) - num_forecast
    training_indices = range(number_of_training_samples)
    X_train, y_train = X_warped[training_indices], y[training_indices]

    kernel = C(100.0, constant_value_bounds="fixed") * RBF(length_scale=0.5, length_scale_bounds=(0.01,10))*ExpSineSquared(1, 0.5*1.43, periodicity_bounds="fixed", length_scale_bounds="fixed") + WhiteKernel(0.918, noise_level_bounds=(0.01,100))
#*ExpSineSquared(1, 0.5*1.43, periodicity_bounds="fixed", length_scale_bounds="fixed") + WhiteKernel(0.918, noise_level_bounds="fixed") #, length_scale_bounds="fixed"(0.001,100)
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)

    gp.fit(X_train, y_train)
    # Display the hyperparameters and their order
    print("Theta (log-space hyperparameters):", gp.kernel_.theta)
    print("Hyperparameter names:", gp.kernel_.hyperparameters)

    # Define hyperparameters to evaluate (log space for stability)
    length_scales = np.linspace(0.1, 1, 10)
    noise_values = np.linspace(0.91, 0.93, 6)
    log_marg_lik_values = np.zeros((len(length_scales), len(noise_values)))
          
    # Calculate log marginal likelihood for each combination
    for i, length_scale in enumerate(length_scales):
        for j, noise_value in enumerate(noise_values):
            theta = np.log([length_scale, noise_value])  # Convert to log-space
            log_marg_lik_values[i, j] = gp.log_marginal_likelihood(theta)
            print(f"l={length_scale} n={noise_value} ML={log_marg_lik_values[i, j]}")

    #for i, length_scale in enumerate(length_scales):
    #        for j, noise_value in enumerate(noise_values):
    #             log_marg_lik_values[j, i] = i + j
    
    # Plot contours of log marginal likelihood
    X, Y = np.meshgrid(length_scales, noise_values, indexing='ij')
    #log_marg_lik_values = np.sin(X) * np.cos(Y)

    plt.contourf(X, Y, log_marg_lik_values, levels=20, cmap="viridis")
    plt.colorbar(label="Log Marginal Likelihood")
    #plt.xscale("log")
    #plt.yscale("log")
    plt.xlabel("Length Scale")
    plt.ylabel("Noise")
    plt.title("Contours of Log Marginal Likelihood")
    plt.show()


process_range(start_date, end_date)
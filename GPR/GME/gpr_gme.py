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

#start_date = pd.to_datetime("2020-09-01", utc=True)
#end_date = pd.to_datetime("2021-03-01", utc=True)

# good example with 0 predictions
#start_date = pd.to_datetime("2020-12-01", utc=True)
#end_date = pd.to_datetime("2021-01-15", utc=True)
#threshold_date = pd.to_datetime("2021-01-15", utc=True)# REAL CHANGES!!
#num_forecast = 0

# good example of bad predictions, but still valid confidence
#start_date = pd.to_datetime("2020-10-01", utc=True)
#end_date = pd.to_datetime("2020-12-01", utc=True)
#num_forecast = 10

#nice example
#start_date = pd.to_datetime("2020-10-10", utc=True)
#end_date = pd.to_datetime("2021-01-01", utc=True)
#num_forecast = 10

#good example, big period
start_date = pd.to_datetime("2020-10-01", utc=True)
end_date = pd.to_datetime("2022-02-01", utc=True)
num_forecast = 70

#good prediction
#start_date = pd.to_datetime("2021-10-01", utc=True)
#end_date = pd.to_datetime("2022-02-01", utc=True)
#num_forecast = 10

def process_range(start_date, end_date):

    data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()
    data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime
    data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]

    date_range = data_history['Date']

    date_to_index = {date.strftime("%Y-%m-%d"): index for index, date in enumerate(date_range)}

    X_ordinal = data_history['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)
    Y = data_history['Open'].values.reshape(-1, 1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_ordinal)

    #-------------------------
    # stationarity test:
    adf_result = adfuller(Y)
    print(f"ADF Statistic: {adf_result[0]}, p-value: {adf_result[1]}")
    # KPSS Test
    def kpss_test(series, **kw):
        statistic, p_value, _, _ = kpss(series, **kw)
        return statistic, p_value

    kpss_stat, kpss_p_value = kpss_test(Y, regression='c')
    print(f"KPSS Statistic: {kpss_stat}, p-value: {kpss_p_value}")

#    plot_acf(Y)  # lags=20 shows ACF up to 20 lags
#    plt.title("Autocorrelation Function (ACF)")
#    plt.xlabel("Lags")
#    plt.ylabel("Autocorrelation")
#    plt.show()
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

    kernel = RBF(length_scale=0.339)

    #fig, ax = plt.subplots(1, 2, figsize=(12, 6))

    # Original space
    #K = kernel(X_scaled)
    #im1 = ax[0].imshow(K, cmap='viridis', interpolation='nearest')
    #fig.colorbar(im1, ax=ax[0], shrink=0.7)#label='Коваріація', 
    #ax[0].set_title("Оригінальний простір")
    #ax[0].set_aspect('equal')

    # Warped space
    #K = kernel(X_warped)
    #im2 = ax[1].imshow(K, cmap='viridis', interpolation='nearest')
    #fig.colorbar(im2, ax=ax[1], shrink=0.7)#label='Коваріація', 
    #ax[1].set_title("Викривлений простір")
    #ax[1].set_aspect('equal')
    #plt.show()

    #X_warped = X_scaled

    #-----------------------------------------
    # Eigenvalue decomposition
    #eigen_values, eigen_vectors = np.linalg.eig(K)
    # Remove small imaginary parts from eigenvectors if any
    #eigen_values = np.real(eigen_values)
    #eigen_vectors = np.real(eigen_vectors)
    #scaled_eigenvectors = eigen_vectors * eigen_values
    #sns.heatmap(scaled_eigenvectors[:,:30], cmap="coolwarm", center=0)#cmap="viridis"
    #plt.title("Heatmap of High-Dimensional Eigenvectors (Magnitude)")
    #plt.xlabel("Eigenvector Index")
    #plt.ylabel("Dimension")
    #plt.show()
    #------------------------------------

    #y_samples = np.random.multivariate_normal(mean=np.zeros(len(X_ordinal)), cov=K, size=4).T
    #y_std = np.sqrt(np.diag(K))
    #confidence_interval = 1.96 * y_std

    # Plot the samples
    #plt.figure()
    #for i, sample in enumerate(y_samples.T):
    #    plt.plot(date_range, sample, label=f'Sample {i+1}')
    # Plot the 95% confidence interval
    #plt.fill_between(date_range, -confidence_interval, confidence_interval,
    #                alpha=0.2, color='gray', label='95% confidence interval')
    #plt.title('Samples from the GP Prior')
    #plt.xlabel('Input')
    #plt.ylabel('Output')
    #plt.legend()
    #plt.show()

    y = Y.flatten()

    number_of_training_samples = len(X_ordinal) - num_forecast
    training_indices = range(number_of_training_samples)
    X_train, y_train = X_warped[training_indices], y[training_indices]

# calculate distance between peaks:
    start_peak_date = "2021-09-01"
    end_peak_date = "2021-11-01"
    start_peak_date_index = date_to_index[start_peak_date]
    end_peak_date_index = date_to_index[end_peak_date]
    dist = X_warped[end_peak_date_index] - X_warped[start_peak_date_index]

    called = False
    lengthscales = []
    evidences = []
    gp = GaussianProcessRegressor()
    def print_callback(lengthscale):
        nonlocal called
        if called:
            return
        called = True
        evidence = -gp.log_marginal_likelihood(gp.kernel_.theta)
        nonlocal lengthscales, evidences
        lengthscales.append(lengthscale)
        evidences.append(evidence)
        print(f"Length-scale:{lengthscale}, evidence={evidence}",)
        called = False

#10**2 * RBF(length_scale=2.29) * ExpSineSquared(length_scale=0.482, periodicity=2.63) + WhiteKernel(noise_level=2.72)
#(0.5, 100)
    kernel = C(100.0, constant_value_bounds="fixed") * RBF_WITH_CALLBACK(print_callback,length_scale=0.5, length_scale_bounds=(0.01,1))*ExpSineSquared(1, 0.5*1.43, periodicity_bounds="fixed", length_scale_bounds="fixed") + WhiteKernel(0.918, noise_level_bounds="fixed") #, length_scale_bounds="fixed"(0.001,100)
#
    gaussian_process = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
    gp = gaussian_process
    gaussian_process.fit(X_train, y_train)
    print(f"score={gaussian_process.score(X_warped,y)}") # may be on test data only?
    print(gaussian_process.kernel_)

    mean_prediction, y_cov  = gaussian_process.predict(X_warped, return_cov=True)

    rmse = np.sqrt(mean_squared_error(y, mean_prediction))

    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

    # Plot the points
    plt.plot(lengthscales, evidences, 'o', label="Data points")  # 'o' makes it a scatter plot
    plt.xlabel("x values")
    plt.ylabel("y values")
    plt.title("Plot of dictionary points (x, y)")
    plt.legend()
    plt.show()

    #plt.figure(figsize=(8, 6))
    #plt.imshow(y_cov, cmap='hot', interpolation='nearest')
    #plt.title('Covariance Matrix Heatmap')
    #plt.colorbar(label='Covariance value')
    #plt.xlabel('Input points')
    #plt.ylabel('Input points')
    #plt.show()

    y_std = np.sqrt(np.diag(y_cov))
    confidence_interval = 1.96 * y_std
    plt.ylim(bottom=0, top=max(y)+1)

    plt.scatter(date_range[:number_of_training_samples], y_train, label="Тренування", marker=".", c="k")
    plt.scatter(date_range[number_of_training_samples:], y[number_of_training_samples:], label="Тест", marker="+", c="g")
    plt.plot(date_range, mean_prediction, label="Прогноз", c="b")
    plt.fill_between(
        date_range,
        mean_prediction - confidence_interval,
        mean_prediction + confidence_interval,
        alpha=0.2,
        label=r"95% інтервал довіри",
    )
    plt.legend()
    #plt.xlabel("$x$")
    plt.ylabel("Ціна, $", rotation=0, labelpad=15)
    #_ = plt.title("Gaussian process regression on noise-free dataset")
    plt.show()

process_range(start_date, end_date)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C, Kernel

class NonStationaryKernel(Kernel):
    def __init__(self, length_scale=1.0, transition_point=0.0, transition_width=1.0):
        # Use the normal __setattr__ to set the attributes directly
        super().__setattr__('length_scale', length_scale)
        super().__setattr__('base_kernel', RBF(length_scale=length_scale, length_scale_bounds=(0.1, 1e2)))
        self.transition_point = transition_point
        self.transition_width = transition_width

    def __call__(self, X, Y=None, eval_gradient=False):
        
        # Create a Gaussian bump function centered at `transition_point`
        
        
        bump_X = np.exp(-0.5 * ((X - self.transition_point) ** 2) / self.transition_width ** 2)
        if Y is None:
            bump_Y = bump_X
        else:
            bump_Y = np.exp(-0.5 * ((Y - self.transition_point) ** 2) / self.transition_width ** 2)
        
        # Modify the covariance by reducing it in the specified region
        scaling_matrix = 1 - 0 * np.outer(bump_X, bump_Y)
        scaling_matrix[scaling_matrix<0]=0

        if eval_gradient:
            K, G = self.base_kernel(X, Y, eval_gradient)
            K = K * scaling_matrix
            return K, G
        K = self.base_kernel(X, Y, eval_gradient)
        K = K * scaling_matrix
        return K

    def diag(self, X):
        # Return diagonal elements
        return self.base_kernel.diag(X)

    def is_stationary(self):
        return self.base_kernel.is_stationary()

    @property
    def hyperparameter_length_scale(self):
        """Expose the length_scale for optimization"""
        return self.base_kernel.hyperparameter_length_scale

    @property
    def theta(self):
        """Return the log of the hyperparameters for optimization"""
        return self.base_kernel.theta

    @theta.setter
    def theta(self, theta):
        """Set the kernel's hyperparameters (inverse of the log)"""
        self.base_kernel.theta = theta
        self.length_scale = np.exp(theta)  # Update length_scale based on theta

    @property
    def bounds(self):
        """Return the bounds for the hyperparameters"""
        return np.array([self.base_kernel.length_scale_bounds])#self.base_kernel.bounds

    def __setattr__(self, name, value):
        # Special case to handle length_scale
        if name == 'length_scale':
            # Update the length_scale in the base kernel as well
            self.base_kernel.length_scale = value
        # Set the attribute normally
        super().__setattr__(name, value)


data_history = pd.read_csv('_data_/gme_stock_history.csv')#.to_numpy()

data_history['Date'] = pd.to_datetime(data_history['Date'], utc=True)  # Convert 'Date' to datetime

# Step 2: Define the date range
start_date = pd.to_datetime("2020-01-01", utc=True)
end_date = pd.to_datetime("2023-01-01", utc=True)

# Step 3: Filter the DataFrame for rows within the date range
data_history = data_history[(data_history['Date'] >= start_date) & (data_history['Date'] <= end_date)]


X = data_history['Date'].map(pd.Timestamp.toordinal).values.reshape(-1, 1)  # Convert to ordinal
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.transform(X)

Y = data_history['Open'].values.reshape(-1, 1) 
scaler = preprocessing.StandardScaler().fit(Y)
Y = scaler.transform(Y).flatten()

plt.figure()
plt.plot(X, Y, 'b-', label='GME')
# Define the kernel
# constant - 1.0 - vertical scaling
# RBF - 1.0 - horizontal scaling (how quickly function changes within input)
rbf_kernel = RBF(0.5, (0.01, 1e2))

# Create the non-stationary kernel with a transition point at X=1.0 and a length scale factor
Nonstationary = NonStationaryKernel(length_scale=0.5, transition_point=-0.5, transition_width=0.1)

kernel = C(1.0, (1e-3, 1e3)) * rbf_kernel#Nonstationary

# Instantiate a Gaussian Process model
gp = GaussianProcessRegressor(kernel=kernel)

# Training data
X_train = X[:100]#np.array([[1], [3], [5], [6], [7], [8]]).reshape(-1, 1)
y_train = Y[:100]#np.array([3, 2, 4, 6, 7, 8])

# Generate sample points
#X = np.linspace(0, 10, 50).reshape(-1, 1)
#print(np.round(X,2))
# Compute the covariance matrix
K = kernel(X_train)
#print(np.round(K,2))
# Plot the covariance matrix
plt.figure(figsize=(8, 6))
plt.imshow(K, cmap='hot', interpolation='nearest')
plt.title('Covariance Matrix Heatmap')
plt.colorbar(label='Covariance value')
plt.xlabel('Input points')
plt.ylabel('Input points')
plt.show()

# Step 3: Compute the standard deviation (square root of the diagonal elements)
y_std = np.sqrt(np.diag(K))
# Calculate the 95% confidence interval
confidence_interval = 1.96 * y_std



# Instantiate a Gaussian Process model
noise_std = 0#alpha=noise_std**2,
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit to the training data
gp.fit(X_train, y_train)

# Print the optimized RBF kernel
print("Built-in RBF kernel after fitting:", gp.kernel_)
#Built-in RBF kernel after fitting: 4.88**2 * RBF(length_scale=1.98)
#Built-in RBF kernel after fitting: CustomRBFKernel(0.309)

# Test points
X_test = X[:105]#np.linspace(0, 10, 100).reshape(-1, 1)

# generate samples from aposterior
#y_samples = gp.sample_y(X_test, n_samples=30)
#plt.figure()
#plt.plot(X_train, y_train, 'r.', markersize=10, label='Training data')
#for i, sample in enumerate(y_samples.T):
#    plt.plot(X_test, sample, label=f'Sample {i+1}')
#plt.show()

y_pred, sigma = gp.predict(X_test, return_std=True)

plt.figure()
#plt.errorbar(
#    X,
#    Y,
#    noise_std,
#    linestyle="None",
#    color="tab:red",
#    marker=".",
#    markersize=10,
#    label="Observations",
#)
plt.plot(X_train, y_train, 'r.', markersize=10, label='Training data')
plt.plot(X_test, y_pred, 'b-', label='Prediction')
#plt.fill_between(X_test.ravel(), y_pred - 1.96*sigma, y_pred + 1.96*sigma,
#                 alpha=0.2, color='k', label='95% confidence interval')
plt.xlabel('Input')
plt.ylabel('Output')
plt.title('Gaussian Process Regression')
plt.legend()
plt.show()

exit()
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

# Generate synthetic data
np.random.seed(42)
x = np.linspace(0, 10, 20)
true_m = 2
true_b = 5
sigma_y = 2  # Standard deviation of the noise
y = true_m * x + true_b + np.random.normal(0, sigma_y, size=x.shape)

# Define the prior distribution: Gaussian with mean (0, 0) and a large covariance
mean_prior = [0, 0]
cov_prior = [[2, 0], [0, 2]]  # Large variance

# Grid of slope (m) and intercept (b) values
m_values = np.linspace(0, 4, 100)
b_values = np.linspace(0, 10, 100)
M, B = np.meshgrid(m_values, b_values)

# Calculate the prior density
prior_density = multivariate_normal(mean_prior, cov_prior).pdf(np.dstack((M, B)))

# Calculate the likelihood
def likelihood(m, b, x, y):
    y_pred = m * x + b
    return np.exp(-0.5 * np.sum((y - y_pred)**2))

likelihood_values = np.array([[likelihood(m, b, x, y) for m in m_values] for b in b_values])

# Calculate the posterior density (unnormalized)
posterior_density = prior_density * likelihood_values

# Normalize the posterior
posterior_density /= np.sum(posterior_density)

# Calculate the predictive mean and confidence interval
m_mean = np.sum(M * posterior_density)
b_mean = np.sum(B * posterior_density)
y_pred_mean = m_mean * x + b_mean

# Calculate the predictive variance (assuming independent Gaussian noise)

y_pred_var = np.array([np.sum(posterior_density * ((M * xi + B) - (m_mean * xi + b_mean))**2) for xi in x]) + sigma_y**2
y_pred_std = np.sqrt(y_pred_var)

# Plot the contours
plt.figure(figsize=(18, 6))

# Prior
plt.subplot(1, 3, 1)
plt.contour(M, B, prior_density, levels=5, cmap='Blues')
plt.title('Prior Distribution Contours')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')

# Likelihood
plt.subplot(1, 3, 2)
plt.contour(M, B, likelihood_values, levels=5, cmap='Reds')
plt.title('Likelihood Contours')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')

# Posterior
plt.subplot(1, 3, 3)
plt.contour(M, B, posterior_density, levels=5, cmap='Greens')
plt.title('Posterior Distribution Contours')
plt.xlabel('Slope (m)')
plt.ylabel('Intercept (b)')

plt.tight_layout()
plt.show()

# Plot the predictive mean and confidence intervals
plt.figure(figsize=(10, 6))
plt.plot(x, y, 'o', label='Observed Data')
plt.plot(x, y_pred_mean, 'r-', label='Predictive Mean')
plt.fill_between(x, y_pred_mean - 2*y_pred_std, y_pred_mean + 2*y_pred_std, color='r', alpha=0.2, label='Predictive Mean ± 2 Std. Dev.')
plt.title('Predictive Mean with Confidence Interval')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

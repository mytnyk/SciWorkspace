import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Generate synthetic data
np.random.seed(4)
x = np.linspace(0, 1, 20)
true_m = 0.4
sigma_y = 0.05  # Standard deviation of the noise
y = true_m * x + np.random.normal(0, sigma_y, size=x.shape)

# Define the prior distribution for the slope m
mean_prior = 0
std_prior = 0.3  # Large standard deviation indicating prior uncertainty
m_values = np.linspace(-0.5, 0.5, 100)

# Calculate the prior density
prior_density = norm.pdf(m_values, mean_prior, std_prior)

# Calculate the likelihood
def likelihood(m, x, y):
    y_pred = m * x
    return np.exp(-0.5 * np.sum((y - y_pred)**2))

likelihood_values = np.array([likelihood(m, x, y) for m in m_values])

# Calculate the posterior density (unnormalized)
posterior_density = prior_density * likelihood_values

# Normalize the posterior
posterior_density /= np.sum(posterior_density)

# Calculate the predictive mean and confidence interval
m_mean = np.sum(m_values * posterior_density)
y_pred_mean = m_mean * x

# Calculate the predictive variance (assuming independent Gaussian noise)

y_pred_var = np.array([np.sum(posterior_density * ((m_values * xi) - (m_mean * xi))**2) for xi in x]) + sigma_y**2
y_pred_std = np.sqrt(y_pred_var)

# Plot the distributions
plt.figure(figsize=(15, 5))

# Prior
plt.subplot(1, 3, 1)
plt.plot(m_values, prior_density, color='blue')
plt.title('Prior Distribution')
plt.xlabel('Slope (m)')
plt.ylabel('Density')

# Likelihood
plt.subplot(1, 3, 2)
plt.plot(m_values, likelihood_values, color='red')
plt.title('Likelihood')
plt.xlabel('Slope (m)')
plt.ylabel('Likelihood')

# Posterior
plt.subplot(1, 3, 3)
plt.plot(m_values, posterior_density, color='green')
plt.title('Posterior Distribution')
plt.xlabel('Slope (m)')
plt.ylabel('Density')

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

import numpy as np
import matplotlib.pyplot as plt

def multivariate_normal(mean, cov, size):
    # Step 1: Generate standard normal variables
    Z = np.random.randn(size, len(mean))
    
    # Step 2: Perform Cholesky decomposition of the covariance matrix
    L = np.linalg.cholesky(cov)
    
    # Step 3: Transform the standard normal variables
    samples = Z @ L.T + mean
    
    return samples

# Define mean vector and covariance matrix
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])

# Generate samples
size = 1000
samples = multivariate_normal(mean, cov, size)

# Plot the samples
plt.figure(figsize=(8, 6))
plt.scatter(samples[:, 0], samples[:, 1], s=2)
plt.title('Samples from a Multivariate Normal Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.axis('equal')
plt.show()

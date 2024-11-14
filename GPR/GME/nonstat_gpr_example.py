import numpy as np
from sklearn.gaussian_process.kernels import RBF, Kernel

class NonStationaryKernel(Kernel):
    """
    A kernel that introduces non-stationarity by modulating a stationary kernel
    (such as RBF) with a Gaussian bump function around a specific point.
    """
    
    def __init__(self, base_kernel, transition_point=0.0, transition_width=1.0):
        """
        Args:
            base_kernel: The stationary kernel (e.g., RBF).
            transition_point: The point where the non-stationarity is centered.
            transition_width: The width of the Gaussian bump (controls spread of non-stationary region).
        """
        self.base_kernel = base_kernel
        self.transition_point = transition_point
        self.transition_width = transition_width
    
    def __call__(self, X, Y=None, eval_gradient=False):
        # Compute the base kernel's covariance matrix
        K_base = self.base_kernel(X, Y)
        
        # Create a Gaussian bump function centered at `transition_point`
        if Y is None:
            Y = X
        
        bump_X = np.exp(-0.5 * ((X - self.transition_point) ** 2) / self.transition_width ** 2)
        bump_Y = np.exp(-0.5 * ((Y - self.transition_point) ** 2) / self.transition_width ** 2)
        
        # Apply the bump function element-wise to modify the covariance
        K_mod = K_base * np.outer(bump_X, bump_Y)
        
        if eval_gradient:
            # Return the modified kernel matrix without gradients for simplicity
            return K_mod, np.zeros_like(K_mod)[:, :, np.newaxis]
        
        return K_mod
    
    def diag(self, X):
        return np.diag(self(X))
    
    def is_stationary(self):
        return False  # This kernel is non-stationary

# Example Usage:

# Create a base RBF kernel
base_kernel = RBF(length_scale=1.0)

# Create a non-stationary kernel with non-stationarity around x = 1.5
Kernel = NonStationaryKernel(base_kernel, transition_point=1.5, transition_width=0.5)

# Generate some sample data
X = np.linspace(0, 3, 100).reshape(-1, 1)

# Compute the kernel matrix
K = Kernel(X)
print(K)

# Plot the kernel matrix to visualize non-stationarity
import matplotlib.pyplot as plt

plt.imshow(K, extent=(0, 3, 0, 3), origin='lower')
plt.colorbar(label='Covariance')
plt.title('Non-Stationary Kernel Matrix')
plt.show()

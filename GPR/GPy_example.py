import GPy
import numpy as np
import matplotlib.pyplot as plt

# Generate synthetic data with heteroscedastic noise
X = np.linspace(0, 10, 100)[:, None]
Y = np.sin(X) + np.random.randn(100, 1) * (0.1 + 0.5 * (X / 10)**2)

# Create an initial noise variance array
initial_noise_var = 0.1 + 0.5 * (X / 10)**2  # Heteroscedastic noise

# Create the heteroscedastic GP model with the RBF kernel
kernel = GPy.kern.RBF(input_dim=1)
model = GPy.models.GPHeteroscedasticRegression(X, Y, kernel=kernel)

# Set the initial noise variance in the model's likelihood
model['.*het_Gauss.variance'] = initial_noise_var

# Optimize the model
model.optimize()
model.plot()
plt.show()  # Ensure the plot is displayed
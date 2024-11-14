import numpy as np
import matplotlib.pyplot as plt

# Step 1: Create a grid of points in the input space
x = np.linspace(-2, 2, 20)
y = np.linspace(-2, 2, 20)
X, Y = np.meshgrid(x, y)

# Step 2: Define a warping transformation function (for example, bending the space)
# This will warp the space by a non-linear transformation
def warp_function(x, y):
    # Apply a non-linear transformation to warp the space
    X_new = x + 0.3 * np.sin(2 * np.pi * y)  # Horizontal bending
    Y_new = y + 0.3 * np.sin(2 * np.pi * x)  # Vertical bending
    return X_new, Y_new

# Step 3: Apply the warping transformation to each point in the grid
X_warped, Y_warped = warp_function(X, Y)

# Step 4: Plot the original and warped grids
fig, ax = plt.subplots(1, 2, figsize=(12, 6))

# Original space
ax[0].plot(X, Y, 'k-', lw=0.5)  # Plot horizontal lines
ax[0].plot(X.T, Y.T, 'k-', lw=0.5)  # Plot vertical lines
ax[0].set_title("Original Space")
ax[0].set_aspect('equal')

# Warped space
ax[1].plot(X_warped, Y_warped, 'k-', lw=0.5)  # Plot warped horizontal lines
ax[1].plot(X_warped.T, Y_warped.T, 'k-', lw=0.5)  # Plot warped vertical lines
ax[1].set_title("Warped Space")
ax[1].set_aspect('equal')

plt.show()

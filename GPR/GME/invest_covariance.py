import numpy as np
import matplotlib.pyplot as plt
import heapq
import seaborn as sns

from rbf_kernels import RBF_MODIFIED


import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA





# Array size
size = 100

# Initialize the array with ones
arr = np.ones(size)

# Define peak parameters
peak_value = 10      # The peak value at the centers
center1, center2 = 30, 60  # Peak centers
width = 1           # Width of the peaks (standard deviation of the Gaussian)

# Add Gaussian peaks centered at 30 and 60
arr += (peak_value - 1) * np.exp(-((np.arange(size) - center1) ** 2) / (2 * width ** 2))
arr += (peak_value - 1) * np.exp(-((np.arange(size) - center2) ** 2) / (2 * width ** 2))

# Plot the result to visualize
plt.plot(arr)
plt.title("Array with Smooth Peaks at 30 and 60")
plt.xlabel("Index")
plt.ylabel("Value")
plt.show()

X = np.linspace(start=0, stop=10, num=size).reshape(-1, 1)
kernel = 1 * RBF_MODIFIED(lambda x: arr, length_scale=1)
K = kernel(X)
plt.figure(figsize=(8, 6))
plt.imshow(K, cmap='hot', interpolation='nearest')
plt.title('Covariance Matrix Heatmap')
plt.colorbar(label='Covariance value')
plt.xlabel('Input points')
plt.ylabel('Input points')
plt.show()



# Знаходимо власні числа і власні вектори
eigen_values, eigen_vectors = np.linalg.eig(K)
# Remove small imaginary parts from eigenvectors if any
eigen_values = np.real(eigen_values)
eigen_vectors = np.real(eigen_vectors)
scaled_eigenvectors = eigen_vectors * eigen_values


# Apply PCA to reduce to 2 dimensions
#pca = PCA(n_components=2)
#eigenvectors_2d = pca.fit_transform(eigen_vectors.T)  # Transpose for correct shape
# Plot the eigenvectors in 2D space
#plt.scatter(eigenvectors_2d[:, 0], eigenvectors_2d[:, 1])
#plt.title("2D Projection of High-Dimensional Eigenvectors")
#plt.xlabel("Principal Component 1")
#plt.ylabel("Principal Component 2")
#plt.show()


#eigenvectors_magnitude = np.abs(eigen_vectors)
# Plot heatmap of the magnitudes of the eigenvectors
sns.heatmap(scaled_eigenvectors[:,:30], cmap="coolwarm", center=0)#cmap="viridis"
plt.title("Heatmap of High-Dimensional Eigenvectors (Magnitude)")
plt.xlabel("Eigenvector Index")
plt.ylabel("Dimension")
plt.show()



# Виведемо власні значення і власні вектори
top_number = 3
top_indices = [i for _, i in heapq.nlargest(top_number, [(value, i) for i, value in enumerate(eigen_values)])]
largest_values_of_eigenvalues = eigen_values[top_indices]
print(f"Eigenvalues ({top_number} largest):", [f"{num:.2f}" for num in largest_values_of_eigenvalues])
for i in top_indices:
    eigen_vector = np.real_if_close(eigen_vectors[i])
    norm = np.linalg.norm(eigen_vector)
    #largest_values_of_eigenvector = heapq.nlargest(10, eigen_vector)
    print(f"Eigenvector {i}: norm={norm}", [f"{num:.2f}" for num in eigen_vector])
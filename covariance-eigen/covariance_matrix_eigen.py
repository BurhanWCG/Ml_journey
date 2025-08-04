import numpy as np

# Step 1=> Dataset with 8 samples and 5 features
data = np.array([
    [2.5, 2.4, 3.1, 4.0, 5.2],
    [0.5, 0.7, 1.0, 1.2, 0.8],
    [2.2, 2.9, 3.0, 3.5, 3.7],
    [1.9, 2.2, 2.7, 3.0, 3.3],
    [3.1, 3.0, 3.8, 4.2, 5.0],
    [2.3, 2.7, 3.0, 3.6, 3.9],
    [2.0, 1.6, 2.5, 2.8, 2.7],
    [1.0, 1.1, 1.2, 1.5, 1.7]
])

# Step 2=> Mean subtraction (mean centering)
mean = np.mean(data, axis=0)
centered_data = data - mean

# Step 3=> Covariance matrix (feature-wise)
cov_matrix = np.cov(centered_data, rowvar=False)
print("Covariance matrix:\n", cov_matrix)

# Step 4=> Eigenvalues and Eigenvectors
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

print("\nEigenvalues:\n", eigenvalues)
print("\nEigenvectors:\n", eigenvectors)

# Step 5=> Find the largest eigenvalue and its corresponding eigenvector
max_index = np.argmax(eigenvalues)
print(f"\nLargest Eigenvalue: {eigenvalues[max_index]}")
print("Eigenvector corresponding to the largest eigenvalue:\n", eigenvectors[:, max_index])

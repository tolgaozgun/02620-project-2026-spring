import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # SVD: X_centered = U S V^T
        # For PCA, V are the principal components
        # Using full_matrices=False for efficiency
        # We use SVD on the centered data matrix directly which is more stable than X^T X
        # X / sqrt(N-1) makes S^2 the eigenvalues of the covariance matrix
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # Store components
        self.components = Vt[:self.n_components]
        
        # Explained variance
        self.explained_variance_ = (S**2) / (X.shape[0] - 1)
        self.explained_variance_ratio_ = self.explained_variance_ / np.sum( (S**2) / (X.shape[0] - 1) )
        
        return self

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

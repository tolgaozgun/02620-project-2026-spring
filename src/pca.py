import numpy as np

class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.total_explained_variance_ratio_ = None

    def fit(self, X):
        # Center the data
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # SVD: X_centered = U S V^T
        # V rows are the principal component directions
        # Using full_matrices=False for efficiency
        U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)

        # Compute variance for ALL components (needed for proper ratio normalization)
        all_explained_variance = (S ** 2) / (X.shape[0] - 1)
        total_variance = np.sum(all_explained_variance)

        # Keep only top n_components
        self.components = Vt[:self.n_components]
        self.explained_variance_ = all_explained_variance[:self.n_components]
        self.explained_variance_ratio_ = self.explained_variance_ / total_variance
        self.total_explained_variance_ratio_ = np.sum(self.explained_variance_ratio_)

        return self

    def transform(self, X):
        X_centered = X - self.mean
        return np.dot(X_centered, self.components.T)

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

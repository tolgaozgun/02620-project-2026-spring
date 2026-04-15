import numpy as np

class KMeans:
    def __init__(self, n_clusters, max_iter=300, n_init=10, tol=1e-4, random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.n_init = n_init
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        if self.random_state is not None:
            np.random.seed(self.random_state)
            
        best_inertia = np.inf
        best_centers = None
        best_labels = None
        
        for _ in range(self.n_init):
            # 1. Initialize centroids randomly from data points
            idx = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            centers = X[idx].copy()
            
            for i in range(self.max_iter):
                # 2. Assignment step
                # Compute distances from each point to each center
                # (X - centers)^2 = X^2 - 2*X*centers + centers^2
                # But simple broadcast is easier for moderate dims
                # dists: (N, K)
                dists = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
                labels = np.argmin(dists, axis=1)
                
                # 3. Update step
                new_centers = np.array([X[labels == k].mean(axis=0) if np.any(labels == k) else centers[k] 
                                        for k in range(self.n_clusters)])
                
                # Check for convergence
                shift = np.linalg.norm(centers - new_centers)
                centers = new_centers
                
                if shift < self.tol:
                    break
            
            # Compute inertia (SSE)
            inertia = np.sum((X - centers[labels])**2)
            
            if inertia < best_inertia:
                best_inertia = inertia
                best_centers = centers
                best_labels = labels
        
        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        return self

    def predict(self, X):
        dists = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(dists, axis=1)

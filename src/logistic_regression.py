import numpy as np

class LogisticRegressionOvR:
    def __init__(self, lr=0.01, l2_lambda=0.1, n_iters=1000):
        self.lr = lr
        self.l2_lambda = l2_lambda
        self.n_iters = n_iters
        self.models = []
        self.classes = None

    def _sigmoid(self, z):
        return 1 / (1 + np.exp(-np.clip(z, -250, 250)))

    def _fit_one(self, X, y):
        n_samples, n_features = X.shape
        weights = np.zeros(n_features)
        bias = 0
        
        for _ in range(self.n_iters):
            model = np.dot(X, weights) + bias
            predictions = self._sigmoid(model)
            
            # Gradients with L2 regularization
            dw = (1 / n_samples) * (np.dot(X.T, (predictions - y)) + self.l2_lambda * weights)
            db = (1 / n_samples) * np.sum(predictions - y)
            
            weights -= self.lr * dw
            bias -= self.lr * db
            
        return weights, bias

    def fit(self, X, y):
        self.classes = np.unique(y)
        self.models = []
        
        for c in self.classes:
            print(f"Training OvR classifier for class {c}...")
            y_binary = np.where(y == c, 1, 0)
            model = self._fit_one(X, y_binary)
            self.models.append(model)
        
        return self

    def predict_proba(self, X):
        probs = []
        for weights, bias in self.models:
            z = np.dot(X, weights) + bias
            probs.append(self._sigmoid(z))
        
        probs = np.array(probs).T # (N, K)
        # Normalize to sum to 1? (Not strictly necessary for OvR prediction but good for probs)
        probs = probs / probs.sum(axis=1, keepdims=True)
        return probs

    def predict(self, X):
        probs = self.predict_proba(X)
        return self.classes[np.argmax(probs, axis=1)]

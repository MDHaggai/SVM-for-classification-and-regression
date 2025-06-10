import numpy as np
from sklearn.svm import SVC

class KernelSVM:
    def __init__(self, kernel='rbf', C=1.0, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        # Use sklearn's SVM as the backend for reliability
        self.model = SVC(kernel=kernel, C=C, gamma=gamma, random_state=42)

    def fit(self, X, y):
        """Train the SVM model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate accuracy score."""
        return self.model.score(X, y)

    def decision_function(self, X):
        """Calculate decision function values."""
        return self.model.decision_function(X)
            return np.zeros(X.shape[0])
        
        kernel_matrix = self._kernel(X, self.support_vectors_)
        decision = np.dot(kernel_matrix, self.support_alpha_ * self.support_labels_) + self.b_
        return np.sign(decision)

    def _kernel(self, X1, X2):
        if X2 is None:
            return np.zeros((X1.shape[0], 0))
            
        if self.kernel == 'linear':
            return np.dot(X1, X2.T)
        elif self.kernel == 'rbf':
            # Compute RBF kernel
            pairwise_sq_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-self.gamma * pairwise_sq_dists)
        elif self.kernel == 'polynomial':
            return (np.dot(X1, X2.T) + 1) ** 2
        else:
            raise ValueError("Unknown kernel type")

    def score(self, X, y):
        predictions = self.predict(X)
        return np.mean(predictions == y)
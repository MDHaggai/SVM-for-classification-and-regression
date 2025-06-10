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

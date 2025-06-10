import numpy as np
from sklearn.svm import SVR as SklearnSVR

class SVR:
    def __init__(self, kernel='rbf', C=1.0, epsilon=0.1, gamma='scale'):
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        # Use sklearn's SVR as the backend for reliability
        self.model = SklearnSVR(kernel=kernel, C=C, epsilon=epsilon, gamma=gamma)

    def fit(self, X, y):
        """Train the SVR model."""
        self.model.fit(X, y)
        return self

    def predict(self, X):
        """Make predictions."""
        return self.model.predict(X)

    def score(self, X, y):
        """Calculate R² score."""
        return self.model.score(X, y)

    def _kernel_function(self, x1, x2):
        """Kernel function for educational purposes."""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'polynomial':
            return (np.dot(x1, x2) + 1) ** 2
        elif self.kernel == 'rbf':
            if self.gamma == 'scale':
                gamma_val = 1.0 / x1.shape[0]
            elif self.gamma == 'auto':
                gamma_val = 1.0 / x1.shape[0]
            else:
                gamma_val = self.gamma
            return np.exp(-gamma_val * np.linalg.norm(x1 - x2) ** 2)
        else:
            raise ValueError("Unknown kernel type")

    def _epsilon_insensitive_loss(self, y_true, y_pred):
        """Calculate the epsilon-insensitive loss."""
        return np.maximum(0, np.abs(y_true - y_pred) - self.epsilon)
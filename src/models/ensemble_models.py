from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
import numpy as np
import pandas as pd

class EnsembleModels:
    def __init__(self, model_type='classification', n_estimators=100, random_state=42):
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.random_state = random_state
        
        if model_type == 'classification':
            self.model = RandomForestClassifier(n_estimators=self.n_estimators, random_state=self.random_state)
        elif model_type == 'regression':
            self.model = RandomForestRegressor(n_estimators=self.n_estimators, random_state=self.random_state)
        else:
            raise ValueError("model_type must be either 'classification' or 'regression'")

    def fit(self, X, y):
        self.model.fit(X, y)

    def predict(self, X):
        return self.model.predict(X)

    def evaluate(self, X, y):
        predictions = self.predict(X)
        if self.model_type == 'classification':
            return accuracy_score(y, predictions)
        elif self.model_type == 'regression':
            return mean_absolute_error(y, predictions)

# Example usage:
# ensemble_model = EnsembleModels(model_type='classification')
# ensemble_model.fit(X_train, y_train)
# accuracy = ensemble_model.evaluate(X_test, y_test)
# print(f"Accuracy: {accuracy:.2f}")
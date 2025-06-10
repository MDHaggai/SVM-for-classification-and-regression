from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import pandas as pd

class BaselineModels:
    def __init__(self):
        self.logistic_model = LogisticRegression()
        self.random_forest_classifier = RandomForestClassifier()
        self.random_forest_regressor = RandomForestRegressor()

    def train_logistic_regression(self, X_train, y_train):
        self.logistic_model.fit(X_train, y_train)

    def predict_logistic_regression(self, X_test):
        return self.logistic_model.predict(X_test)

    def evaluate_logistic_regression(self, X_test, y_test):
        predictions = self.predict_logistic_regression(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def train_random_forest_classifier(self, X_train, y_train):
        self.random_forest_classifier.fit(X_train, y_train)

    def predict_random_forest_classifier(self, X_test):
        return self.random_forest_classifier.predict(X_test)

    def evaluate_random_forest_classifier(self, X_test, y_test):
        predictions = self.predict_random_forest_classifier(X_test)
        accuracy = accuracy_score(y_test, predictions)
        return accuracy

    def train_random_forest_regressor(self, X_train, y_train):
        self.random_forest_regressor.fit(X_train, y_train)

    def predict_random_forest_regressor(self, X_test):
        return self.random_forest_regressor.predict(X_test)

    def evaluate_random_forest_regressor(self, X_test, y_test):
        predictions = self.predict_random_forest_regressor(X_test)
        mse = mean_squared_error(y_test, predictions)
        return mse

class LogisticRegressionModel:
    """Simple wrapper class for Logistic Regression to match notebook expectations."""
    
    def __init__(self, **kwargs):
        self.model = LogisticRegression(**kwargs)
    
    def fit(self, X, y):
        """Train the logistic regression model."""
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """Make predictions using the trained model."""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities."""
        return self.model.predict_proba(X)
    
    def score(self, X, y):
        """Get the accuracy score."""
        return self.model.score(X, y)
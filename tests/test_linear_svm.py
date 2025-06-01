"""
Unit tests for Linear SVM implementation.
"""
import unittest
import numpy as np
from sklearn.datasets import make_classification, make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svm.linear_svm import LinearSVM


class TestLinearSVM(unittest.TestCase):
    """Test cases for Linear SVM implementation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate simple binary classification dataset
        self.X, self.y = make_classification(
            n_samples=100, n_features=2, n_redundant=0, 
            n_informative=2, random_state=42, n_clusters_per_class=1
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X = scaler.fit_transform(self.X)
        
        # Convert labels to -1, 1
        self.y[self.y == 0] = -1
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )
        
        # Create SVM instance
        self.svm = LinearSVM(C=1.0, max_iter=1000, tol=1e-3)

    def test_initialization(self):
        """Test SVM initialization with different parameters."""
        # Test default parameters
        svm_default = LinearSVM()
        self.assertEqual(svm_default.C, 1.0)
        self.assertEqual(svm_default.max_iter, 1000)
        self.assertEqual(svm_default.tol, 1e-3)
        
        # Test custom parameters
        svm_custom = LinearSVM(C=0.5, max_iter=500, tol=1e-4)
        self.assertEqual(svm_custom.C, 0.5)
        self.assertEqual(svm_custom.max_iter, 500)
        self.assertEqual(svm_custom.tol, 1e-4)

    def test_fit(self):
        """Test the fitting process."""
        # Test that model can be fitted
        self.svm.fit(self.X_train, self.y_train)
        
        # Check that weights and bias are set
        self.assertIsNotNone(self.svm.w)
        self.assertIsNotNone(self.svm.b)
        self.assertIsNotNone(self.svm.alpha)
        
        # Check dimensions
        self.assertEqual(len(self.svm.w), self.X_train.shape[1])
        self.assertEqual(len(self.svm.alpha), len(self.y_train))

    def test_predict(self):
        """Test prediction functionality."""
        # Fit the model
        self.svm.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = self.svm.predict(self.X_test)
        
        # Check output format
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(all(p in [-1, 1] for p in predictions))
        
        # Test accuracy is reasonable (> 70%)
        accuracy = np.mean(predictions == self.y_test)
        self.assertGreater(accuracy, 0.7)

    def test_decision_function(self):
        """Test decision function."""
        # Fit the model
        self.svm.fit(self.X_train, self.y_train)
        
        # Test decision function
        scores = self.svm.decision_function(self.X_test)
        
        # Check output format
        self.assertEqual(len(scores), len(self.y_test))
        self.assertTrue(isinstance(scores, np.ndarray))
        
        # Check that predictions match decision function sign
        predictions = self.svm.predict(self.X_test)
        predicted_signs = np.sign(scores)
        np.testing.assert_array_equal(predictions, predicted_signs)

    def test_support_vectors(self):
        """Test support vector identification."""
        # Fit the model
        self.svm.fit(self.X_train, self.y_train)
        
        # Get support vectors
        support_vectors = self.svm.get_support_vectors()
        
        # Check that we have some support vectors
        self.assertGreater(len(support_vectors), 0)
        self.assertLessEqual(len(support_vectors), len(self.X_train))

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with single sample
        X_single = self.X_train[:1]
        y_single = self.y_train[:1]
        
        with self.assertRaises((ValueError, RuntimeError)):
            self.svm.fit(X_single, y_single)
        
        # Test prediction before fitting
        svm_unfitted = LinearSVM()
        with self.assertRaises(AttributeError):
            svm_unfitted.predict(self.X_test)

    def test_different_c_values(self):
        """Test model behavior with different C values."""
        c_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for C in c_values:
            svm = LinearSVM(C=C, max_iter=1000)
            svm.fit(self.X_train, self.y_train)
            predictions = svm.predict(self.X_test)
            accuracy = np.mean(predictions == self.y_test)
            accuracies.append(accuracy)
        
        # All should achieve reasonable accuracy
        self.assertTrue(all(acc > 0.6 for acc in accuracies))

    def test_linearly_separable_data(self):
        """Test with perfectly linearly separable data."""
        # Create linearly separable data
        X_sep, y_sep = make_blobs(
            n_samples=50, centers=2, n_features=2, 
            cluster_std=1.0, center_box=(-10.0, 10.0), random_state=42
        )
        y_sep[y_sep == 0] = -1
        
        # Fit SVM
        svm = LinearSVM(C=1.0)
        svm.fit(X_sep, y_sep)
        
        # Should achieve perfect or near-perfect accuracy
        predictions = svm.predict(X_sep)
        accuracy = np.mean(predictions == y_sep)
        self.assertGreaterEqual(accuracy, 0.95)

    def test_convergence(self):
        """Test convergence behavior."""
        # Test with different tolerance values
        tol_values = [1e-2, 1e-3, 1e-4]
        
        for tol in tol_values:
            svm = LinearSVM(C=1.0, tol=tol, max_iter=1000)
            svm.fit(self.X_train, self.y_train)
            
            # Should converge within max_iter
            self.assertLessEqual(svm.n_iter_, svm.max_iter)


if __name__ == '__main__':
    unittest.main()

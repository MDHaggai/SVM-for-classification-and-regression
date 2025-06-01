"""
Unit tests for Kernel SVM implementation.
"""
import unittest
import numpy as np
from sklearn.datasets import make_classification, make_circles
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svm.kernel_svm import KernelSVM
from svm.kernels import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel


class TestKernelSVM(unittest.TestCase):
    """Test cases for Kernel SVM implementation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate linearly separable data for linear kernel
        self.X_linear, self.y_linear = make_classification(
            n_samples=100, n_features=2, n_redundant=0, 
            n_informative=2, random_state=42, n_clusters_per_class=1
        )
        
        # Generate non-linearly separable data for non-linear kernels
        self.X_nonlinear, self.y_nonlinear = make_circles(
            n_samples=100, noise=0.1, factor=0.3, random_state=42
        )
        
        # Scale features
        scaler = StandardScaler()
        self.X_linear = scaler.fit_transform(self.X_linear)
        self.X_nonlinear = scaler.fit_transform(self.X_nonlinear)
        
        # Convert labels to -1, 1
        self.y_linear[self.y_linear == 0] = -1
        self.y_nonlinear[self.y_nonlinear == 0] = -1
        
        # Split data
        self.X_train_linear, self.X_test_linear, self.y_train_linear, self.y_test_linear = train_test_split(
            self.X_linear, self.y_linear, test_size=0.3, random_state=42
        )
        
        self.X_train_nonlinear, self.X_test_nonlinear, self.y_train_nonlinear, self.y_test_nonlinear = train_test_split(
            self.X_nonlinear, self.y_nonlinear, test_size=0.3, random_state=42
        )

    def test_initialization(self):
        """Test KernelSVM initialization with different parameters."""
        # Test default parameters
        svm_default = KernelSVM()
        self.assertEqual(svm_default.C, 1.0)
        self.assertEqual(svm_default.kernel, 'rbf')
        self.assertEqual(svm_default.gamma, 'scale')
        
        # Test custom parameters
        svm_custom = KernelSVM(C=0.5, kernel='polynomial', degree=3, gamma=0.1)
        self.assertEqual(svm_custom.C, 0.5)
        self.assertEqual(svm_custom.kernel, 'polynomial')
        self.assertEqual(svm_custom.degree, 3)
        self.assertEqual(svm_custom.gamma, 0.1)

    def test_linear_kernel(self):
        """Test KernelSVM with linear kernel."""
        svm = KernelSVM(kernel='linear', C=1.0, max_iter=1000)
        svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Test prediction
        predictions = svm.predict(self.X_test_linear)
        accuracy = np.mean(predictions == self.y_test_linear)
        
        # Should achieve reasonable accuracy
        self.assertGreater(accuracy, 0.7)
        
        # Check that attributes are set
        self.assertIsNotNone(svm.alpha)
        self.assertIsNotNone(svm.b)
        self.assertIsNotNone(svm.support_vectors)

    def test_rbf_kernel(self):
        """Test KernelSVM with RBF kernel."""
        svm = KernelSVM(kernel='rbf', C=1.0, gamma=1.0, max_iter=1000)
        svm.fit(self.X_train_nonlinear, self.y_train_nonlinear)
        
        # Test prediction
        predictions = svm.predict(self.X_test_nonlinear)
        accuracy = np.mean(predictions == self.y_test_nonlinear)
        
        # RBF should handle non-linear data better
        self.assertGreater(accuracy, 0.8)

    def test_polynomial_kernel(self):
        """Test KernelSVM with polynomial kernel."""
        svm = KernelSVM(kernel='polynomial', degree=2, C=1.0, max_iter=1000)
        svm.fit(self.X_train_nonlinear, self.y_train_nonlinear)
        
        # Test prediction
        predictions = svm.predict(self.X_test_nonlinear)
        accuracy = np.mean(predictions == self.y_test_nonlinear)
        
        # Should achieve reasonable accuracy
        self.assertGreater(accuracy, 0.7)

    def test_sigmoid_kernel(self):
        """Test KernelSVM with sigmoid kernel."""
        svm = KernelSVM(kernel='sigmoid', C=1.0, gamma=1.0, coef0=0.0, max_iter=1000)
        svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Test prediction
        predictions = svm.predict(self.X_test_linear)
        accuracy = np.mean(predictions == self.y_test_linear)
        
        # Should achieve reasonable accuracy
        self.assertGreater(accuracy, 0.6)

    def test_custom_kernel_function(self):
        """Test KernelSVM with custom kernel function."""
        # Define custom linear kernel
        def custom_linear_kernel(X1, X2):
            return np.dot(X1, X2.T)
        
        svm = KernelSVM(kernel=custom_linear_kernel, C=1.0, max_iter=1000)
        svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Test prediction
        predictions = svm.predict(self.X_test_linear)
        accuracy = np.mean(predictions == self.y_test_linear)
        
        # Should achieve reasonable accuracy
        self.assertGreater(accuracy, 0.7)

    def test_decision_function(self):
        """Test decision function."""
        svm = KernelSVM(kernel='rbf', C=1.0, max_iter=1000)
        svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Test decision function
        scores = svm.decision_function(self.X_test_linear)
        
        # Check output format
        self.assertEqual(len(scores), len(self.y_test_linear))
        self.assertTrue(isinstance(scores, np.ndarray))
        
        # Check that predictions match decision function sign
        predictions = svm.predict(self.X_test_linear)
        predicted_signs = np.sign(scores)
        np.testing.assert_array_equal(predictions, predicted_signs)

    def test_support_vectors(self):
        """Test support vector identification."""
        svm = KernelSVM(kernel='rbf', C=1.0, max_iter=1000)
        svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Check support vectors
        self.assertIsNotNone(svm.support_vectors)
        self.assertIsNotNone(svm.support_vector_labels)
        self.assertIsNotNone(svm.support_vector_alphas)
        
        # Should have some support vectors
        self.assertGreater(len(svm.support_vectors), 0)
        self.assertLessEqual(len(svm.support_vectors), len(self.X_train_linear))

    def test_gamma_parameter(self):
        """Test different gamma values for RBF kernel."""
        gamma_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for gamma in gamma_values:
            svm = KernelSVM(kernel='rbf', gamma=gamma, C=1.0, max_iter=1000)
            svm.fit(self.X_train_nonlinear, self.y_train_nonlinear)
            predictions = svm.predict(self.X_test_nonlinear)
            accuracy = np.mean(predictions == self.y_test_nonlinear)
            accuracies.append(accuracy)
        
        # All should achieve reasonable accuracy
        self.assertTrue(all(acc > 0.6 for acc in accuracies))

    def test_c_parameter(self):
        """Test different C values."""
        c_values = [0.1, 1.0, 10.0]
        accuracies = []
        
        for C in c_values:
            svm = KernelSVM(kernel='rbf', C=C, max_iter=1000)
            svm.fit(self.X_train_linear, self.y_train_linear)
            predictions = svm.predict(self.X_test_linear)
            accuracy = np.mean(predictions == self.y_test_linear)
            accuracies.append(accuracy)
        
        # All should achieve reasonable accuracy
        self.assertTrue(all(acc > 0.6 for acc in accuracies))

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with invalid kernel
        with self.assertRaises(ValueError):
            svm = KernelSVM(kernel='invalid_kernel')
            svm.fit(self.X_train_linear, self.y_train_linear)
        
        # Test prediction before fitting
        svm_unfitted = KernelSVM()
        with self.assertRaises(AttributeError):
            svm_unfitted.predict(self.X_test_linear)

    def test_kernel_matrix_computation(self):
        """Test kernel matrix computation."""
        svm = KernelSVM(kernel='rbf', gamma=1.0)
        
        # Compute kernel matrix
        K = svm._compute_kernel_matrix(self.X_train_linear[:5], self.X_train_linear[:5])
        
        # Check properties
        self.assertEqual(K.shape, (5, 5))
        self.assertTrue(np.allclose(K, K.T))  # Should be symmetric
        self.assertTrue(np.all(np.diag(K) >= 0))  # Diagonal should be non-negative

    def test_gamma_scale_auto(self):
        """Test automatic gamma scaling."""
        # Test 'scale' gamma
        svm_scale = KernelSVM(kernel='rbf', gamma='scale')
        svm_scale.fit(self.X_train_linear, self.y_train_linear)
        
        # Should set gamma based on features
        expected_gamma = 1.0 / (self.X_train_linear.shape[1] * self.X_train_linear.var())
        self.assertAlmostEqual(svm_scale.gamma_, expected_gamma, places=5)
        
        # Test 'auto' gamma
        svm_auto = KernelSVM(kernel='rbf', gamma='auto')
        svm_auto.fit(self.X_train_linear, self.y_train_linear)
        
        # Should set gamma based on features
        expected_gamma = 1.0 / self.X_train_linear.shape[1]
        self.assertAlmostEqual(svm_auto.gamma_, expected_gamma, places=5)


if __name__ == '__main__':
    unittest.main()

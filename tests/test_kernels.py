"""
Unit tests for kernel functions.
"""
import unittest
import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svm.kernels import linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel


class TestKernelFunctions(unittest.TestCase):
    """Test cases for kernel function implementations."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Create sample data
        np.random.seed(42)
        self.X1 = np.random.randn(5, 3)
        self.X2 = np.random.randn(4, 3)
        self.X_square = np.random.randn(3, 3)

    def test_linear_kernel(self):
        """Test linear kernel function."""
        # Test basic functionality
        K = linear_kernel(self.X1, self.X2)
        
        # Check output shape
        self.assertEqual(K.shape, (5, 4))
        
        # Manual computation for verification
        K_manual = np.dot(self.X1, self.X2.T)
        np.testing.assert_array_almost_equal(K, K_manual)
        
        # Test with same input (should be symmetric)
        K_sym = linear_kernel(self.X_square, self.X_square)
        self.assertEqual(K_sym.shape, (3, 3))
        np.testing.assert_array_almost_equal(K_sym, K_sym.T)

    def test_polynomial_kernel(self):
        """Test polynomial kernel function."""
        # Test with default parameters
        K = polynomial_kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (5, 4))
        
        # Test with custom parameters
        K_custom = polynomial_kernel(self.X1, self.X2, degree=2, gamma=0.5, coef0=1.0)
        self.assertEqual(K_custom.shape, (5, 4))
        
        # Manual computation for degree=2, gamma=1.0, coef0=0.0
        linear_part = np.dot(self.X1, self.X2.T)
        K_manual = (1.0 * linear_part + 0.0) ** 2
        K_test = polynomial_kernel(self.X1, self.X2, degree=2, gamma=1.0, coef0=0.0)
        np.testing.assert_array_almost_equal(K_test, K_manual)
        
        # Test symmetry
        K_sym = polynomial_kernel(self.X_square, self.X_square, degree=3)
        np.testing.assert_array_almost_equal(K_sym, K_sym.T)
        
        # Test that all values are non-negative for positive inputs
        X_pos = np.abs(self.X1)
        K_pos = polynomial_kernel(X_pos, X_pos, degree=2, coef0=0.0)
        self.assertTrue(np.all(K_pos >= 0))

    def test_rbf_kernel(self):
        """Test RBF (Gaussian) kernel function."""
        # Test basic functionality
        K = rbf_kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (5, 4))
        
        # All values should be between 0 and 1
        self.assertTrue(np.all(K >= 0))
        self.assertTrue(np.all(K <= 1))
        
        # Test with custom gamma
        K_custom = rbf_kernel(self.X1, self.X2, gamma=0.5)
        self.assertEqual(K_custom.shape, (5, 4))
        
        # Test diagonal should be 1 for identical points
        K_diag = rbf_kernel(self.X_square, self.X_square)
        np.testing.assert_array_almost_equal(np.diag(K_diag), np.ones(3))
        
        # Test symmetry
        np.testing.assert_array_almost_equal(K_diag, K_diag.T)
        
        # Manual computation verification
        gamma = 1.0
        squared_dists = np.sum(self.X1**2, axis=1).reshape(-1, 1) + \
                       np.sum(self.X2**2, axis=1) - 2 * np.dot(self.X1, self.X2.T)
        K_manual = np.exp(-gamma * squared_dists)
        K_test = rbf_kernel(self.X1, self.X2, gamma=gamma)
        np.testing.assert_array_almost_equal(K_test, K_manual)

    def test_sigmoid_kernel(self):
        """Test sigmoid (tanh) kernel function."""
        # Test basic functionality
        K = sigmoid_kernel(self.X1, self.X2)
        self.assertEqual(K.shape, (5, 4))
        
        # All values should be between -1 and 1
        self.assertTrue(np.all(K >= -1))
        self.assertTrue(np.all(K <= 1))
        
        # Test with custom parameters
        K_custom = sigmoid_kernel(self.X1, self.X2, gamma=0.5, coef0=1.0)
        self.assertEqual(K_custom.shape, (5, 4))
        
        # Test symmetry
        K_sym = sigmoid_kernel(self.X_square, self.X_square)
        np.testing.assert_array_almost_equal(K_sym, K_sym.T)
        
        # Manual computation verification
        gamma = 1.0
        coef0 = 0.0
        linear_part = np.dot(self.X1, self.X2.T)
        K_manual = np.tanh(gamma * linear_part + coef0)
        K_test = sigmoid_kernel(self.X1, self.X2, gamma=gamma, coef0=coef0)
        np.testing.assert_array_almost_equal(K_test, K_manual)

    def test_kernel_properties(self):
        """Test mathematical properties of kernels."""
        # Test that kernels produce positive semi-definite matrices
        kernels_to_test = [
            (linear_kernel, {}),
            (polynomial_kernel, {'degree': 2}),
            (rbf_kernel, {'gamma': 1.0}),
        ]
        
        for kernel_func, params in kernels_to_test:
            K = kernel_func(self.X_square, self.X_square, **params)
            
            # Check symmetry
            np.testing.assert_array_almost_equal(K, K.T, 
                                               err_msg=f"Kernel {kernel_func.__name__} not symmetric")
            
            # Check positive semi-definiteness (all eigenvalues >= 0)
            eigenvals = np.linalg.eigvals(K)
            self.assertTrue(np.all(eigenvals >= -1e-10), 
                          f"Kernel {kernel_func.__name__} not positive semi-definite")

    def test_gamma_parameter_effects(self):
        """Test the effect of gamma parameter on RBF and sigmoid kernels."""
        gamma_values = [0.1, 1.0, 10.0]
        
        # Test RBF kernel
        for i, gamma1 in enumerate(gamma_values[:-1]):
            gamma2 = gamma_values[i + 1]
            
            K1 = rbf_kernel(self.X1, self.X2, gamma=gamma1)
            K2 = rbf_kernel(self.X1, self.X2, gamma=gamma2)
            
            # Higher gamma should generally lead to smaller kernel values
            # (except for very close points)
            mean_K1 = np.mean(K1)
            mean_K2 = np.mean(K2)
            
            # This is a general trend, not always true for all points
            self.assertNotEqual(mean_K1, mean_K2)

    def test_polynomial_degree_effects(self):
        """Test the effect of degree parameter on polynomial kernel."""
        degrees = [1, 2, 3, 4]
        
        for degree in degrees:
            K = polynomial_kernel(self.X_square, self.X_square, degree=degree)
            
            # Check that diagonal elements are correct
            # For polynomial kernel: K(x,x) = (gamma * x·x + coef0)^degree
            gamma = 1.0
            coef0 = 0.0
            diagonal_manual = []
            for i in range(len(self.X_square)):
                x = self.X_square[i]
                val = (gamma * np.dot(x, x) + coef0) ** degree
                diagonal_manual.append(val)
            
            np.testing.assert_array_almost_equal(np.diag(K), diagonal_manual)

    def test_edge_cases(self):
        """Test edge cases and boundary conditions."""
        # Test with single point
        X_single = np.array([[1.0, 2.0, 3.0]])
        
        for kernel_func in [linear_kernel, polynomial_kernel, rbf_kernel, sigmoid_kernel]:
            K = kernel_func(X_single, X_single)
            self.assertEqual(K.shape, (1, 1))
        
        # Test with zero vectors
        X_zero = np.zeros((2, 3))
        
        K_linear = linear_kernel(X_zero, X_zero)
        np.testing.assert_array_almost_equal(K_linear, np.zeros((2, 2)))
        
        K_rbf = rbf_kernel(X_zero, X_zero)
        np.testing.assert_array_almost_equal(K_rbf, np.ones((2, 2)))

    def test_numerical_stability(self):
        """Test numerical stability with extreme values."""
        # Test with very large values
        X_large = np.array([[1e6, 1e6], [1e6, -1e6]])
        
        # RBF kernel should handle large values gracefully
        K_rbf = rbf_kernel(X_large, X_large, gamma=1e-12)
        self.assertTrue(np.all(np.isfinite(K_rbf)))
        self.assertTrue(np.all(K_rbf >= 0))
        self.assertTrue(np.all(K_rbf <= 1))
        
        # Test with very small values
        X_small = np.array([[1e-10, 1e-10], [1e-10, -1e-10]])
        
        K_rbf_small = rbf_kernel(X_small, X_small)
        self.assertTrue(np.all(np.isfinite(K_rbf_small)))

    def test_kernel_consistency(self):
        """Test consistency between different ways of computing kernels."""
        # Test that kernel(X, Y) gives same result as kernel(Y, X).T
        K1 = rbf_kernel(self.X1, self.X2)
        K2 = rbf_kernel(self.X2, self.X1)
        
        np.testing.assert_array_almost_equal(K1, K2.T)
        
        # Same for other kernels
        for kernel_func in [linear_kernel, polynomial_kernel, sigmoid_kernel]:
            K1 = kernel_func(self.X1, self.X2)
            K2 = kernel_func(self.X2, self.X1)
            np.testing.assert_array_almost_equal(K1, K2.T)


if __name__ == '__main__':
    unittest.main()

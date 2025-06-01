"""
Unit tests for Support Vector Regression (SVR) implementation.
"""
import unittest
import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from svm.svr import SupportVectorRegressor as SVR


class TestSVR(unittest.TestCase):
    """Test cases for Support Vector Regression implementation."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        # Generate regression dataset
        self.X, self.y = make_regression(
            n_samples=100, n_features=2, noise=0.1, random_state=42
        )
        
        # Scale features and targets
        self.scaler_X = StandardScaler()
        self.scaler_y = StandardScaler()
        
        self.X = self.scaler_X.fit_transform(self.X)
        self.y = self.scaler_y.fit_transform(self.y.reshape(-1, 1)).ravel()
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.3, random_state=42
        )

    def test_initialization(self):
        """Test SVR initialization with different parameters."""
        # Test default parameters
        svr_default = SVR()
        self.assertEqual(svr_default.C, 1.0)
        self.assertEqual(svr_default.epsilon, 0.1)
        self.assertEqual(svr_default.kernel, 'rbf')
        self.assertEqual(svr_default.gamma, 'scale')
        
        # Test custom parameters
        svr_custom = SVR(C=0.5, epsilon=0.01, kernel='linear', max_iter=500)
        self.assertEqual(svr_custom.C, 0.5)
        self.assertEqual(svr_custom.epsilon, 0.01)
        self.assertEqual(svr_custom.kernel, 'linear')
        self.assertEqual(svr_custom.max_iter, 500)

    def test_linear_svr(self):
        """Test SVR with linear kernel."""
        svr = SVR(kernel='linear', C=1.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = svr.predict(self.X_test)
        
        # Check output format
        self.assertEqual(len(predictions), len(self.y_test))
        self.assertTrue(isinstance(predictions, np.ndarray))
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        mae = mean_absolute_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        # Should achieve reasonable performance
        self.assertLess(mse, 1.0)  # MSE should be reasonable
        self.assertLess(mae, 1.0)  # MAE should be reasonable
        self.assertGreater(r2, 0.5)  # R² should be decent
        
        # Check that attributes are set
        self.assertIsNotNone(svr.alpha)
        self.assertIsNotNone(svr.b)

    def test_rbf_svr(self):
        """Test SVR with RBF kernel."""
        svr = SVR(kernel='rbf', C=1.0, gamma=1.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = svr.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        # RBF should handle non-linear relationships well
        self.assertLess(mse, 1.0)
        self.assertGreater(r2, 0.5)

    def test_polynomial_svr(self):
        """Test SVR with polynomial kernel."""
        svr = SVR(kernel='polynomial', degree=2, C=1.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = svr.predict(self.X_test)
        
        # Calculate metrics
        mse = mean_squared_error(self.y_test, predictions)
        r2 = r2_score(self.y_test, predictions)
        
        # Should achieve reasonable performance
        self.assertLess(mse, 1.5)
        self.assertGreater(r2, 0.3)

    def test_support_vectors(self):
        """Test support vector identification in regression."""
        svr = SVR(kernel='rbf', C=1.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Check support vectors
        support_vectors = svr.get_support_vectors()
        
        # Should have some support vectors
        self.assertIsNotNone(support_vectors)
        self.assertGreater(len(support_vectors), 0)
        self.assertLessEqual(len(support_vectors), len(self.X_train))

    def test_epsilon_parameter(self):
        """Test different epsilon values."""
        epsilon_values = [0.01, 0.1, 0.5]
        mse_values = []
        
        for epsilon in epsilon_values:
            svr = SVR(kernel='rbf', epsilon=epsilon, C=1.0, max_iter=1000)
            svr.fit(self.X_train, self.y_train)
            predictions = svr.predict(self.X_test)
            mse = mean_squared_error(self.y_test, predictions)
            mse_values.append(mse)
        
        # All should achieve reasonable performance
        self.assertTrue(all(mse < 2.0 for mse in mse_values))

    def test_c_parameter(self):
        """Test different C values."""
        c_values = [0.1, 1.0, 10.0]
        r2_values = []
        
        for C in c_values:
            svr = SVR(kernel='rbf', C=C, epsilon=0.1, max_iter=1000)
            svr.fit(self.X_train, self.y_train)
            predictions = svr.predict(self.X_test)
            r2 = r2_score(self.y_test, predictions)
            r2_values.append(r2)
        
        # All should achieve reasonable performance
        self.assertTrue(all(r2 > 0.3 for r2 in r2_values))

    def test_gamma_parameter(self):
        """Test different gamma values for RBF kernel."""
        gamma_values = [0.1, 1.0, 10.0]
        r2_values = []
        
        for gamma in gamma_values:
            svr = SVR(kernel='rbf', gamma=gamma, C=1.0, epsilon=0.1, max_iter=1000)
            svr.fit(self.X_train, self.y_train)
            predictions = svr.predict(self.X_test)
            r2 = r2_score(self.y_test, predictions)
            r2_values.append(r2)
        
        # All should achieve reasonable performance
        self.assertTrue(all(r2 > 0.2 for r2 in r2_values))

    def test_prediction_interval(self):
        """Test prediction within epsilon tube."""
        svr = SVR(kernel='linear', C=10.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Predict on training data
        train_predictions = svr.predict(self.X_train)
        
        # Calculate residuals
        residuals = np.abs(self.y_train - train_predictions)
        
        # Most training points should be within epsilon or be support vectors
        # This is a soft constraint due to the C parameter
        within_epsilon = np.sum(residuals <= svr.epsilon)
        total_points = len(self.y_train)
        
        # At least some points should be within epsilon
        self.assertGreater(within_epsilon, total_points * 0.3)

    def test_custom_kernel_function(self):
        """Test SVR with custom kernel function."""
        # Define custom RBF kernel
        def custom_rbf_kernel(X1, X2, gamma=1.0):
            squared_dists = np.sum(X1**2, axis=1).reshape(-1, 1) + \
                           np.sum(X2**2, axis=1) - 2 * np.dot(X1, X2.T)
            return np.exp(-gamma * squared_dists)
        
        svr = SVR(kernel=custom_rbf_kernel, C=1.0, epsilon=0.1, max_iter=1000)
        svr.fit(self.X_train, self.y_train)
        
        # Test prediction
        predictions = svr.predict(self.X_test)
        r2 = r2_score(self.y_test, predictions)
        
        # Should achieve reasonable performance
        self.assertGreater(r2, 0.3)

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Test with very small dataset
        X_small = self.X_train[:3]
        y_small = self.y_train[:3]
        
        svr = SVR(max_iter=100)
        # Should still work but might not converge well
        svr.fit(X_small, y_small)
        predictions = svr.predict(X_small)
        self.assertEqual(len(predictions), len(y_small))
        
        # Test prediction before fitting
        svr_unfitted = SVR()
        with self.assertRaises(AttributeError):
            svr_unfitted.predict(self.X_test)

    def test_convergence(self):
        """Test convergence behavior."""
        # Test with different tolerance values
        tol_values = [1e-2, 1e-3, 1e-4]
        
        for tol in tol_values:
            svr = SVR(kernel='linear', tol=tol, max_iter=1000)
            svr.fit(self.X_train, self.y_train)
            
            # Should converge within max_iter
            self.assertLessEqual(svr.n_iter_, svr.max_iter)

    def test_gamma_scale_auto(self):
        """Test automatic gamma scaling."""
        # Test 'scale' gamma
        svr_scale = SVR(kernel='rbf', gamma='scale')
        svr_scale.fit(self.X_train, self.y_train)
        
        # Should set gamma based on features
        expected_gamma = 1.0 / (self.X_train.shape[1] * self.X_train.var())
        self.assertAlmostEqual(svr_scale.gamma_, expected_gamma, places=5)
        
        # Test 'auto' gamma
        svr_auto = SVR(kernel='rbf', gamma='auto')
        svr_auto.fit(self.X_train, self.y_train)
        
        # Should set gamma based on features
        expected_gamma = 1.0 / self.X_train.shape[1]
        self.assertAlmostEqual(svr_auto.gamma_, expected_gamma, places=5)

    def test_nonlinear_regression(self):
        """Test SVR on non-linear regression problem."""
        # Generate non-linear data
        np.random.seed(42)
        X_nonlinear = np.random.uniform(-2, 2, (100, 1))
        y_nonlinear = X_nonlinear.ravel()**2 + 0.1 * np.random.randn(100)
        
        X_train_nl, X_test_nl, y_train_nl, y_test_nl = train_test_split(
            X_nonlinear, y_nonlinear, test_size=0.3, random_state=42
        )
        
        # Linear SVR should struggle
        svr_linear = SVR(kernel='linear', C=1.0, epsilon=0.1)
        svr_linear.fit(X_train_nl, y_train_nl)
        pred_linear = svr_linear.predict(X_test_nl)
        r2_linear = r2_score(y_test_nl, pred_linear)
        
        # RBF SVR should do better
        svr_rbf = SVR(kernel='rbf', C=1.0, gamma=1.0, epsilon=0.1)
        svr_rbf.fit(X_train_nl, y_train_nl)
        pred_rbf = svr_rbf.predict(X_test_nl)
        r2_rbf = r2_score(y_test_nl, pred_rbf)
        
        # RBF should outperform linear for this non-linear problem
        self.assertGreater(r2_rbf, r2_linear)
        self.assertGreater(r2_rbf, 0.7)  # RBF should achieve good performance


if __name__ == '__main__':
    unittest.main()

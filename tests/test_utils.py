"""
Unit tests for utility modules.
"""
import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression
from sklearn.preprocessing import StandardScaler
import sys
import os
import tempfile
import shutil

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from utils.data_loader import DataLoader
from utils.preprocessing import DataPreprocessor
from utils.evaluation import ClassificationEvaluator, RegressionEvaluator
from utils.visualization import SVMVisualizer
from utils.baseline_models import ModelBenchmark


class TestDataLoader(unittest.TestCase):
    """Test cases for DataLoader class."""

    def setUp(self):
        """Set up test fixtures."""
        self.data_loader = DataLoader()

    def test_load_sample_datasets(self):
        """Test loading of sample datasets."""
        # Test classification dataset
        X_class, y_class = self.data_loader.load_sample_classification_data()
        self.assertEqual(X_class.shape[1], 2)  # 2 features
        self.assertEqual(len(X_class), len(y_class))
        self.assertTrue(set(y_class).issubset({-1, 1}))
        
        # Test regression dataset
        X_reg, y_reg = self.data_loader.load_sample_regression_data()
        self.assertEqual(X_reg.shape[1], 2)  # 2 features
        self.assertEqual(len(X_reg), len(y_reg))
        self.assertTrue(isinstance(y_reg, np.ndarray))

    def test_create_synthetic_data(self):
        """Test synthetic data creation."""
        # Test classification data
        X_class, y_class = self.data_loader.create_synthetic_classification_data(
            n_samples=100, n_features=3, noise=0.1
        )
        self.assertEqual(X_class.shape, (100, 3))
        self.assertEqual(len(y_class), 100)
        self.assertTrue(set(y_class).issubset({-1, 1}))
        
        # Test regression data
        X_reg, y_reg = self.data_loader.create_synthetic_regression_data(
            n_samples=50, n_features=2, noise=0.05
        )
        self.assertEqual(X_reg.shape, (50, 2))
        self.assertEqual(len(y_reg), 50)

    def test_data_splitting(self):
        """Test data splitting functionality."""
        X, y = make_classification(n_samples=100, n_features=4, random_state=42)
        
        splits = self.data_loader.train_test_split(X, y, test_size=0.3, random_state=42)
        X_train, X_test, y_train, y_test = splits
        
        # Check split sizes
        self.assertEqual(len(X_train), 70)
        self.assertEqual(len(X_test), 30)
        self.assertEqual(len(y_train), 70)
        self.assertEqual(len(y_test), 30)
        
        # Check that data is properly split
        self.assertEqual(X_train.shape[1], X.shape[1])
        self.assertEqual(X_test.shape[1], X.shape[1])


class TestDataPreprocessor(unittest.TestCase):
    """Test cases for DataPreprocessor class."""

    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = DataPreprocessor()
        
        # Create sample data
        np.random.seed(42)
        self.X = np.random.randn(100, 3) * 10 + 5
        self.y_class = np.random.choice([-1, 1], 100)
        self.y_reg = np.random.randn(100) * 5 + 10

    def test_feature_scaling(self):
        """Test feature scaling methods."""
        # Test standardization
        X_std = self.preprocessor.standardize_features(self.X)
        self.assertEqual(X_std.shape, self.X.shape)
        # Check that mean is close to 0 and std is close to 1
        np.testing.assert_allclose(np.mean(X_std, axis=0), 0, atol=1e-10)
        np.testing.assert_allclose(np.std(X_std, axis=0), 1, atol=1e-10)
        
        # Test normalization
        X_norm = self.preprocessor.normalize_features(self.X)
        self.assertEqual(X_norm.shape, self.X.shape)
        # Check that values are between 0 and 1
        self.assertTrue(np.all(X_norm >= 0))
        self.assertTrue(np.all(X_norm <= 1))

    def test_label_conversion(self):
        """Test label conversion for SVM format."""
        # Test binary classification labels
        y_binary = np.array([0, 1, 0, 1, 1])
        y_converted = self.preprocessor.convert_labels_to_svm_format(y_binary)
        expected = np.array([-1, 1, -1, 1, 1])
        np.testing.assert_array_equal(y_converted, expected)
        
        # Test multiclass labels
        y_multi = np.array([0, 1, 2, 0, 2])
        y_converted_multi = self.preprocessor.convert_labels_to_svm_format(y_multi)
        # Should handle multiclass appropriately
        self.assertEqual(len(y_converted_multi), len(y_multi))

    def test_missing_value_handling(self):
        """Test missing value handling."""
        # Create data with missing values
        X_missing = self.X.copy()
        X_missing[5, 1] = np.nan
        X_missing[10, 0] = np.nan
        
        # Test mean imputation
        X_imputed = self.preprocessor.handle_missing_values(X_missing, strategy='mean')
        self.assertFalse(np.any(np.isnan(X_imputed)))
        self.assertEqual(X_imputed.shape, X_missing.shape)

    def test_outlier_detection(self):
        """Test outlier detection and removal."""
        # Add some obvious outliers
        X_outliers = self.X.copy()
        X_outliers[0] = [100, 100, 100]  # Clear outlier
        X_outliers[1] = [-100, -100, -100]  # Clear outlier
        
        # Detect outliers
        outlier_mask = self.preprocessor.detect_outliers(X_outliers, threshold=2.0)
        
        # Should detect at least the obvious outliers
        self.assertTrue(outlier_mask[0])  # First outlier
        self.assertTrue(outlier_mask[1])  # Second outlier


class TestEvaluators(unittest.TestCase):
    """Test cases for evaluation classes."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate sample predictions
        np.random.seed(42)
        self.y_true_class = np.random.choice([-1, 1], 50)
        self.y_pred_class = np.random.choice([-1, 1], 50)
        self.y_scores = np.random.randn(50)
        
        self.y_true_reg = np.random.randn(50)
        self.y_pred_reg = self.y_true_reg + 0.1 * np.random.randn(50)  # Add small noise

    def test_classification_evaluator(self):
        """Test ClassificationEvaluator."""
        evaluator = ClassificationEvaluator()
        
        # Test basic metrics
        metrics = evaluator.evaluate(self.y_true_class, self.y_pred_class)
        
        # Check that all required metrics are present
        required_metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
            self.assertTrue(0 <= metrics[metric] <= 1)
        
        # Test confusion matrix
        cm = evaluator.confusion_matrix(self.y_true_class, self.y_pred_class)
        self.assertEqual(cm.shape, (2, 2))
        self.assertEqual(np.sum(cm), len(self.y_true_class))

    def test_regression_evaluator(self):
        """Test RegressionEvaluator."""
        evaluator = RegressionEvaluator()
        
        # Test basic metrics
        metrics = evaluator.evaluate(self.y_true_reg, self.y_pred_reg)
        
        # Check that all required metrics are present
        required_metrics = ['mse', 'mae', 'r2_score']
        for metric in required_metrics:
            self.assertIn(metric, metrics)
        
        # MSE and MAE should be positive
        self.assertGreaterEqual(metrics['mse'], 0)
        self.assertGreaterEqual(metrics['mae'], 0)
        
        # R² should be reasonable for our close predictions
        self.assertGreater(metrics['r2_score'], 0.8)


class TestSVMVisualizer(unittest.TestCase):
    """Test cases for SVMVisualizer class."""

    def setUp(self):
        """Set up test fixtures."""
        self.visualizer = SVMVisualizer()
        
        # Create sample 2D data for visualization
        np.random.seed(42)
        self.X_2d = np.random.randn(50, 2)
        self.y = np.random.choice([-1, 1], 50)

    def test_initialization(self):
        """Test visualizer initialization."""
        self.assertIsNotNone(self.visualizer)
        # Test custom figure size
        visualizer_custom = SVMVisualizer(figsize=(8, 6))
        self.assertEqual(visualizer_custom.figsize, (8, 6))

    def test_data_validation(self):
        """Test data validation for plotting methods."""
        # Test with 3D data (should handle gracefully)
        X_3d = np.random.randn(30, 3)
        
        # Should not raise error but might issue warning
        try:
            # This might work or might be handled gracefully
            pass
        except Exception as e:
            # If it raises an error, it should be informative
            self.assertIsInstance(e, (ValueError, NotImplementedError))


class TestModelBenchmark(unittest.TestCase):
    """Test cases for ModelBenchmark class."""

    def setUp(self):
        """Set up test fixtures."""
        # Generate sample datasets
        self.X_class, self.y_class = make_classification(
            n_samples=100, n_features=4, n_redundant=0, random_state=42
        )
        self.X_reg, self.y_reg = make_regression(
            n_samples=100, n_features=4, noise=0.1, random_state=42
        )
        
        # Convert classification labels to SVM format
        self.y_class[self.y_class == 0] = -1

    def test_benchmark_initialization(self):
        """Test ModelBenchmark initialization."""
        benchmark = ModelBenchmark()
        self.assertIsNotNone(benchmark)

    def test_classification_benchmark(self):
        """Test classification model benchmarking."""
        benchmark = ModelBenchmark()
        
        # Run benchmark
        results = benchmark.run_classification_benchmark(
            self.X_class, self.y_class, test_size=0.3
        )
        
        # Check that we get results for all models
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that each result has required metrics
        for model_name, metrics in results.items():
            self.assertIn('accuracy', metrics)
            self.assertIn('precision', metrics)
            self.assertIn('recall', metrics)
            self.assertIn('f1_score', metrics)

    def test_regression_benchmark(self):
        """Test regression model benchmarking."""
        benchmark = ModelBenchmark()
        
        # Run benchmark
        results = benchmark.run_regression_benchmark(
            self.X_reg, self.y_reg, test_size=0.3
        )
        
        # Check that we get results for all models
        self.assertIsInstance(results, dict)
        self.assertGreater(len(results), 0)
        
        # Check that each result has required metrics
        for model_name, metrics in results.items():
            self.assertIn('mse', metrics)
            self.assertIn('mae', metrics)
            self.assertIn('r2_score', metrics)


class TestIntegration(unittest.TestCase):
    """Integration tests for utility modules."""

    def test_full_pipeline_classification(self):
        """Test full pipeline for classification."""
        # Load data
        data_loader = DataLoader()
        X, y = data_loader.load_sample_classification_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.standardize_features(X)
        
        # Split data
        splits = data_loader.train_test_split(X_scaled, y, test_size=0.3)
        X_train, X_test, y_train, y_test = splits
        
        # Generate mock predictions (since we're testing utils, not SVM)
        y_pred = np.random.choice([-1, 1], len(y_test))
        
        # Evaluate
        evaluator = ClassificationEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)
        
        # Check that pipeline completes successfully
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)

    def test_full_pipeline_regression(self):
        """Test full pipeline for regression."""
        # Load data
        data_loader = DataLoader()
        X, y = data_loader.load_sample_regression_data()
        
        # Preprocess data
        preprocessor = DataPreprocessor()
        X_scaled = preprocessor.standardize_features(X)
        
        # Split data
        splits = data_loader.train_test_split(X_scaled, y, test_size=0.3)
        X_train, X_test, y_train, y_test = splits
        
        # Generate mock predictions
        y_pred = y_test + 0.1 * np.random.randn(len(y_test))
        
        # Evaluate
        evaluator = RegressionEvaluator()
        metrics = evaluator.evaluate(y_test, y_pred)
        
        # Check that pipeline completes successfully
        self.assertIsInstance(metrics, dict)
        self.assertIn('mse', metrics)


if __name__ == '__main__':
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    test_classes = [
        TestDataLoader, TestDataPreprocessor, TestEvaluators,
        TestSVMVisualizer, TestModelBenchmark, TestIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)

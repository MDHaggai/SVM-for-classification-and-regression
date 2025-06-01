"""
Baseline models for comparison with SVM implementations.

This module provides standard scikit-learn implementations of various
machine learning algorithms to serve as baselines for comparison.
"""

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.base import BaseEstimator
from typing import Dict, Any, List, Optional
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """Collection of baseline models for comparison."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.classification_models = {}
        self.regression_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all baseline models."""
        # Classification models
        self.classification_models = {
            'Logistic Regression': LogisticRegression(
                random_state=self.random_state, max_iter=1000
            ),
            'Decision Tree': DecisionTreeClassifier(
                random_state=self.random_state, max_depth=10
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=100, random_state=self.random_state, max_depth=10
            ),
            'K-Nearest Neighbors': KNeighborsClassifier(n_neighbors=5),
            'Naive Bayes': GaussianNB(),
            'SVM (Scikit-learn)': SVC(
                kernel='rbf', random_state=self.random_state, probability=True
            )
        }
        
        # Regression models
        self.regression_models = {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=1.0, random_state=self.random_state),
            'Lasso Regression': Lasso(alpha=1.0, random_state=self.random_state),
            'Decision Tree': DecisionTreeRegressor(
                random_state=self.random_state, max_depth=10
            ),
            'Random Forest': RandomForestRegressor(
                n_estimators=100, random_state=self.random_state, max_depth=10
            ),
            'K-Nearest Neighbors': KNeighborsRegressor(n_neighbors=5),
            'SVR (Scikit-learn)': SVR(kernel='rbf')
        }
    
    def get_classification_models(self) -> Dict[str, BaseEstimator]:
        """Get all classification models."""
        return self.classification_models.copy()
    
    def get_regression_models(self) -> Dict[str, BaseEstimator]:
        """Get all regression models."""
        return self.regression_models.copy()
    
    def get_model(self, model_name: str, task_type: str) -> Optional[BaseEstimator]:
        """
        Get a specific model by name and task type.
        
        Args:
            model_name: Name of the model
            task_type: 'classification' or 'regression'
            
        Returns:
            Model instance or None if not found
        """
        if task_type == 'classification':
            return self.classification_models.get(model_name)
        elif task_type == 'regression':
            return self.regression_models.get(model_name)
        else:
            raise ValueError("task_type must be 'classification' or 'regression'")


class ModelBenchmark:
    """Benchmark multiple models on a dataset."""
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.baseline_models = BaselineModels(random_state)
        self.results = {}
    
    def run_classification_benchmark(self, X_train: np.ndarray, X_test: np.ndarray,
                                   y_train: np.ndarray, y_test: np.ndarray,
                                   custom_models: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
        """
        Run benchmark on classification task.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
            custom_models: Additional custom models to include
            
        Returns:
            Dictionary with results for each model
        """
        from .evaluation import ClassificationEvaluator
        
        # Get baseline models
        models = self.baseline_models.get_classification_models()
        
        # Add custom models if provided
        if custom_models:
            models.update(custom_models)
        
        evaluator = ClassificationEvaluator()
        results = {}
        
        print("Running classification benchmark...")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                print(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    try:
                        y_prob_full = model.predict_proba(X_test)
                        # For binary classification, use positive class probability
                        if y_prob_full.shape[1] == 2:
                            y_prob = y_prob_full[:, 1]
                    except:
                        pass
                elif hasattr(model, 'decision_function'):
                    try:
                        y_prob = model.decision_function(X_test)
                        # Normalize to [0, 1] for binary classification
                        if len(np.unique(y_test)) == 2:
                            from sklearn.preprocessing import MinMaxScaler
                            scaler = MinMaxScaler()
                            y_prob = scaler.fit_transform(y_prob.reshape(-1, 1)).flatten()
                    except:
                        pass
                
                # Evaluate
                metrics = evaluator.evaluate(y_test, y_pred, y_prob)
                results[name] = metrics
                
                # Print basic metrics
                print(f"  Accuracy: {metrics['accuracy']:.4f}")
                print(f"  F1-Score: {metrics['f1_score']:.4f}")
                if 'roc_auc' in metrics:
                    print(f"  ROC AUC: {metrics['roc_auc']:.4f}")
                print()
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                print()
                continue
        
        self.results['classification'] = results
        return results
    
    def run_regression_benchmark(self, X_train: np.ndarray, X_test: np.ndarray,
                                y_train: np.ndarray, y_test: np.ndarray,
                                custom_models: Optional[Dict[str, Any]] = None) -> Dict[str, Dict]:
        """
        Run benchmark on regression task.
        
        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training values
            y_test: Test values
            custom_models: Additional custom models to include
            
        Returns:
            Dictionary with results for each model
        """
        from .evaluation import RegressionEvaluator
        
        # Get baseline models
        models = self.baseline_models.get_regression_models()
        
        # Add custom models if provided
        if custom_models:
            models.update(custom_models)
        
        evaluator = RegressionEvaluator()
        results = {}
        
        print("Running regression benchmark...")
        print("-" * 50)
        
        for name, model in models.items():
            try:
                print(f"Training {name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = evaluator.evaluate(y_test, y_pred)
                results[name] = metrics
                
                # Print basic metrics
                print(f"  RMSE: {metrics['rmse']:.4f}")
                print(f"  R² Score: {metrics['r2_score']:.4f}")
                print(f"  MAE: {metrics['mae']:.4f}")
                print()
                
            except Exception as e:
                print(f"  Error training {name}: {str(e)}")
                print()
                continue
        
        self.results['regression'] = results
        return results
    
    def get_best_model(self, task_type: str, metric: str = None) -> tuple:
        """
        Get the best performing model.
        
        Args:
            task_type: 'classification' or 'regression'
            metric: Specific metric to optimize (optional)
            
        Returns:
            Tuple of (model_name, metrics)
        """
        if task_type not in self.results:
            raise ValueError(f"No results found for {task_type}")
        
        results = self.results[task_type]
        
        if not results:
            return None, None
        
        # Default metrics for optimization
        if metric is None:
            if task_type == 'classification':
                metric = 'accuracy'
            else:
                metric = 'r2_score'
        
        # Find best model
        best_score = None
        best_model = None
        
        for model_name, metrics in results.items():
            if metric in metrics:
                score = metrics[metric]
                
                # For error metrics (lower is better)
                if metric in ['mse', 'rmse', 'mae', 'mape']:
                    if best_score is None or score < best_score:
                        best_score = score
                        best_model = model_name
                # For performance metrics (higher is better)
                else:
                    if best_score is None or score > best_score:
                        best_score = score
                        best_model = model_name
        
        return best_model, results.get(best_model)


def quick_model_comparison(X_train: np.ndarray, X_test: np.ndarray,
                         y_train: np.ndarray, y_test: np.ndarray,
                         task_type: str, custom_models: Optional[Dict] = None) -> Dict:
    """
    Quick function to compare models on a dataset.
    
    Args:
        X_train: Training features
        X_test: Test features
        y_train: Training targets
        y_test: Test targets
        task_type: 'classification' or 'regression'
        custom_models: Additional models to include
        
    Returns:
        Dictionary with comparison results
    """
    benchmark = ModelBenchmark()
    
    if task_type == 'classification':
        results = benchmark.run_classification_benchmark(
            X_train, X_test, y_train, y_test, custom_models
        )
    elif task_type == 'regression':
        results = benchmark.run_regression_benchmark(
            X_train, X_test, y_train, y_test, custom_models
        )
    else:
        raise ValueError("task_type must be 'classification' or 'regression'")
    
    # Get best model
    best_model, best_metrics = benchmark.get_best_model(task_type)
    
    print(f"\nBest Model: {best_model}")
    if best_metrics:
        print("Best Model Metrics:")
        for metric, value in best_metrics.items():
            if isinstance(value, (int, float)):
                print(f"  {metric}: {value:.4f}")
    
    return {
        'all_results': results,
        'best_model': best_model,
        'best_metrics': best_metrics,
        'benchmark_object': benchmark
    }

"""
Baseline models for comparison with SVM implementations.

This module provides implementations of various baseline models for both
classification and regression tasks to establish performance benchmarks.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.model_selection import GridSearchCV
from typing import Dict, Any, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class BaselineModels:
    """
    Collection of baseline models for comparison with custom SVM implementations.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize baseline models.
        
        Args:
            random_state: Random state for reproducibility
        """
        self.random_state = random_state
        self.classification_models = {}
        self.regression_models = {}
        self._initialize_models()
    
    def _initialize_models(self):
        """Initialize all baseline models with default parameters."""
        
        # Classification models
        self.classification_models = {
            'Logistic_Regression': LogisticRegression(random_state=self.random_state, max_iter=1000),
            'Random_Forest': RandomForestClassifier(random_state=self.random_state, n_estimators=100),
            'Decision_Tree': DecisionTreeClassifier(random_state=self.random_state),
            'KNN': KNeighborsClassifier(n_neighbors=5),
            'Naive_Bayes': GaussianNB(),
            'SVM_sklearn': SVC(random_state=self.random_state, probability=True)
        }
        
        # Regression models
        self.regression_models = {
            'Linear_Regression': LinearRegression(),
            'Ridge_Regression': Ridge(random_state=self.random_state),
            'Lasso_Regression': Lasso(random_state=self.random_state),
            'Random_Forest': RandomForestRegressor(random_state=self.random_state, n_estimators=100),
            'Decision_Tree': DecisionTreeRegressor(random_state=self.random_state),
            'KNN': KNeighborsRegressor(n_neighbors=5),
            'SVR_sklearn': SVR()
        }
    
    def get_classification_models(self) -> Dict[str, Any]:
        """Get all classification baseline models."""
        return self.classification_models.copy()
    
    def get_regression_models(self) -> Dict[str, Any]:
        """Get all regression baseline models."""
        return self.regression_models.copy()
    
    def train_all_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                      X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all classification models and return results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary with model results
        """
        from .evaluation import ClassificationEvaluator
        
        results = {}
        evaluator = ClassificationEvaluator()
        
        for model_name, model in self.classification_models.items():
            try:
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Get probabilities if available
                y_prob = None
                if hasattr(model, 'predict_proba'):
                    y_prob_full = model.predict_proba(X_test)
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        y_prob = y_prob_full[:, 1]
                elif hasattr(model, 'decision_function'):
                    if len(np.unique(y_test)) == 2:  # Binary classification
                        y_prob = model.decision_function(X_test)
                
                # Evaluate
                metrics = evaluator.evaluate(y_test, y_pred, y_prob)
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'probabilities': y_prob,
                    'metrics': metrics
                }
                
                print(f"  {model_name} - Accuracy: {metrics['accuracy']:.3f}")
                
            except Exception as e:
                print(f"  {model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results
    
    def train_all_regression_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                   X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Dict[str, Any]]:
        """
        Train all regression models and return results.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_test: Test features
            y_test: Test targets
            
        Returns:
            Dictionary with model results
        """
        from .evaluation import RegressionEvaluator
        
        results = {}
        evaluator = RegressionEvaluator()
        
        for model_name, model in self.regression_models.items():
            try:
                print(f"Training {model_name}...")
                
                # Train model
                model.fit(X_train, y_train)
                
                # Make predictions
                y_pred = model.predict(X_test)
                
                # Evaluate
                metrics = evaluator.evaluate(y_test, y_pred)
                
                results[model_name] = {
                    'model': model,
                    'predictions': y_pred,
                    'metrics': metrics
                }
                
                print(f"  {model_name} - RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2_score']:.3f}")
                
            except Exception as e:
                print(f"  {model_name} failed: {str(e)}")
                results[model_name] = {'error': str(e)}
        
        return results


class HyperparameterTuner:
    """
    Automated hyperparameter tuning for baseline models.
    """
    
    def __init__(self, cv_folds: int = 5, random_state: int = 42):
        """
        Initialize hyperparameter tuner.
        
        Args:
            cv_folds: Number of cross-validation folds
            random_state: Random state for reproducibility
        """
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Define parameter grids
        self.classification_param_grids = {
            'Logistic_Regression': {
                'C': [0.1, 1.0, 10.0],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear']
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Decision_Tree': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'SVM_sklearn': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
        
        self.regression_param_grids = {
            'Ridge_Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Lasso_Regression': {
                'alpha': [0.1, 1.0, 10.0, 100.0]
            },
            'Random_Forest': {
                'n_estimators': [50, 100, 200],
                'max_depth': [None, 10, 20],
                'min_samples_split': [2, 5, 10]
            },
            'Decision_Tree': {
                'max_depth': [None, 5, 10, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            },
            'KNN': {
                'n_neighbors': [3, 5, 7, 9],
                'weights': ['uniform', 'distance'],
                'metric': ['euclidean', 'manhattan']
            },
            'SVR_sklearn': {
                'C': [0.1, 1.0, 10.0],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto']
            }
        }
    
    def tune_classification_models(self, X_train: np.ndarray, y_train: np.ndarray,
                                 models_to_tune: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for classification models.
        
        Args:
            X_train: Training features
            y_train: Training labels
            models_to_tune: List of model names to tune (default: all)
            
        Returns:
            Dictionary with tuned models and best parameters
        """
        baseline_models = BaselineModels(self.random_state)
        models = baseline_models.get_classification_models()
        
        if models_to_tune:
            models = {k: v for k, v in models.items() if k in models_to_tune}
        
        tuned_results = {}
        
        for model_name, model in models.items():
            if model_name in self.classification_param_grids:
                print(f"Tuning {model_name}...")
                
                try:
                    # Skip models that don't have parameter grids defined
                    param_grid = self.classification_param_grids[model_name]
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        model, param_grid, cv=self.cv_folds,
                        scoring='accuracy', n_jobs=-1, verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    tuned_results[model_name] = {
                        'best_model': grid_search.best_estimator_,
                        'best_params': grid_search.best_params_,
                        'best_score': grid_search.best_score_,
                        'cv_results': grid_search.cv_results_
                    }
                    
                    print(f"  Best CV score: {grid_search.best_score_:.3f}")
                    print(f"  Best params: {grid_search.best_params_}")
                    
                except Exception as e:
                    print(f"  {model_name} tuning failed: {str(e)}")
                    tuned_results[model_name] = {'error': str(e)}
            else:
                # Use default model if no parameter grid
                tuned_results[model_name] = {
                    'best_model': model,
                    'best_params': 'default',
                    'best_score': None
                }
        
        return tuned_results
    
    def tune_regression_models(self, X_train: np.ndarray, y_train: np.ndarray,
                             models_to_tune: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Tune hyperparameters for regression models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            models_to_tune: List of model names to tune (default: all)
            
        Returns:
            Dictionary with tuned models and best parameters
        """
        baseline_models = BaselineModels(self.random_state)
        models = baseline_models.get_regression_models()
        
        if models_to_tune:
            models = {k: v for k, v in models.items() if k in models_to_tune}
        
        tuned_results = {}
        
        for model_name, model in models.items():
            if model_name in self.regression_param_grids:
                print(f"Tuning {model_name}...")
                
                try:
                    param_grid = self.regression_param_grids[model_name]
                    
                    # Perform grid search
                    grid_search = GridSearchCV(
                        model, param_grid, cv=self.cv_folds,
                        scoring='neg_mean_squared_error', n_jobs=-1, verbose=0
                    )
                    
                    grid_search.fit(X_train, y_train)
                    
                    tuned_results[model_name] = {
                        'best_model': grid_search.best_estimator_,
                        'best_params': grid_search.best_params_,
                        'best_score': -grid_search.best_score_,  # Convert back to MSE
                        'cv_results': grid_search.cv_results_
                    }
                    
                    print(f"  Best CV MSE: {-grid_search.best_score_:.3f}")
                    print(f"  Best params: {grid_search.best_params_}")
                    
                except Exception as e:
                    print(f"  {model_name} tuning failed: {str(e)}")
                    tuned_results[model_name] = {'error': str(e)}
            else:
                # Use default model if no parameter grid
                tuned_results[model_name] = {
                    'best_model': model,
                    'best_params': 'default',
                    'best_score': None
                }
        
        return tuned_results


def quick_baseline_comparison(X_train: np.ndarray, y_train: np.ndarray,
                            X_test: np.ndarray, y_test: np.ndarray,
                            task_type: str = 'classification') -> Dict[str, Any]:
    """
    Quick comparison of baseline models without hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training targets/labels
        X_test: Test features
        y_test: Test targets/labels
        task_type: 'classification' or 'regression'
        
    Returns:
        Dictionary with comparison results
    """
    baseline_models = BaselineModels()
    
    if task_type == 'classification':
        results = baseline_models.train_all_classification_models(
            X_train, y_train, X_test, y_test
        )
        
        # Extract key metrics for comparison
        comparison = {}
        for model_name, result in results.items():
            if 'metrics' in result:
                comparison[model_name] = {
                    'accuracy': result['metrics']['accuracy'],
                    'f1_score': result['metrics']['f1_score'],
                    'precision': result['metrics']['precision'],
                    'recall': result['metrics']['recall']
                }
        
    else:  # regression
        results = baseline_models.train_all_regression_models(
            X_train, y_train, X_test, y_test
        )
        
        # Extract key metrics for comparison
        comparison = {}
        for model_name, result in results.items():
            if 'metrics' in result:
                comparison[model_name] = {
                    'rmse': result['metrics']['rmse'],
                    'mae': result['metrics']['mae'],
                    'r2_score': result['metrics']['r2_score'],
                    'mse': result['metrics']['mse']
                }
    
    return {
        'detailed_results': results,
        'comparison': comparison
    }


def get_best_baseline_model(comparison_results: Dict[str, Dict[str, float]],
                          task_type: str = 'classification') -> Tuple[str, Dict[str, float]]:
    """
    Get the best performing baseline model.
    
    Args:
        comparison_results: Results from quick_baseline_comparison
        task_type: 'classification' or 'regression'
        
    Returns:
        Tuple of (best_model_name, best_model_metrics)
    """
    comparison = comparison_results['comparison']
    
    if task_type == 'classification':
        # Sort by accuracy
        best_model = max(comparison.items(), key=lambda x: x[1]['accuracy'])
    else:  # regression
        # Sort by R² score (higher is better)
        best_model = max(comparison.items(), key=lambda x: x[1]['r2_score'])
    
    return best_model[0], best_model[1]

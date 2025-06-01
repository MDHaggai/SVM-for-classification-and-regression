"""
Evaluation metrics for SVM models.

This module provides comprehensive evaluation metrics for both classification
and regression tasks, including standard metrics and custom visualizations.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, classification_report, roc_curve, auc,
    mean_squared_error, mean_absolute_error, r2_score
)
from typing import Tuple, Dict, Any, Optional
import pandas as pd


class ClassificationEvaluator:
    """Comprehensive evaluation for classification models."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray, 
                y_prob: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute comprehensive classification metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_prob: Predicted probabilities (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        # Basic metrics
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'confusion_matrix': cm
        }
        
        # ROC AUC for binary classification
        if len(np.unique(y_true)) == 2 and y_prob is not None:
            fpr, tpr, _ = roc_curve(y_true, y_prob)
            roc_auc = auc(fpr, tpr)
            metrics['roc_auc'] = roc_auc
            metrics['fpr'] = fpr
            metrics['tpr'] = tpr
        
        self.metrics = metrics
        return metrics
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            class_names: Optional[list] = None, 
                            title: str = "Confusion Matrix") -> plt.Figure:
        """Plot confusion matrix heatmap."""
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_names, yticklabels=class_names)
        ax.set_title(title)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')
        
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_prob: np.ndarray,
                      title: str = "ROC Curve") -> plt.Figure:
        """Plot ROC curve for binary classification."""
        if len(np.unique(y_true)) != 2:
            raise ValueError("ROC curve is only available for binary classification")
        
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color='darkorange', lw=2,
                label=f'ROC curve (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(title)
        ax.legend(loc="lower right")
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def generate_report(self, y_true: np.ndarray, y_pred: np.ndarray,
                       class_names: Optional[list] = None) -> str:
        """Generate detailed classification report."""
        report = classification_report(y_true, y_pred, 
                                     target_names=class_names)
        return report


class RegressionEvaluator:
    """Comprehensive evaluation for regression models."""
    
    def __init__(self):
        self.metrics = {}
    
    def evaluate(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Compute comprehensive regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            
        Returns:
            Dictionary containing all metrics
        """
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        
        # Additional metrics
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'mape': mape
        }
        
        self.metrics = metrics
        return metrics
    
    def plot_predictions_vs_actual(self, y_true: np.ndarray, y_pred: np.ndarray,
                                  title: str = "Predictions vs Actual") -> plt.Figure:
        """Plot predicted vs actual values."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Scatter plot
        ax.scatter(y_true, y_pred, alpha=0.6, color='blue', s=50)
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2,
                label='Perfect Prediction')
        
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add R² score
        r2 = r2_score(y_true, y_pred)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        return fig
    
    def plot_residuals(self, y_true: np.ndarray, y_pred: np.ndarray,
                      title: str = "Residual Plot") -> plt.Figure:
        """Plot residuals to check model assumptions."""
        residuals = y_true - y_pred
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Residuals vs Predicted
        ax1.scatter(y_pred, residuals, alpha=0.6, color='blue', s=50)
        ax1.axhline(y=0, color='red', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted')
        ax1.grid(True, alpha=0.3)
        
        # Histogram of residuals
        ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig


class ModelComparator:
    """Compare multiple models across different metrics."""
    
    def __init__(self):
        self.results = {}
    
    def add_model_results(self, model_name: str, metrics: Dict[str, float]):
        """Add results for a model."""
        self.results[model_name] = metrics
    
    def compare_models(self, task_type: str = 'classification') -> pd.DataFrame:
        """
        Create comparison DataFrame.
        
        Args:
            task_type: 'classification' or 'regression'
            
        Returns:
            DataFrame with model comparison
        """
        if not self.results:
            raise ValueError("No model results added")
        
        df = pd.DataFrame(self.results).T
        
        # Sort by best metric
        if task_type == 'classification':
            # Sort by accuracy (or F1 score if available)
            sort_col = 'accuracy' if 'accuracy' in df.columns else df.columns[0]
            df = df.sort_values(sort_col, ascending=False)
        else:  # regression
            # Sort by R² score (higher is better) or RMSE (lower is better)
            if 'r2_score' in df.columns:
                df = df.sort_values('r2_score', ascending=False)
            elif 'rmse' in df.columns:
                df = df.sort_values('rmse', ascending=True)
        
        return df
    
    def plot_model_comparison(self, metric: str, 
                            title: str = "Model Comparison") -> plt.Figure:
        """Plot bar chart comparing models on a specific metric."""
        if not self.results:
            raise ValueError("No model results added")
        
        models = list(self.results.keys())
        values = [self.results[model].get(metric, 0) for model in models]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(models, values, color=['skyblue', 'lightcoral', 'lightgreen', 
                                           'gold', 'plum'][:len(models)])
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{value:.3f}', ha='center', va='bottom')
        
        ax.set_ylabel(metric.upper())
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis='y')
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        return fig


def cross_validation_scores(model, X: np.ndarray, y: np.ndarray, 
                          cv: int = 5) -> Dict[str, Any]:
    """
    Perform cross-validation and return detailed scores.
    
    Args:
        model: Model with fit and predict methods
        X: Features
        y: Target values
        cv: Number of cross-validation folds
        
    Returns:
        Dictionary with cross-validation results
    """
    from sklearn.model_selection import cross_validate
    
    # Define scoring metrics based on problem type
    if len(np.unique(y)) <= 10:  # Classification
        scoring = ['accuracy', 'precision_weighted', 'recall_weighted', 'f1_weighted']
    else:  # Regression
        scoring = ['neg_mean_squared_error', 'neg_mean_absolute_error', 'r2']
    
    cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring, 
                               return_train_score=True)
    
    # Process results
    results = {}
    for metric in scoring:
        test_scores = cv_results[f'test_{metric}']
        train_scores = cv_results[f'train_{metric}']
        
        # Handle negative metrics (sklearn convention)
        if metric.startswith('neg_'):
            test_scores = -test_scores
            train_scores = -train_scores
            metric = metric[4:]  # Remove 'neg_' prefix
        
        results[metric] = {
            'test_mean': test_scores.mean(),
            'test_std': test_scores.std(),
            'train_mean': train_scores.mean(),
            'train_std': train_scores.std(),
            'test_scores': test_scores,
            'train_scores': train_scores
        }
    
    return results

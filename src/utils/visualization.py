"""
Visualization Utilities for SVM Analysis
========================================

This module provides comprehensive visualization tools for SVM analysis including:
- Decision boundary plots
- Kernel effect comparisons
- Performance metrics visualization
- Model comparison charts
- 3D visualizations for kernel effects
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import warnings
from pathlib import Path

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

class SVMVisualizer:
    """
    Comprehensive visualization toolkit for SVM analysis
    """
    
    def __init__(self, save_dir: str = "visualizations", figsize: Tuple[int, int] = (12, 8)):
        """
        Initialize SVM visualizer
        
        Args:
            save_dir: Directory to save visualizations
            figsize: Default figure size
        """
        self.save_dir = Path(save_dir)
        self.figsize = figsize
        
        # Create subdirectories
        for subdir in ['decision_boundaries', 'kernel_effects', 'performance_plots', 'comparison_charts']:
            (self.save_dir / subdir).mkdir(parents=True, exist_ok=True)
        
        # Color schemes
        self.colors = {
            'primary': '#2E86AB',
            'secondary': '#A23B72', 
            'accent': '#F18F01',
            'success': '#C73E1D',
            'background': '#F5F5F5'
        }
    
    def plot_decision_boundary_2d(self, X: np.ndarray, y: np.ndarray, 
                                 classifier, title: str = "SVM Decision Boundary",
                                 save_name: Optional[str] = None) -> None:
        """
        Plot 2D decision boundary for SVM classifier
        
        Args:
            X: 2D feature data
            y: Labels
            classifier: Trained classifier with predict method
            title: Plot title
            save_name: Filename to save plot
        """
        if X.shape[1] != 2:
            # Use PCA to reduce to 2D
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            print(f"Reduced {X.shape[1]}D data to 2D using PCA (explained variance: {pca.explained_variance_ratio_.sum():.3f})")
        else:
            X_2d = X
        
        fig, ax = plt.subplots(figsize=self.figsize)
        
        # Create mesh
        h = 0.02  # Step size
        x_min, x_max = X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1
        y_min, y_max = X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        
        # Make predictions on mesh
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        
        # If we used PCA, need to inverse transform or retrain classifier
        if X.shape[1] != 2:
            # For visualization, we'll create a simplified boundary
            # In practice, you'd want to retrain on 2D data or use proper inverse transform
            try:
                Z = classifier.predict(mesh_points)
            except:
                # Fallback: create approximate boundary
                Z = np.zeros(mesh_points.shape[0])
                for i, point in enumerate(mesh_points):
                    # Find nearest training point
                    distances = np.linalg.norm(X_2d - point, axis=1)
                    nearest_idx = np.argmin(distances)
                    Z[i] = y[nearest_idx]
        else:
            Z = classifier.predict(mesh_points)
        
        Z = Z.reshape(xx.shape)
        
        # Plot decision boundary
        ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
        ax.contour(xx, yy, Z, colors='black', linestyles='--', linewidths=0.5)
        
        # Plot data points
        unique_labels = np.unique(y)
        colors = ['red', 'blue', 'green', 'purple', 'orange']
        
        for i, label in enumerate(unique_labels):
            mask = (y == label)
            ax.scatter(X_2d[mask, 0], X_2d[mask, 1], 
                      c=colors[i % len(colors)], label=f'Class {label}',
                      alpha=0.7, s=50)
        
        # Highlight support vectors if available
        if hasattr(classifier, 'support_vectors_'):
            sv = classifier.support_vectors_
            if X.shape[1] != 2:
                sv_2d = pca.transform(sv)
            else:
                sv_2d = sv
            ax.scatter(sv_2d[:, 0], sv_2d[:, 1], s=100, 
                      facecolors='none', edgecolors='black', 
                      linewidths=2, label='Support Vectors')
        
        ax.set_xlabel('Feature 1' if X.shape[1] == 2 else 'PC1')
        ax.set_ylabel('Feature 2' if X.shape[1] == 2 else 'PC2')
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / 'decision_boundaries' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_kernel_comparison(self, X: np.ndarray, y: np.ndarray, 
                              kernels: List[str] = ['linear', 'poly', 'rbf', 'sigmoid'],
                              save_name: Optional[str] = None) -> None:
        \"\"\"
        Compare different kernel effects on the same dataset
        
        Args:
            X: Feature data
            y: Labels
            kernels: List of kernel names to compare
            save_name: Filename to save plot
        \"\"\"
        from ..svm.kernel_svm import KernelSVM
        
        # Reduce to 2D if necessary
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_plot = pca.fit_transform(X)
        else:
            X_plot = X
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes = axes.ravel()
        
        for i, kernel in enumerate(kernels[:4]):
            ax = axes[i]
            
            # Train SVM with current kernel
            svm = KernelSVM(kernel=kernel, C=1.0)
            svm.fit(X_plot, y)  # Use 2D data for visualization
            
            # Create mesh for decision boundary
            h = 0.02
            x_min, x_max = X_plot[:, 0].min() - 1, X_plot[:, 0].max() + 1
            y_min, y_max = X_plot[:, 1].min() - 1, X_plot[:, 1].max() + 1
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                np.arange(y_min, y_max, h))
            
            # Predict on mesh
            Z = svm.predict(np.c_[xx.ravel(), yy.ravel()])
            Z = Z.reshape(xx.shape)
            
            # Plot
            ax.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.RdYlBu)
            ax.contour(xx, yy, Z, colors='black', linestyles='--', linewidths=0.5)
            
            # Plot data points
            unique_labels = np.unique(y)
            colors = ['red', 'blue', 'green', 'purple']
            
            for j, label in enumerate(unique_labels):
                mask = (y == label)
                ax.scatter(X_plot[mask, 0], X_plot[mask, 1], 
                          c=colors[j % len(colors)], label=f'Class {label}',
                          alpha=0.7, s=30)
            
            # Highlight support vectors
            if hasattr(svm, 'support_vectors_'):
                sv = svm.support_vectors_
                ax.scatter(sv[:, 0], sv[:, 1], s=80, 
                          facecolors='none', edgecolors='black', linewidths=1.5)
            
            ax.set_title(f'{kernel.upper()} Kernel')
            ax.set_xlabel('Feature 1')
            ax.set_ylabel('Feature 2')
            ax.grid(True, alpha=0.3)
            
            if i == 0:
                ax.legend()
        
        plt.suptitle('Kernel Comparison on Same Dataset', fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / 'kernel_effects' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_performance_metrics(self, results_dict: Dict[str, Dict[str, float]],
                                metric_names: List[str] = ['accuracy', 'precision', 'recall', 'f1'],
                                title: str = \"Model Performance Comparison\",
                                save_name: Optional[str] = None) -> None:
        \"\"\"
        Plot performance metrics comparison
        
        Args:
            results_dict: Dictionary with model names as keys and metrics as values
            metric_names: List of metrics to plot
            title: Plot title
            save_name: Filename to save plot
        \"\"\"
        # Prepare data
        models = list(results_dict.keys())
        metrics_data = {metric: [] for metric in metric_names}
        
        for model in models:
            for metric in metric_names:
                metrics_data[metric].append(results_dict[model].get(metric, 0))
        
        # Create plot
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        axes = axes.ravel()
        
        for i, metric in enumerate(metric_names[:4]):
            ax = axes[i]
            
            bars = ax.bar(models, metrics_data[metric], 
                         color=[self.colors['primary'], self.colors['secondary'], 
                               self.colors['accent'], self.colors['success']][:len(models)])
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.set_title(f'{metric.capitalize()}')
            ax.set_ylabel('Score')
            ax.set_ylim(0, 1.1)
            ax.grid(True, alpha=0.3, axis='y')
            
            # Rotate x-axis labels if needed
            if len(max(models, key=len)) > 8:
                ax.tick_params(axis='x', rotation=45)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / 'performance_plots' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                             labels: Optional[List[str]] = None,
                             title: str = \"Confusion Matrix\",
                             save_name: Optional[str] = None) -> None:
        \"\"\"
        Plot confusion matrix heatmap
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            labels: Class labels
            title: Plot title
            save_name: Filename to save plot
        \"\"\"
        from sklearn.metrics import confusion_matrix
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        if save_name:
            plt.savefig(self.save_dir / 'performance_plots' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_roc_curves(self, y_true_dict: Dict[str, np.ndarray], 
                       y_scores_dict: Dict[str, np.ndarray],
                       title: str = \"ROC Curves Comparison\",
                       save_name: Optional[str] = None) -> None:
        \"\"\"
        Plot ROC curves for multiple models
        
        Args:
            y_true_dict: Dictionary of true labels for each model
            y_scores_dict: Dictionary of prediction scores for each model
            title: Plot title
            save_name: Filename to save plot
        \"\"\"
        from sklearn.metrics import roc_curve, auc
        
        plt.figure(figsize=self.figsize)
        
        colors = ['blue', 'red', 'green', 'purple', 'orange']
        
        for i, (model_name, y_true) in enumerate(y_true_dict.items()):
            y_scores = y_scores_dict[model_name]
            
            # Compute ROC curve
            fpr, tpr, _ = roc_curve(y_true, y_scores)
            roc_auc = auc(fpr, tpr)
            
            plt.plot(fpr, tpr, color=colors[i % len(colors)], lw=2,
                    label=f'{model_name} (AUC = {roc_auc:.3f})')
        
        # Plot diagonal line
        plt.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--')
        
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(title)
        plt.legend(loc=\"lower right\")
        plt.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / 'performance_plots' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_learning_curves(self, train_scores: np.ndarray, val_scores: np.ndarray,
                           train_sizes: np.ndarray, title: str = \"Learning Curves\",
                           save_name: Optional[str] = None) -> None:
        \"\"\"
        Plot learning curves
        
        Args:
            train_scores: Training scores for different sample sizes
            val_scores: Validation scores for different sample sizes
            train_sizes: Sample sizes
            title: Plot title
            save_name: Filename to save plot
        \"\"\"
        plt.figure(figsize=self.figsize)
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        val_mean = np.mean(val_scores, axis=1)
        val_std = np.std(val_scores, axis=1)
        
        # Plot
        plt.plot(train_sizes, train_mean, 'o-', color=self.colors['primary'],
                label='Training Score')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std,
                        alpha=0.2, color=self.colors['primary'])
        
        plt.plot(train_sizes, val_mean, 'o-', color=self.colors['secondary'],
                label='Validation Score')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std,
                        alpha=0.2, color=self.colors['secondary'])
        
        plt.xlabel('Training Set Size')
        plt.ylabel('Score')
        plt.title(title)
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_name:
            plt.savefig(self.save_dir / 'performance_plots' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_3d_kernel_effect(self, X: np.ndarray, y: np.ndarray, 
                             kernel: str = 'rbf', save_name: Optional[str] = None) -> None:
        \"\"\"
        Create 3D visualization of kernel effect
        
        Args:
            X: Feature data (will be reduced to 2D if needed)
            y: Labels
            kernel: Kernel type
            save_name: Filename to save plot
        \"\"\"
        # Reduce to 2D if necessary
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
        else:
            X_2d = X
        
        # Create mesh
        x_range = np.linspace(X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1, 50)
        y_range = np.linspace(X_2d[:, 1].min() - 1, X_2d[:, 1].max() + 1, 50)
        xx, yy = np.meshgrid(x_range, y_range)
        
        # Train SVM
        from ..svm.kernel_svm import KernelSVM
        svm = KernelSVM(kernel=kernel, C=1.0)
        svm.fit(X_2d, y)
        
        # Get decision function values
        mesh_points = np.c_[xx.ravel(), yy.ravel()]
        zz = svm.decision_function(mesh_points).reshape(xx.shape)
        
        # Create 3D plot
        fig = go.Figure(data=[go.Surface(x=xx, y=yy, z=zz, colorscale='RdBu', opacity=0.7)])
        
        # Add data points
        unique_labels = np.unique(y)
        colors_3d = ['red', 'blue', 'green', 'purple']
        
        for i, label in enumerate(unique_labels):
            mask = (y == label)
            fig.add_trace(go.Scatter3d(
                x=X_2d[mask, 0],
                y=X_2d[mask, 1], 
                z=np.zeros(np.sum(mask)),  # Project to z=0 plane
                mode='markers',
                marker=dict(size=8, color=colors_3d[i % len(colors_3d)]),
                name=f'Class {label}'
            ))
        
        fig.update_layout(
            title=f'3D {kernel.upper()} Kernel Decision Surface',
            scene=dict(
                xaxis_title='Feature 1',
                yaxis_title='Feature 2',
                zaxis_title='Decision Function'
            )
        )
        
        if save_name:
            fig.write_html(str(self.save_dir / 'kernel_effects' / f'{save_name}_3d.html'))
        
        fig.show()
    
    def plot_regression_results(self, y_true: np.ndarray, y_pred: np.ndarray,
                              title: str = \"Regression Results\",
                              save_name: Optional[str] = None) -> None:
        \"\"\"
        Plot regression prediction vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values  
            title: Plot title
            save_name: Filename to save plot
        \"\"\"
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot: predicted vs actual
        ax1.scatter(y_true, y_pred, alpha=0.6, color=self.colors['primary'])
        
        # Perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
        
        ax1.set_xlabel('Actual Values')
        ax1.set_ylabel('Predicted Values')
        ax1.set_title('Predicted vs Actual')
        ax1.grid(True, alpha=0.3)
        
        # Residuals plot
        residuals = y_true - y_pred
        ax2.scatter(y_pred, residuals, alpha=0.6, color=self.colors['secondary'])
        ax2.axhline(y=0, color='r', linestyle='--')
        
        ax2.set_xlabel('Predicted Values')
        ax2.set_ylabel('Residuals')
        ax2.set_title('Residual Plot')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(title, fontsize=16)
        plt.tight_layout()
        
        if save_name:
            plt.savefig(self.save_dir / 'performance_plots' / f'{save_name}.png', 
                       dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def create_comprehensive_report(self, classification_results: Dict,
                                  regression_results: Dict,
                                  save_name: str = \"svm_analysis_report\") -> None:
        \"\"\"
        Create a comprehensive visualization report
        
        Args:
            classification_results: Classification analysis results
            regression_results: Regression analysis results
            save_name: Base filename for saving
        \"\"\"
        # Create a multi-page report
        fig = plt.figure(figsize=(20, 15))
        
        # Page 1: Classification Performance
        plt.subplot(3, 3, 1)
        self._plot_metric_bars(classification_results, 'accuracy', 'Classification Accuracy')
        
        plt.subplot(3, 3, 2)
        self._plot_metric_bars(classification_results, 'f1', 'F1 Score')
        
        plt.subplot(3, 3, 3)
        self._plot_metric_bars(classification_results, 'precision', 'Precision')
        
        # Page 2: Regression Performance
        plt.subplot(3, 3, 4)
        self._plot_metric_bars(regression_results, 'rmse', 'RMSE', lower_is_better=True)
        
        plt.subplot(3, 3, 5)
        self._plot_metric_bars(regression_results, 'r2', 'R² Score')
        
        plt.subplot(3, 3, 6)
        self._plot_metric_bars(regression_results, 'mae', 'MAE', lower_is_better=True)
        
        # Summary statistics
        plt.subplot(3, 1, 3)
        self._create_summary_table(classification_results, regression_results)
        
        plt.suptitle('SVM Analysis Comprehensive Report', fontsize=20)
        plt.tight_layout()
        
        plt.savefig(self.save_dir / f'{save_name}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def _plot_metric_bars(self, results: Dict, metric: str, title: str, 
                         lower_is_better: bool = False) -> None:
        \"\"\"
        Helper function to plot metric bars
        \"\"\"
        models = []
        values = []
        
        for model_type, model_results in results.items():
            for model_name, metrics in model_results.items():
                if metric in metrics:
                    models.append(f\"{model_type}_{model_name}\")
                    values.append(metrics[metric])
        
        if values:
            colors = plt.cm.viridis(np.linspace(0, 1, len(values)))
            bars = plt.bar(range(len(models)), values, color=colors)
            
            # Highlight best performance
            if lower_is_better:
                best_idx = np.argmin(values)
            else:
                best_idx = np.argmax(values)
            
            bars[best_idx].set_color('gold')
            bars[best_idx].set_edgecolor('black')
            bars[best_idx].set_linewidth(2)
            
            plt.xticks(range(len(models)), models, rotation=45, ha='right')
            plt.ylabel(metric.upper())
            plt.title(title)
            plt.grid(True, alpha=0.3, axis='y')
    
    def _create_summary_table(self, classification_results: Dict, 
                            regression_results: Dict) -> None:
        \"\"\"
        Create summary statistics table
        \"\"\"
        plt.axis('off')
        
        # Prepare summary data
        summary_data = []
        
        # Classification summary
        for model_type, results in classification_results.items():
            for model_name, metrics in results.items():
                summary_data.append([
                    f\"{model_type}_{model_name}\",
                    \"Classification\",
                    f\"{metrics.get('accuracy', 0):.3f}\",
                    f\"{metrics.get('f1', 0):.3f}\",
                    \"-\",
                    \"-\"
                ])
        
        # Regression summary
        for model_type, results in regression_results.items():
            for model_name, metrics in results.items():
                summary_data.append([
                    f\"{model_type}_{model_name}\",
                    \"Regression\",
                    \"-\",
                    \"-\", 
                    f\"{metrics.get('rmse', 0):.3f}\",
                    f\"{metrics.get('r2', 0):.3f}\"
                ])
        
        # Create table
        headers = ['Model', 'Task', 'Accuracy', 'F1', 'RMSE', 'R²']
        
        table = plt.table(cellText=summary_data,
                         colLabels=headers,
                         cellLoc='center',
                         loc='center',
                         bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        plt.title('Model Performance Summary', pad=20)
    
    def save_all_plots(self) -> None:
        \"\"\"
        Save all generated plots to respective directories
        \"\"\"
        print(f\"All visualizations saved to: {self.save_dir}\")
        print(\"Directory structure:\")
        for subdir in self.save_dir.iterdir():
            if subdir.is_dir():
                files = list(subdir.glob(\"*.png\")) + list(subdir.glob(\"*.html\"))
                print(f\"  {subdir.name}/: {len(files)} files\")"

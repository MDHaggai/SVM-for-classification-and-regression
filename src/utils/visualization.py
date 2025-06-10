from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np

def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    """
    Plots the decision boundary of a model.
    
    Parameters:
    - model: The trained model (SVM).
    - X: Feature data (2D).
    - y: Target labels.
    - title: Title of the plot.
    """
    # Create a mesh grid for plotting
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))

    # Predict on the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o', s=50)
    plt.title(title)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.colorbar()
    plt.show()

def plot_performance_metrics(metrics, title="Model Performance Metrics"):
    """
    Plots performance metrics such as accuracy, precision, recall, and F1-score.
    
    Parameters:
    - metrics: Dictionary containing performance metrics.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(metrics.keys()), y=list(metrics.values()))
    plt.title(title)
    plt.ylabel('Score')
    plt.ylim(0, 1)
    plt.show()

def plot_learning_curve(train_sizes, train_scores, test_scores, title="Learning Curve"):
    """
    Plots the learning curve of a model.
    
    Parameters:
    - train_sizes: Array of training sizes.
    - train_scores: Array of training scores.
    - test_scores: Array of test scores.
    - title: Title of the plot.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, np.mean(train_scores, axis=1), label='Training Score', color='blue')
    plt.plot(train_sizes, np.mean(test_scores, axis=1), label='Cross-Validation Score', color='orange')
    plt.title(title)
    plt.xlabel('Training Size')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes=None, title="Confusion Matrix"):
    """
    Plots a confusion matrix.
    
    Parameters:
    - y_true: True labels.
    - y_pred: Predicted labels.
    - classes: List of class names.
    - title: Title of the plot.
    """
    from sklearn.metrics import confusion_matrix
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def plot_training_history(history, title="Training History"):
    """
    Plots training history if available (for iterative models).
    
    Parameters:
    - history: Training history data.
    - title: Title of the plot.
    """
    if history is None or len(history) == 0:
        print("No training history available to plot.")
        return
        
    plt.figure(figsize=(10, 6))
    if isinstance(history, dict):
        for metric, values in history.items():
            plt.plot(values, label=metric)
    elif isinstance(history, list):
        plt.plot(history, label='Loss/Score')
    
    plt.title(title)
    plt.xlabel('Iteration/Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.grid()
    plt.show()

def plot_residuals(y_true, y_pred, title="Residuals Plot"):
    """
    Plots residuals for regression analysis.
    
    Parameters:
    - y_true: True target values.
    - y_pred: Predicted target values.
    - title: Title of the plot.
    """
    residuals = y_true - y_pred
    plt.figure(figsize=(10, 6))
    plt.scatter(y_pred, residuals, alpha=0.6)
    plt.axhline(y=0, color='red', linestyle='--')
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals')
    plt.title(title)
    plt.grid(True)
    plt.show()
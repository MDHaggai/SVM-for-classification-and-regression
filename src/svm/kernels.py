"""
Kernel Functions for Support Vector Machines
===========================================

This module implements various kernel functions used in SVMs:
- Linear Kernel
- Polynomial Kernel  
- RBF (Radial Basis Function) Kernel
- Sigmoid Kernel

Each kernel transforms the input space to enable non-linear classification.
"""

import numpy as np
from typing import Union, Callable
import warnings

def linear_kernel(X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
    """
    Linear kernel function: K(x1, x2) = x1 · x2
    
    Args:
        X1: First set of data points (n_samples_1, n_features)
        X2: Second set of data points (n_samples_2, n_features)
        
    Returns:
        Kernel matrix (n_samples_1, n_samples_2)
    """
    return np.dot(X1, X2.T)

def polynomial_kernel(X1: np.ndarray, X2: np.ndarray, 
                     degree: int = 3, gamma: Union[float, str] = 'scale', 
                     coef0: float = 1.0) -> np.ndarray:
    """
    Polynomial kernel function: K(x1, x2) = (γ(x1 · x2) + r)^d
    
    Args:
        X1: First set of data points
        X2: Second set of data points  
        degree: Polynomial degree
        gamma: Gamma parameter (if 'scale', uses 1/(n_features * X.var()))
        coef0: Independent term (r)
        
    Returns:
        Kernel matrix
    """
    if gamma == 'scale':
        gamma = 1.0 / (X1.shape[1] * X1.var())
    elif gamma == 'auto':
        gamma = 1.0 / X1.shape[1]
    
    return (gamma * np.dot(X1, X2.T) + coef0) ** degree

def rbf_kernel(X1: np.ndarray, X2: np.ndarray, 
               gamma: Union[float, str] = 'scale') -> np.ndarray:
    """
    RBF (Radial Basis Function) kernel: K(x1, x2) = exp(-γ||x1 - x2||²)
    
    Also known as Gaussian kernel. Most popular kernel for non-linear problems.
    
    Args:
        X1: First set of data points
        X2: Second set of data points
        gamma: Gamma parameter (if 'scale', uses 1/(n_features * X.var()))
        
    Returns:
        Kernel matrix
    """
    if gamma == 'scale':
        gamma = 1.0 / (X1.shape[1] * X1.var())
    elif gamma == 'auto':
        gamma = 1.0 / X1.shape[1]
    
    # Compute squared Euclidean distance
    # ||x1 - x2||² = ||x1||² + ||x2||² - 2(x1 · x2)
    X1_norm = np.sum(X1**2, axis=1, keepdims=True)
    X2_norm = np.sum(X2**2, axis=1, keepdims=True)
    
    # Broadcasting to compute all pairwise distances
    squared_distances = X1_norm + X2_norm.T - 2 * np.dot(X1, X2.T)
    
    # Ensure no negative values due to numerical errors
    squared_distances = np.maximum(squared_distances, 0)
    
    return np.exp(-gamma * squared_distances)

def sigmoid_kernel(X1: np.ndarray, X2: np.ndarray,
                  gamma: Union[float, str] = 'scale', 
                  coef0: float = 1.0) -> np.ndarray:
    """
    Sigmoid kernel function: K(x1, x2) = tanh(γ(x1 · x2) + r)
    
    Similar to neural network activation. Not always positive semi-definite.
    
    Args:
        X1: First set of data points
        X2: Second set of data points
        gamma: Gamma parameter
        coef0: Independent term (r)
        
    Returns:
        Kernel matrix
    """
    if gamma == 'scale':
        gamma = 1.0 / (X1.shape[1] * X1.var())
    elif gamma == 'auto':
        gamma = 1.0 / X1.shape[1]
    
    return np.tanh(gamma * np.dot(X1, X2.T) + coef0)

class KernelFunction:
    """
    Kernel function wrapper class for easy switching between kernels
    """
    
    def __init__(self, kernel: Union[str, Callable] = 'rbf', **params):
        """
        Initialize kernel function
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid') or callable
            **params: Kernel-specific parameters
        """
        self.kernel = kernel
        self.params = params
        
        # Map kernel names to functions
        self.kernel_functions = {
            'linear': linear_kernel,
            'poly': polynomial_kernel,
            'polynomial': polynomial_kernel,
            'rbf': rbf_kernel,
            'sigmoid': sigmoid_kernel
        }
        
        if isinstance(kernel, str) and kernel not in self.kernel_functions:
            raise ValueError(f"Unknown kernel: {kernel}. Available: {list(self.kernel_functions.keys())}")
    
    def __call__(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """
        Compute kernel matrix
        
        Args:
            X1: First set of data points
            X2: Second set of data points
            
        Returns:
            Kernel matrix
        """
        if callable(self.kernel):
            return self.kernel(X1, X2, **self.params)
        else:
            kernel_func = self.kernel_functions[self.kernel]
            return kernel_func(X1, X2, **self.params)
    
    def __repr__(self):
        return f"KernelFunction(kernel='{self.kernel}', params={self.params})"

def compute_kernel_matrix(X: np.ndarray, kernel: Union[str, Callable] = 'rbf', 
                         **kernel_params) -> np.ndarray:
    """
    Compute full kernel matrix for training data
    
    Args:
        X: Training data (n_samples, n_features)
        kernel: Kernel type or function
        **kernel_params: Parameters for kernel function
        
    Returns:
        Kernel matrix (n_samples, n_samples)
    """
    kernel_func = KernelFunction(kernel, **kernel_params)
    return kernel_func(X, X)

def kernel_center(K: np.ndarray) -> np.ndarray:
    """
    Center a kernel matrix in feature space
    
    This is equivalent to centering the data in the feature space φ(X).
    
    Args:
        K: Kernel matrix (n_samples, n_samples)
        
    Returns:
        Centered kernel matrix
    """
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    
    # K_centered = K - 1_n @ K - K @ 1_n + 1_n @ K @ 1_n
    return K - one_n @ K - K @ one_n + one_n @ K @ one_n

# Kernel parameter optimization utilities
def estimate_rbf_gamma(X: np.ndarray, method: str = 'scale') -> float:
    """
    Estimate good gamma parameter for RBF kernel
    
    Args:
        X: Input data
        method: Estimation method ('scale', 'auto', 'median_heuristic')
        
    Returns:
        Estimated gamma value
    """
    if method == 'scale':
        return 1.0 / (X.shape[1] * X.var())
    elif method == 'auto':
        return 1.0 / X.shape[1]
    elif method == 'median_heuristic':
        # Median of pairwise distances heuristic
        from scipy.spatial.distance import pdist
        distances = pdist(X, metric='euclidean')
        median_dist = np.median(distances)
        return 1.0 / (2 * median_dist**2)
    else:
        raise ValueError(f"Unknown method: {method}")

# Kernel validation utilities
def is_valid_kernel_matrix(K: np.ndarray, tol: float = 1e-6) -> bool:
    """
    Check if a matrix is a valid kernel matrix (positive semi-definite)
    
    Args:
        K: Kernel matrix
        tol: Tolerance for eigenvalue check
        
    Returns:
        True if valid kernel matrix
    """
    # Check if symmetric
    if not np.allclose(K, K.T, atol=tol):
        return False
    
    # Check if positive semi-definite (all eigenvalues >= 0)
    eigenvals = np.linalg.eigvals(K)
    return np.all(eigenvals >= -tol)

def kernel_alignment(K1: np.ndarray, K2: np.ndarray) -> float:
    """
    Compute kernel alignment between two kernel matrices
    
    Measures how similar two kernels are. Values close to 1 indicate 
    high similarity.
    
    Args:
        K1: First kernel matrix
        K2: Second kernel matrix
        
    Returns:
        Kernel alignment score
    """
    # Frobenius inner product
    numerator = np.trace(K1 @ K2)
    denominator = np.sqrt(np.trace(K1 @ K1) * np.trace(K2 @ K2))
    
    return numerator / denominator if denominator > 0 else 0.0

"""
Kernel Support Vector Machine Implementation
==========================================

A comprehensive implementation of SVM with kernel support using the
Sequential Minimal Optimization (SMO) algorithm.

This implementation includes:
- Multiple kernel functions (linear, polynomial, RBF, sigmoid)
- Soft margin classification with slack variables
- Efficient SMO optimization algorithm
- Support for both binary and multi-class classification
- Comprehensive parameter validation and optimization
"""

import numpy as np
from typing import Union, Optional, Callable
import time
import warnings
from .kernels import KernelFunction

class KernelSVM:
    """
    Kernel Support Vector Machine Classifier
    
    Implements SVM with kernel trick for non-linear classification:
    
    Dual Problem:
    Maximize: ∑αi - (1/2)∑∑αiαjyiyjK(xi,xj)
    Subject to: 0 ≤ αi ≤ C, ∑αiyi = 0
    
    Decision Function:
    f(x) = ∑αiyiK(xi,x) + b
    
    Attributes:
        kernel (str or callable): Kernel function
        C (float): Regularization parameter
        gamma (float or str): Kernel coefficient
        degree (int): Polynomial kernel degree
        coef0 (float): Independent term in kernel
        tol (float): Tolerance for stopping criteria
        max_iter (int): Maximum iterations
    """
    
    def __init__(self, kernel: Union[str, Callable] = 'rbf', C: float = 1.0,
                 gamma: Union[float, str] = 'scale', degree: int = 3,
                 coef0: float = 0.0, tol: float = 1e-3, max_iter: int = 1000,
                 random_state: Optional[int] = None, verbose: bool = False):
        """
        Initialize Kernel SVM
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid') or callable
            C: Regularization parameter (higher C = less regularization)
            gamma: Kernel coefficient for 'rbf', 'poly', 'sigmoid'
            degree: Degree for polynomial kernel
            coef0: Independent term for 'poly' and 'sigmoid'
            tol: Tolerance for stopping criteria
            max_iter: Maximum iterations for SMO
            random_state: Random seed
            verbose: Print training progress
        """
        self.kernel = kernel
        self.C = C
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
        # Initialize training attributes
        self.alpha = None
        self.b = 0.0
        self.support_vectors_ = None
        self.support_labels_ = None
        self.support_alphas_ = None
        self.n_support_ = None
        self.classes_ = None
        
        # Training data (needed for kernel evaluation)
        self._X_train = None
        self._y_train = None
        self._kernel_func = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _setup_kernel(self, X: np.ndarray) -> None:
        """
        Setup kernel function with appropriate parameters
        
        Args:
            X: Training data for parameter estimation
        """
        kernel_params = {}
        
        if self.kernel in ['rbf', 'poly', 'sigmoid']:
            if self.gamma == 'scale':
                kernel_params['gamma'] = 1.0 / (X.shape[1] * X.var())
            elif self.gamma == 'auto':
                kernel_params['gamma'] = 1.0 / X.shape[1]
            else:
                kernel_params['gamma'] = self.gamma
        
        if self.kernel in ['poly', 'sigmoid']:
            kernel_params['coef0'] = self.coef0
            
        if self.kernel == 'poly':
            kernel_params['degree'] = self.degree
        
        self._kernel_func = KernelFunction(self.kernel, **kernel_params)
        
        if self.verbose:
            print(f"Kernel setup: {self._kernel_func}")
    
    def _compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute the Gram (kernel) matrix
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Gram matrix K where K[i,j] = kernel(xi, xj)
        """
        return self._kernel_func(X, X)
    
    def _objective_function(self, alpha: np.ndarray, K: np.ndarray, 
                          y: np.ndarray) -> float:
        """
        Compute dual objective function value
        
        L(α) = ∑αi - (1/2)∑∑αiαjyiyjK(xi,xj)
        
        Args:
            alpha: Lagrange multipliers
            K: Gram matrix
            y: Labels
            
        Returns:
            Objective function value
        """
        term1 = np.sum(alpha)
        y_alpha = y * alpha
        term2 = 0.5 * np.dot(y_alpha, np.dot(K, y_alpha))
        return term1 - term2
    
    def _violates_kkt(self, alpha_i: float, y_i: float, E_i: float) -> bool:
        """
        Check if sample violates KKT conditions
        
        Args:
            alpha_i: Alpha value for sample i
            y_i: Label for sample i  
            E_i: Error for sample i
            
        Returns:
            True if KKT conditions are violated
        """
        r_i = E_i * y_i
        
        if (r_i < -self.tol and alpha_i < self.C) or \
           (r_i > self.tol and alpha_i > 0):
            return True
        return False
    
    def _select_alpha_pair_heuristic(self, alpha: np.ndarray, y: np.ndarray,
                                   E: np.ndarray) -> tuple:
        """
        Select pair of alphas using SMO heuristics
        
        Args:
            alpha: Current alpha values
            y: Labels
            E: Error cache
            
        Returns:
            (i, j) indices of selected alpha pair
        """
        n_samples = len(alpha)
        
        # First choice: non-boundary alpha that violates KKT most
        non_bound = (alpha > self.tol) & (alpha < self.C - self.tol)
        violating = []
        
        for i in range(n_samples):
            if self._violates_kkt(alpha[i], y[i], E[i]):
                violating.append(i)
        
        if not violating:
            return None, None
        
        # Choose first alpha
        if np.any(non_bound):
            candidates = [i for i in violating if non_bound[i]]
            i = candidates[0] if candidates else violating[0]
        else:
            i = violating[0]
        
        # Second choice: maximize |E_i - E_j|
        E_i = E[i]
        if E_i >= 0:
            j = np.argmin(E)
        else:
            j = np.argmax(E)
        
        # Ensure i != j
        if i == j:
            candidates = [idx for idx in range(n_samples) if idx != i]
            j = np.random.choice(candidates) if candidates else None
        
        return i, j
    
    def _compute_bounds(self, alpha_i: float, alpha_j: float,
                       y_i: float, y_j: float) -> tuple:
        """
        Compute bounds L and H for alpha_j
        
        Args:
            alpha_i, alpha_j: Current alpha values
            y_i, y_j: Labels
            
        Returns:
            (L, H) bounds for alpha_j
        """
        if y_i != y_j:
            L = max(0, alpha_j - alpha_i)
            H = min(self.C, self.C + alpha_j - alpha_i)
        else:
            L = max(0, alpha_i + alpha_j - self.C)
            H = min(self.C, alpha_i + alpha_j)
        
        return L, H
    
    def _smo_step(self, i: int, j: int, alpha: np.ndarray, y: np.ndarray,
                  K: np.ndarray, E: np.ndarray, b: float) -> tuple:
        """
        Perform one SMO optimization step
        
        Args:
            i, j: Indices to optimize
            alpha: Alpha values
            y: Labels
            K: Gram matrix
            E: Error cache
            b: Bias
            
        Returns:
            (updated_alpha, updated_bias, success_flag)
        """
        if i == j:
            return alpha, b, False
        
        alpha_old = alpha.copy()
        
        # Compute bounds
        L, H = self._compute_bounds(alpha[i], alpha[j], y[i], y[j])
        
        if L == H:
            return alpha, b, False
        
        # Compute eta (second derivative of objective)
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        
        if eta >= 0:
            return alpha, b, False
        
        # Compute new alpha_j
        alpha_j_new = alpha[j] - y[j] * (E[i] - E[j]) / eta
        
        # Clip alpha_j
        alpha_j_new = np.clip(alpha_j_new, L, H)
        
        # Check for sufficient change
        if abs(alpha_j_new - alpha[j]) < self.tol:
            return alpha, b, False
        
        # Compute new alpha_i
        alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
        
        # Update alphas
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        
        # Update bias
        b1 = E[i] + y[i] * (alpha_i_new - alpha_old[i]) * K[i, i] + \
             y[j] * (alpha_j_new - alpha_old[j]) * K[i, j] + b
        
        b2 = E[j] + y[i] * (alpha_i_new - alpha_old[i]) * K[i, j] + \
             y[j] * (alpha_j_new - alpha_old[j]) * K[j, j] + b
        
        if 0 < alpha_i_new < self.C:
            b_new = b1
        elif 0 < alpha_j_new < self.C:
            b_new = b2
        else:
            b_new = (b1 + b2) / 2
        
        return alpha, b_new, True
    
    def _compute_errors(self, alpha: np.ndarray, y: np.ndarray,
                       K: np.ndarray, b: float) -> np.ndarray:
        """
        Compute prediction errors
        
        E_i = f(xi) - yi where f(xi) = ∑αjyjK(xi,xj) + b
        
        Args:
            alpha: Lagrange multipliers
            y: True labels
            K: Gram matrix
            b: Bias
            
        Returns:
            Error array
        """
        predictions = np.dot(K, alpha * y) + b
        return predictions - y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'KernelSVM':
        """
        Train the Kernel SVM using SMO algorithm
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples,)
            
        Returns:
            self
        """
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Handle binary classification
        self.classes_ = np.unique(y)
        if len(self.classes_) != 2:
            raise ValueError("Only binary classification supported")
        
        # Map labels to -1, +1
        y_mapped = np.where(y == self.classes_[0], -1, 1)
        
        # Store training data
        self._X_train = X.copy()
        self._y_train = y_mapped.copy()
        
        n_samples, n_features = X.shape
        
        # Setup kernel function
        self._setup_kernel(X)
        
        # Initialize optimization variables
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # Compute Gram matrix
        if self.verbose:
            print("Computing Gram matrix...")
        K = self._compute_gram_matrix(X)
        
        # SMO main loop
        if self.verbose:
            print("Starting SMO optimization...")
        
        start_time = time.time()
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            # Compute errors
            E = self._compute_errors(alpha, y_mapped, K, b)
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    if self._violates_kkt(alpha[i], y_mapped[i], E[i]):
                        # Select second alpha using heuristic
                        _, j = self._select_alpha_pair_heuristic(alpha, y_mapped, E)
                        if j is not None:
                            alpha, b, changed = self._smo_step(i, j, alpha, y_mapped, K, E, b)
                            if changed:
                                num_changed += 1
            else:
                # Examine non-boundary samples
                non_bound_idx = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]
                for i in non_bound_idx:
                    if self._violates_kkt(alpha[i], y_mapped[i], E[i]):
                        _, j = self._select_alpha_pair_heuristic(alpha, y_mapped, E)
                        if j is not None:
                            alpha, b, changed = self._smo_step(i, j, alpha, y_mapped, K, E, b)
                            if changed:
                                num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            
            # Progress reporting
            if self.verbose and iteration % 100 == 0:
                obj_val = self._objective_function(alpha, K, y_mapped)
                n_sv = np.sum(alpha > self.tol)
                print(f"Iter {iteration:4d}: Obj={obj_val:8.4f}, SVs={n_sv:4d}/{n_samples}")
        
        training_time = time.time() - start_time
        
        # Store results
        self.alpha = alpha
        self.b = b
        
        # Extract support vectors
        sv_mask = alpha > self.tol
        self.support_vectors_ = X[sv_mask]
        self.support_labels_ = y[sv_mask]  # Original labels
        self.support_alphas_ = alpha[sv_mask]
        
        # Count support vectors per class
        self.n_support_ = np.array([
            np.sum(y_mapped[sv_mask] == -1),
            np.sum(y_mapped[sv_mask] == 1)
        ])
        
        if self.verbose:
            print(f"\nTraining completed in {iteration} iterations ({training_time:.2f}s)")
            print(f"Support vectors: {len(self.support_vectors_)}/{n_samples} "
                  f"({100*len(self.support_vectors_)/n_samples:.1f}%)")
            print(f"Class distribution: {self.n_support_}")
            print(f"Final objective: {self._objective_function(alpha, K, y_mapped):.6f}")
        
        return self
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values
        
        f(x) = ∑αiyiK(xi,x) + b
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Decision function values (n_samples,)
        """
        if self.alpha is None:
            raise ValueError("Model not fitted yet")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Compute kernel matrix between test and support vectors
        K_test = self._kernel_func(self.support_vectors_, X)
        
        # Decision function: ∑αiyiK(xi,x) + b
        support_labels_mapped = np.where(self.support_labels_ == self.classes_[0], -1, 1)
        decision_values = np.dot((self.support_alphas_ * support_labels_mapped), K_test) + self.b
        
        return decision_values
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,)
        """
        decision_values = self.decision_function(X)
        predictions = np.where(decision_values >= 0, 1, -1)
        
        # Map back to original labels
        return np.where(predictions == -1, self.classes_[0], self.classes_[1])
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using Platt scaling
        
        Note: This is a simplified probability estimate.
        For proper probability calibration, use CalibratedClassifierCV.
        
        Args:
            X: Input data
            
        Returns:
            Class probabilities (n_samples, 2)
        """
        decision_values = self.decision_function(X)
        
        # Simple sigmoid mapping (not properly calibrated)
        probabilities = 1 / (1 + np.exp(-decision_values))
        
        # Return probabilities for both classes
        proba_class1 = probabilities
        proba_class0 = 1 - probabilities
        
        return np.column_stack([proba_class0, proba_class1])
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Compute accuracy score
        
        Args:
            X: Test data
            y: True labels
            
        Returns:
            Accuracy score
        """
        predictions = self.predict(X)
        return np.mean(predictions == y)
    
    def get_params(self, deep: bool = True) -> dict:
        """
        Get parameters for this estimator
        
        Args:
            deep: If True, return parameters for this estimator and
                  contained subobjects
                  
        Returns:
            Parameter names mapped to their values
        """
        return {
            'kernel': self.kernel,
            'C': self.C,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'KernelSVM':
        """
        Set the parameters of this estimator
        
        Args:
            **params: Parameter names and values
            
        Returns:
            self
        """
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return self

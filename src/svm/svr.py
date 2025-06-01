"""
Support Vector Regression (SVR) Implementation
=============================================

Implementation of Support Vector Regression using the ε-insensitive loss function
and kernel trick for non-linear regression problems.

SVR solves the following optimization problem:

Minimize: (1/2)||w||² + C∑(ξi + ξi*)

Subject to:
- yi - w·φ(xi) - b ≤ ε + ξi
- w·φ(xi) + b - yi ≤ ε + ξi*
- ξi, ξi* ≥ 0

The dual formulation becomes:
Maximize: -ε∑(αi + αi*) + ∑yi(αi - αi*) - (1/2)∑∑(αi - αi*)(αj - αj*)K(xi,xj)

Subject to:
- 0 ≤ αi, αi* ≤ C
- ∑(αi - αi*) = 0
"""

import numpy as np
from typing import Union, Optional, Callable
import time
import warnings
from .kernels import KernelFunction

class SupportVectorRegressor:
    """
    Support Vector Regression with kernel support
    
    Attributes:
        kernel (str or callable): Kernel function
        C (float): Regularization parameter
        epsilon (float): Epsilon parameter for ε-insensitive loss
        gamma (float or str): Kernel coefficient
        degree (int): Polynomial kernel degree
        coef0 (float): Independent term in kernel
        tol (float): Tolerance for stopping criteria
        max_iter (int): Maximum iterations
    """
    
    def __init__(self, kernel: Union[str, Callable] = 'rbf', C: float = 1.0,
                 epsilon: float = 0.1, gamma: Union[float, str] = 'scale',
                 degree: int = 3, coef0: float = 0.0, tol: float = 1e-3,
                 max_iter: int = 1000, random_state: Optional[int] = None,
                 verbose: bool = False):
        """
        Initialize Support Vector Regressor
        
        Args:
            kernel: Kernel type ('linear', 'poly', 'rbf', 'sigmoid') or callable
            C: Regularization parameter
            epsilon: Epsilon parameter for ε-insensitive loss
            gamma: Kernel coefficient
            degree: Degree for polynomial kernel
            coef0: Independent term for 'poly' and 'sigmoid'
            tol: Tolerance for stopping criteria
            max_iter: Maximum iterations for optimization
            random_state: Random seed
            verbose: Print training progress
        """
        self.kernel = kernel
        self.C = C
        self.epsilon = epsilon
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        self.verbose = verbose
        
        # Training attributes
        self.alpha = None
        self.alpha_star = None
        self.b = 0.0
        self.support_vectors_ = None
        self.support_targets_ = None
        self.support_alphas_ = None
        self.support_alphas_star_ = None
        
        # Training data
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
    
    def _objective_function(self, alpha: np.ndarray, alpha_star: np.ndarray,
                          K: np.ndarray, y: np.ndarray) -> float:
        """
        Compute SVR dual objective function value
        
        L(α,α*) = -ε∑(αi + αi*) + ∑yi(αi - αi*) - (1/2)∑∑(αi - αi*)(αj - αj*)K(xi,xj)
        
        Args:
            alpha: Alpha multipliers
            alpha_star: Alpha* multipliers
            K: Gram matrix
            y: Target values
            
        Returns:
            Objective function value
        """
        alpha_diff = alpha - alpha_star
        
        term1 = -self.epsilon * np.sum(alpha + alpha_star)
        term2 = np.dot(y, alpha_diff)
        term3 = -0.5 * np.dot(alpha_diff, np.dot(K, alpha_diff))
        
        return term1 + term2 + term3
    
    def _violates_kkt_regression(self, alpha_i: float, alpha_star_i: float,
                               error_i: float) -> bool:
        """
        Check if sample violates KKT conditions for regression
        
        Args:
            alpha_i: Alpha value for sample i
            alpha_star_i: Alpha* value for sample i
            error_i: Prediction error for sample i
            
        Returns:
            True if KKT conditions are violated
        """
        # Check optimality conditions
        if alpha_i > self.tol and error_i < -self.epsilon - self.tol:
            return True
        if alpha_star_i > self.tol and error_i > self.epsilon + self.tol:
            return True
        if alpha_i < self.C - self.tol and error_i > -self.epsilon + self.tol:
            return True
        if alpha_star_i < self.C - self.tol and error_i < self.epsilon - self.tol:
            return True
        
        return False
    
    def _select_working_set(self, alpha: np.ndarray, alpha_star: np.ndarray,
                          errors: np.ndarray) -> tuple:
        """
        Select working set for optimization (simplified version)
        
        Args:
            alpha: Alpha values
            alpha_star: Alpha* values  
            errors: Prediction errors
            
        Returns:
            (i, j) indices of selected pair
        """
        n_samples = len(alpha)
        
        # Find violating samples
        violating = []
        for i in range(n_samples):
            if self._violates_kkt_regression(alpha[i], alpha_star[i], errors[i]):
                violating.append(i)
        
        if len(violating) < 2:
            return None, None
        
        # Simple selection: first two violating samples
        return violating[0], violating[1] if len(violating) > 1 else violating[0]
    
    def _smo_step_regression(self, i: int, j: int, alpha: np.ndarray,
                           alpha_star: np.ndarray, y: np.ndarray,
                           K: np.ndarray, errors: np.ndarray,
                           b: float) -> tuple:
        """
        Perform one SMO step for regression
        
        Args:
            i, j: Indices to optimize
            alpha: Alpha values
            alpha_star: Alpha* values
            y: Target values
            K: Gram matrix
            errors: Prediction errors
            b: Bias
            
        Returns:
            (updated_alpha, updated_alpha_star, updated_bias, success_flag)
        """
        if i == j:
            return alpha, alpha_star, b, False
        
        alpha_old = alpha.copy()
        alpha_star_old = alpha_star.copy()
        
        # Compute eta
        eta = K[i, i] + K[j, j] - 2 * K[i, j]
        
        if eta <= 0:
            return alpha, alpha_star, b, False
        
        # Compute bounds (simplified)
        L_i = max(0, alpha[i] - self.C)
        H_i = min(self.C, alpha[i])
        L_star_i = max(0, alpha_star[i] - self.C)
        H_star_i = min(self.C, alpha_star[i])
        
        # Update alpha_i (simplified update rule)
        delta_alpha_i = (errors[i] - errors[j]) / eta
        alpha[i] = np.clip(alpha[i] + delta_alpha_i, L_i, H_i)
        
        # Update alpha_star_i based on complementarity
        if errors[i] > self.epsilon:
            alpha_star[i] = 0
        elif errors[i] < -self.epsilon:
            alpha[i] = 0
        
        # Check for sufficient change
        if (abs(alpha[i] - alpha_old[i]) < self.tol and 
            abs(alpha_star[i] - alpha_star_old[i]) < self.tol):
            return alpha, alpha_star, b, False
        
        # Update bias (simplified)
        support_mask = (alpha > self.tol) | (alpha_star > self.tol)
        if np.any(support_mask):
            # Use support vectors to estimate bias
            predictions = np.dot(K, alpha - alpha_star)
            residuals = y - predictions
            b = np.mean(residuals[support_mask])
        
        return alpha, alpha_star, b, True
    
    def _compute_errors_regression(self, alpha: np.ndarray, alpha_star: np.ndarray,
                                 y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:
        \"\"\"
        Compute prediction errors for regression
        
        Args:
            alpha: Alpha multipliers
            alpha_star: Alpha* multipliers
            y: True targets
            K: Gram matrix
            b: Bias
            
        Returns:
            Error array
        \"\"\"
        predictions = np.dot(K, alpha - alpha_star) + b
        return predictions - y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorRegressor':
        \"\"\"
        Train the Support Vector Regressor
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training targets (n_samples,)
            
        Returns:
            self
        \"\"\"
        # Input validation
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError(\"X must be 2D array\")
        if y.ndim != 1:
            raise ValueError(\"y must be 1D array\")
        if X.shape[0] != y.shape[0]:
            raise ValueError(\"X and y must have same number of samples\")
        
        # Store training data
        self._X_train = X.copy()
        self._y_train = y.copy()
        
        n_samples, n_features = X.shape
        
        # Setup kernel function
        self._setup_kernel(X)
        
        # Initialize optimization variables
        alpha = np.zeros(n_samples)
        alpha_star = np.zeros(n_samples)
        b = 0.0
        
        # Compute Gram matrix
        if self.verbose:
            print(\"Computing Gram matrix for SVR...\")
        K = self._compute_gram_matrix(X)
        
        # SMO main loop for regression
        if self.verbose:
            print(\"Starting SMO optimization for SVR...\")
        
        start_time = time.time()
        num_changed = 0
        examine_all = True
        iteration = 0
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            # Compute errors
            errors = self._compute_errors_regression(alpha, alpha_star, y, K, b)
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    if self._violates_kkt_regression(alpha[i], alpha_star[i], errors[i]):
                        # Simple second choice
                        j = (i + 1) % n_samples
                        alpha, alpha_star, b, changed = self._smo_step_regression(
                            i, j, alpha, alpha_star, y, K, errors, b)
                        if changed:
                            num_changed += 1
            else:
                # Examine support vectors
                sv_mask = (alpha > self.tol) | (alpha_star > self.tol)
                for i in np.where(sv_mask)[0]:
                    if self._violates_kkt_regression(alpha[i], alpha_star[i], errors[i]):
                        j = (i + 1) % n_samples
                        alpha, alpha_star, b, changed = self._smo_step_regression(
                            i, j, alpha, alpha_star, y, K, errors, b)
                        if changed:
                            num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            
            # Progress reporting
            if self.verbose and iteration % 100 == 0:
                obj_val = self._objective_function(alpha, alpha_star, K, y)
                n_sv = np.sum((alpha > self.tol) | (alpha_star > self.tol))
                print(f\"Iter {iteration:4d}: Obj={obj_val:8.4f}, SVs={n_sv:4d}/{n_samples}\")
        
        training_time = time.time() - start_time
        
        # Store results
        self.alpha = alpha
        self.alpha_star = alpha_star
        self.b = b
        
        # Extract support vectors
        sv_mask = (alpha > self.tol) | (alpha_star > self.tol)
        self.support_vectors_ = X[sv_mask]
        self.support_targets_ = y[sv_mask]
        self.support_alphas_ = alpha[sv_mask]
        self.support_alphas_star_ = alpha_star[sv_mask]
        
        if self.verbose:
            print(f\"\\nSVR training completed in {iteration} iterations ({training_time:.2f}s)\")
            print(f\"Support vectors: {len(self.support_vectors_)}/{n_samples} \"
                  f\"({100*len(self.support_vectors_)/n_samples:.1f}%)\")
            print(f\"Final objective: {self._objective_function(alpha, alpha_star, K, y):.6f}\")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        \"\"\"
        Predict using the trained SVR
        
        f(x) = ∑(αi - αi*)K(xi,x) + b
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Predicted values (n_samples,)
        \"\"\"
        if self.alpha is None:
            raise ValueError(\"Model not fitted yet\")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Compute kernel matrix between test and support vectors
        K_test = self._kernel_func(self.support_vectors_, X)
        
        # Prediction: ∑(αi - αi*)K(xi,x) + b
        alpha_diff = self.support_alphas_ - self.support_alphas_star_
        predictions = np.dot(alpha_diff, K_test) + self.b
        
        return predictions
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        \"\"\"
        Compute coefficient of determination R² score
        
        Args:
            X: Test data
            y: True targets
            
        Returns:
            R² score
        \"\"\"
        predictions = self.predict(X)
        
        # R² = 1 - SS_res / SS_tot
        ss_res = np.sum((y - predictions) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    def get_params(self, deep: bool = True) -> dict:
        \"\"\"
        Get parameters for this estimator
        
        Returns:
            Parameter dictionary
        \"\"\"
        return {
            'kernel': self.kernel,
            'C': self.C,
            'epsilon': self.epsilon,
            'gamma': self.gamma,
            'degree': self.degree,
            'coef0': self.coef0,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'verbose': self.verbose
        }
    
    def set_params(self, **params) -> 'SupportVectorRegressor':
        \"\"\"
        Set the parameters of this estimator
        
        Returns:
            self
        \"\"\"
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f\"Invalid parameter: {param}\")
        return self"

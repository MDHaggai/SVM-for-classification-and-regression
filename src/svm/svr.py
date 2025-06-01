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
        
        # Update alpha_i (simplified update rule)\n        delta_alpha_i = (errors[i] - errors[j]) / eta\n        alpha[i] = np.clip(alpha[i] + delta_alpha_i, L_i, H_i)\n        \n        # Update alpha_star_i based on complementarity\n        if errors[i] > self.epsilon:\n            alpha_star[i] = 0\n        elif errors[i] < -self.epsilon:\n            alpha[i] = 0\n        \n        # Check for sufficient change\n        if (abs(alpha[i] - alpha_old[i]) < self.tol and \n            abs(alpha_star[i] - alpha_star_old[i]) < self.tol):\n            return alpha, alpha_star, b, False\n        \n        # Update bias (simplified)\n        support_mask = (alpha > self.tol) | (alpha_star > self.tol)\n        if np.any(support_mask):\n            # Use support vectors to estimate bias\n            predictions = np.dot(K, alpha - alpha_star)\n            residuals = y - predictions\n            b = np.mean(residuals[support_mask])\n        \n        return alpha, alpha_star, b, True\n    \n    def _compute_errors_regression(self, alpha: np.ndarray, alpha_star: np.ndarray,\n                                 y: np.ndarray, K: np.ndarray, b: float) -> np.ndarray:\n        \"\"\"\n        Compute prediction errors for regression\n        \n        Args:\n            alpha: Alpha multipliers\n            alpha_star: Alpha* multipliers\n            y: True targets\n            K: Gram matrix\n            b: Bias\n            \n        Returns:\n            Error array\n        \"\"\"\n        predictions = np.dot(K, alpha - alpha_star) + b\n        return predictions - y\n    \n    def fit(self, X: np.ndarray, y: np.ndarray) -> 'SupportVectorRegressor':\n        \"\"\"\n        Train the Support Vector Regressor\n        \n        Args:\n            X: Training data (n_samples, n_features)\n            y: Training targets (n_samples,)\n            \n        Returns:\n            self\n        \"\"\"\n        # Input validation\n        X = np.asarray(X, dtype=np.float64)\n        y = np.asarray(y, dtype=np.float64)\n        \n        if X.ndim != 2:\n            raise ValueError(\"X must be 2D array\")\n        if y.ndim != 1:\n            raise ValueError(\"y must be 1D array\")\n        if X.shape[0] != y.shape[0]:\n            raise ValueError(\"X and y must have same number of samples\")\n        \n        # Store training data\n        self._X_train = X.copy()\n        self._y_train = y.copy()\n        \n        n_samples, n_features = X.shape\n        \n        # Setup kernel function\n        self._setup_kernel(X)\n        \n        # Initialize optimization variables\n        alpha = np.zeros(n_samples)\n        alpha_star = np.zeros(n_samples)\n        b = 0.0\n        \n        # Compute Gram matrix\n        if self.verbose:\n            print(\"Computing Gram matrix for SVR...\")\n        K = self._compute_gram_matrix(X)\n        \n        # SMO main loop for regression\n        if self.verbose:\n            print(\"Starting SMO optimization for SVR...\")\n        \n        start_time = time.time()\n        num_changed = 0\n        examine_all = True\n        iteration = 0\n        \n        while (num_changed > 0 or examine_all) and iteration < self.max_iter:\n            num_changed = 0\n            \n            # Compute errors\n            errors = self._compute_errors_regression(alpha, alpha_star, y, K, b)\n            \n            if examine_all:\n                # Examine all samples\n                for i in range(n_samples):\n                    if self._violates_kkt_regression(alpha[i], alpha_star[i], errors[i]):\n                        # Simple second choice\n                        j = (i + 1) % n_samples\n                        alpha, alpha_star, b, changed = self._smo_step_regression(\n                            i, j, alpha, alpha_star, y, K, errors, b)\n                        if changed:\n                            num_changed += 1\n            else:\n                # Examine support vectors\n                sv_mask = (alpha > self.tol) | (alpha_star > self.tol)\n                for i in np.where(sv_mask)[0]:\n                    if self._violates_kkt_regression(alpha[i], alpha_star[i], errors[i]):\n                        j = (i + 1) % n_samples\n                        alpha, alpha_star, b, changed = self._smo_step_regression(\n                            i, j, alpha, alpha_star, y, K, errors, b)\n                        if changed:\n                            num_changed += 1\n            \n            if examine_all:\n                examine_all = False\n            elif num_changed == 0:\n                examine_all = True\n            \n            iteration += 1\n            \n            # Progress reporting\n            if self.verbose and iteration % 100 == 0:\n                obj_val = self._objective_function(alpha, alpha_star, K, y)\n                n_sv = np.sum((alpha > self.tol) | (alpha_star > self.tol))\n                print(f\"Iter {iteration:4d}: Obj={obj_val:8.4f}, SVs={n_sv:4d}/{n_samples}\")\n        \n        training_time = time.time() - start_time\n        \n        # Store results\n        self.alpha = alpha\n        self.alpha_star = alpha_star\n        self.b = b\n        \n        # Extract support vectors\n        sv_mask = (alpha > self.tol) | (alpha_star > self.tol)\n        self.support_vectors_ = X[sv_mask]\n        self.support_targets_ = y[sv_mask]\n        self.support_alphas_ = alpha[sv_mask]\n        self.support_alphas_star_ = alpha_star[sv_mask]\n        \n        if self.verbose:\n            print(f\"\\nSVR training completed in {iteration} iterations ({training_time:.2f}s)\")\n            print(f\"Support vectors: {len(self.support_vectors_)}/{n_samples} \"\n                  f\"({100*len(self.support_vectors_)/n_samples:.1f}%)\")\n            print(f\"Final objective: {self._objective_function(alpha, alpha_star, K, y):.6f}\")\n        \n        return self\n    \n    def predict(self, X: np.ndarray) -> np.ndarray:\n        \"\"\"\n        Predict using the trained SVR\n        \n        f(x) = ∑(αi - αi*)K(xi,x) + b\n        \n        Args:\n            X: Input data (n_samples, n_features)\n            \n        Returns:\n            Predicted values (n_samples,)\n        \"\"\"\n        if self.alpha is None:\n            raise ValueError(\"Model not fitted yet\")\n        \n        X = np.asarray(X, dtype=np.float64)\n        \n        # Compute kernel matrix between test and support vectors\n        K_test = self._kernel_func(self.support_vectors_, X)\n        \n        # Prediction: ∑(αi - αi*)K(xi,x) + b\n        alpha_diff = self.support_alphas_ - self.support_alphas_star_\n        predictions = np.dot(alpha_diff, K_test) + self.b\n        \n        return predictions\n    \n    def score(self, X: np.ndarray, y: np.ndarray) -> float:\n        \"\"\"\n        Compute coefficient of determination R² score\n        \n        Args:\n            X: Test data\n            y: True targets\n            \n        Returns:\n            R² score\n        \"\"\"\n        predictions = self.predict(X)\n        \n        # R² = 1 - SS_res / SS_tot\n        ss_res = np.sum((y - predictions) ** 2)\n        ss_tot = np.sum((y - np.mean(y)) ** 2)\n        \n        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0\n    \n    def get_params(self, deep: bool = True) -> dict:\n        \"\"\"\n        Get parameters for this estimator\n        \n        Returns:\n            Parameter dictionary\n        \"\"\"\n        return {\n            'kernel': self.kernel,\n            'C': self.C,\n            'epsilon': self.epsilon,\n            'gamma': self.gamma,\n            'degree': self.degree,\n            'coef0': self.coef0,\n            'tol': self.tol,\n            'max_iter': self.max_iter,\n            'random_state': self.random_state,\n            'verbose': self.verbose\n        }\n    \n    def set_params(self, **params) -> 'SupportVectorRegressor':\n        \"\"\"\n        Set the parameters of this estimator\n        \n        Returns:\n            self\n        \"\"\"\n        for param, value in params.items():\n            if hasattr(self, param):\n                setattr(self, param, value)\n            else:\n                raise ValueError(f\"Invalid parameter: {param}\")\n        return self"

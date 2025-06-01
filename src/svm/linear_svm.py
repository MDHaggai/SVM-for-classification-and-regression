"""
Linear Support Vector Machine Implementation
==========================================

A from-scratch implementation of linear SVM using the dual formulation
and Sequential Minimal Optimization (SMO) algorithm.

This implementation focuses on linear separable data and includes:
- Dual optimization with SMO
- Soft margin support with slack variables
- Efficient sparse computation
- Detailed mathematical documentation
"""

import numpy as np
from typing import Optional, Tuple
import warnings
from scipy import sparse
import time

class LinearSVM:
    """
    Linear Support Vector Machine Classifier
    
    Implements SVM for linearly separable data using the dual formulation:
    
    Maximize: ∑αi - (1/2)∑∑αiαjyiyj(xi·xj)
    Subject to: 0 ≤ αi ≤ C, ∑αiyi = 0
    
    Attributes:
        C (float): Regularization parameter
        tol (float): Tolerance for stopping criteria
        max_iter (int): Maximum iterations for SMO
        alpha (np.ndarray): Lagrange multipliers
        b (float): Bias term
        w (np.ndarray): Weight vector (for linear kernel)
        support_vectors_ (np.ndarray): Support vectors
        support_labels_ (np.ndarray): Support vector labels
        n_support_ (np.ndarray): Number of support vectors per class
    """
    
    def __init__(self, C: float = 1.0, tol: float = 1e-3, 
                 max_iter: int = 1000, random_state: Optional[int] = None):
        """
        Initialize Linear SVM
        
        Args:
            C: Regularization parameter (higher C = less regularization)
            tol: Tolerance for stopping criteria
            max_iter: Maximum iterations for SMO algorithm
            random_state: Random seed for reproducibility
        """
        self.C = C
        self.tol = tol
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Initialize attributes
        self.alpha = None
        self.b = 0.0
        self.w = None
        self.support_vectors_ = None
        self.support_labels_ = None
        self.n_support_ = None
        
        # Training data (kept for support vector extraction)
        self._X_train = None
        self._y_train = None
        
        # Set random seed
        if random_state is not None:
            np.random.seed(random_state)
    
    def _compute_gram_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        Compute Gram matrix (kernel matrix) for linear kernel
        
        For linear kernel: K(xi, xj) = xi · xj
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Gram matrix (n_samples, n_samples)
        """
        return np.dot(X, X.T)
    
    def _objective_function(self, alpha: np.ndarray, K: np.ndarray, 
                          y: np.ndarray) -> float:
        """
        Compute SVM dual objective function value
        
        L(α) = ∑αi - (1/2)∑∑αiαjyiyjK(xi,xj)
        
        Args:
            alpha: Lagrange multipliers
            K: Gram matrix
            y: Labels
            
        Returns:
            Objective function value
        """
        # First term: ∑αi
        term1 = np.sum(alpha)
        
        # Second term: (1/2)∑∑αiαjyiyjK(xi,xj)
        y_alpha = y * alpha
        term2 = 0.5 * np.dot(y_alpha, np.dot(K, y_alpha))
        
        return term1 - term2
    
    def _select_alphas_smo(self, alpha: np.ndarray, y: np.ndarray, 
                          K: np.ndarray, E: np.ndarray) -> Tuple[int, int]:
        """
        Select pair of alphas to optimize using SMO heuristics
        
        Args:
            alpha: Current alpha values
            y: Labels
            K: Gram matrix
            E: Error cache
            
        Returns:
            Indices of selected alpha pair (i, j)
        """
        n_samples = len(alpha)
        
        # First heuristic: choose alpha that violates KKT conditions most
        non_bound_idx = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]
        
        if len(non_bound_idx) > 0:
            # Choose from non-boundary alphas
            i = np.random.choice(non_bound_idx)
        else:
            # Choose randomly from all alphas
            i = np.random.randint(0, n_samples)
        
        # Second heuristic: choose alpha that maximizes |E_i - E_j|
        E_i = E[i]
        if E_i >= 0:
            j = np.argmin(E)
        else:
            j = np.argmax(E)
        
        # Ensure i != j
        if i == j:
            candidates = list(range(n_samples))
            candidates.remove(i)
            j = np.random.choice(candidates)
        
        return i, j
    
    def _clip_alpha(self, alpha_new: float, L: float, H: float) -> float:
        """
        Clip alpha to feasible range [L, H]
        
        Args:
            alpha_new: New alpha value
            L: Lower bound
            H: Upper bound
            
        Returns:
            Clipped alpha value
        """
        if alpha_new > H:
            return H
        elif alpha_new < L:
            return L
        else:
            return alpha_new
    
    def _smo_step(self, i: int, j: int, alpha: np.ndarray, 
                  y: np.ndarray, K: np.ndarray, E: np.ndarray, 
                  b: float) -> Tuple[np.ndarray, float, bool]:
        """
        Perform one SMO optimization step
        
        Args:
            i, j: Indices of alphas to optimize
            alpha: Current alpha values
            y: Labels  
            K: Gram matrix
            E: Error cache
            b: Current bias
            
        Returns:
            Updated alpha, bias, and success flag
        """
        if i == j:
            return alpha, b, False
        
        alpha_old = alpha.copy()
        
        # Calculate bounds L and H
        if y[i] != y[j]:
            L = max(0, alpha[j] - alpha[i])
            H = min(self.C, self.C + alpha[j] - alpha[i])
        else:
            L = max(0, alpha[i] + alpha[j] - self.C)
            H = min(self.C, alpha[i] + alpha[j])
        
        if L == H:
            return alpha, b, False
        
        # Calculate eta (second derivative)
        eta = 2 * K[i, j] - K[i, i] - K[j, j]
        
        if eta >= 0:
            # Unusual case, skip this pair
            return alpha, b, False
        
        # Calculate new alpha_j
        alpha_j_new = alpha[j] - y[j] * (E[i] - E[j]) / eta
        
        # Clip alpha_j
        alpha_j_new = self._clip_alpha(alpha_j_new, L, H)
        
        # If change is too small, skip
        if abs(alpha_j_new - alpha[j]) < self.tol:
            return alpha, b, False
        
        # Calculate new alpha_i
        alpha_i_new = alpha[i] + y[i] * y[j] * (alpha[j] - alpha_j_new)
        
        # Update alphas
        alpha[i] = alpha_i_new
        alpha[j] = alpha_j_new
        
        # Update bias
        b1 = E[i] + y[i] * (alpha_i_new - alpha_old[i]) * K[i, i] + \
             y[j] * (alpha_j_new - alpha_old[j]) * K[i, j] + b
        
        b2 = E[j] + y[i] * (alpha_i_new - alpha_old[i]) * K[i, j] + \
             y[j] * (alpha_j_new - alpha_old[j]) * K[j, j] + b
        
        # Choose bias based on alpha values
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
        Compute prediction errors for all samples
        
        E_i = f(xi) - yi where f(xi) = ∑αjyjK(xi,xj) + b
        
        Args:
            alpha: Lagrange multipliers
            y: True labels
            K: Gram matrix  
            b: Bias term
            
        Returns:
            Error array
        """
        # f(x) = ∑αjyjK(xi,xj) + b
        predictions = np.dot(K, alpha * y) + b
        return predictions - y
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'LinearSVM':
        """
        Train the Linear SVM using SMO algorithm
        
        Args:
            X: Training data (n_samples, n_features)
            y: Training labels (n_samples,) - should be -1 or +1
            
        Returns:
            self
        """
        # Validate input
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        
        if X.ndim != 2:
            raise ValueError("X must be 2D array")
        if y.ndim != 1:
            raise ValueError("y must be 1D array")
        if X.shape[0] != y.shape[0]:
            raise ValueError("X and y must have same number of samples")
        
        # Convert labels to -1, +1
        unique_labels = np.unique(y)
        if len(unique_labels) != 2:
            raise ValueError("Only binary classification supported")
        
        # Map labels to -1, +1
        self.classes_ = unique_labels
        y_mapped = np.where(y == unique_labels[0], -1, 1)
        
        # Store training data
        self._X_train = X.copy()
        self._y_train = y_mapped.copy()
        
        n_samples, n_features = X.shape
        
        # Initialize alphas and bias
        alpha = np.zeros(n_samples)
        b = 0.0
        
        # Compute Gram matrix
        K = self._compute_gram_matrix(X)
        
        # SMO main loop
        num_changed = 0
        examine_all = True
        iteration = 0
        
        print(f"Starting SMO optimization...")
        start_time = time.time()
        
        while (num_changed > 0 or examine_all) and iteration < self.max_iter:
            num_changed = 0
            
            # Compute errors
            E = self._compute_errors(alpha, y_mapped, K, b)
            
            if examine_all:
                # Examine all samples
                for i in range(n_samples):
                    # Check KKT conditions
                    if self._violates_kkt(alpha[i], y_mapped[i], E[i]):
                        j = (i + 1) % n_samples  # Simple selection
                        alpha, b, changed = self._smo_step(i, j, alpha, y_mapped, K, E, b)
                        if changed:
                            num_changed += 1
            else:
                # Examine non-bound samples
                non_bound_idx = np.where((alpha > self.tol) & (alpha < self.C - self.tol))[0]
                for i in non_bound_idx:
                    if self._violates_kkt(alpha[i], y_mapped[i], E[i]):
                        j = (i + 1) % n_samples
                        alpha, b, changed = self._smo_step(i, j, alpha, y_mapped, K, E, b)
                        if changed:
                            num_changed += 1
            
            if examine_all:
                examine_all = False
            elif num_changed == 0:
                examine_all = True
            
            iteration += 1
            
            if iteration % 100 == 0:
                obj_val = self._objective_function(alpha, K, y_mapped)
                print(f"Iteration {iteration}, Objective: {obj_val:.6f}, "
                      f"Non-zero alphas: {np.sum(alpha > self.tol)}")
        
        training_time = time.time() - start_time
        print(f"SMO completed in {iteration} iterations ({training_time:.2f}s)")
        
        # Store results
        self.alpha = alpha
        self.b = b
        
        # Compute weight vector for linear kernel
        self.w = np.dot((alpha * y_mapped).T, X)
        
        # Extract support vectors
        sv_idx = alpha > self.tol
        self.support_vectors_ = X[sv_idx]
        self.support_labels_ = y[sv_idx]  # Original labels
        self.n_support_ = np.array([np.sum(y_mapped[sv_idx] == -1), 
                                   np.sum(y_mapped[sv_idx] == 1)])
        
        print(f"Training completed:")
        print(f"  Support vectors: {len(self.support_vectors_)}/{n_samples} "
              f"({100*len(self.support_vectors_)/n_samples:.1f}%)")
        print(f"  Class distribution: {self.n_support_}")
        
        return self
    
    def _violates_kkt(self, alpha_i: float, y_i: float, E_i: float) -> bool:
        """
        Check if sample violates KKT conditions
        
        KKT conditions for SVM:
        - αi = 0 ⟹ yi*f(xi) ≥ 1
        - 0 < αi < C ⟹ yi*f(xi) = 1  
        - αi = C ⟹ yi*f(xi) ≤ 1
        
        Args:
            alpha_i: Alpha value for sample i
            y_i: Label for sample i
            E_i: Error for sample i
            
        Returns:
            True if KKT conditions are violated
        """
        r_i = E_i * y_i  # yi * (f(xi) - yi) = yi*f(xi) - 1
        
        if (r_i < -self.tol and alpha_i < self.C) or \
           (r_i > self.tol and alpha_i > 0):
            return True
        return False
    
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """
        Compute decision function values
        
        For linear SVM: f(x) = w·x + b
        
        Args:
            X: Input data (n_samples, n_features)
            
        Returns:
            Decision function values (n_samples,)
        """
        if self.w is None:
            raise ValueError("Model not fitted yet")
        
        X = np.asarray(X, dtype=np.float64)
        return np.dot(X, self.w) + self.b
    
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
    
    def get_params(self) -> dict:
        """Get hyperparameters"""
        return {
            'C': self.C,
            'tol': self.tol,
            'max_iter': self.max_iter,
            'random_state': self.random_state
        }
    
    def set_params(self, **params) -> 'LinearSVM':
        """Set hyperparameters"""
        for param, value in params.items():
            if hasattr(self, param):
                setattr(self, param, value)
            else:
                raise ValueError(f"Invalid parameter: {param}")
        return self

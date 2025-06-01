"""
Data preprocessing utilities for SVM implementations.

This module provides comprehensive preprocessing functions including:
- Feature scaling and normalization
- Text preprocessing for NLP tasks
- Data splitting and validation
- Feature selection and dimensionality reduction
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.decomposition import PCA
from typing import Tuple, Optional, Union, Dict, Any
import re
import string


class DataPreprocessor:
    """
    Comprehensive data preprocessing class for SVM applications.
    
    Handles both numerical and text data preprocessing with various
    scaling, encoding, and feature selection options.
    """
    
    def __init__(self):
        self.scalers = {}
        self.encoders = {}
        self.vectorizers = {}
        self.feature_selectors = {}
        self.fitted = False
    
    def fit_transform_numerical(
        self, 
        X: np.ndarray, 
        y: Optional[np.ndarray] = None,
        scaling_method: str = 'standard',
        feature_selection: Optional[str] = None,
        n_features: Optional[int] = None
    ) -> np.ndarray:
        """
        Fit and transform numerical features.
        
        Args:
            X: Input features
            y: Target values (optional, needed for feature selection)
            scaling_method: 'standard', 'minmax', or 'none'
            feature_selection: 'univariate' or None
            n_features: Number of features to select
            
        Returns:
            Transformed features
        """
        X_processed = X.copy()
        
        # Feature scaling
        if scaling_method == 'standard':
            scaler = StandardScaler()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers['standard'] = scaler
        elif scaling_method == 'minmax':
            scaler = MinMaxScaler()
            X_processed = scaler.fit_transform(X_processed)
            self.scalers['minmax'] = scaler
        
        # Feature selection
        if feature_selection == 'univariate' and y is not None:
            if n_features is None:
                n_features = min(X.shape[1] // 2, 20)
            
            # Determine if classification or regression
            if len(np.unique(y)) < 10:  # Likely classification
                selector = SelectKBest(f_classif, k=n_features)
            else:  # Likely regression
                selector = SelectKBest(f_regression, k=n_features)
            
            X_processed = selector.fit_transform(X_processed, y)
            self.feature_selectors['univariate'] = selector
        
        self.fitted = True
        return X_processed
    
    def transform_numerical(self, X: np.ndarray) -> np.ndarray:
        """Transform numerical features using fitted preprocessors."""
        if not self.fitted:
            raise ValueError("Preprocessor must be fitted before transform")
        
        X_processed = X.copy()
        
        # Apply scaling
        if 'standard' in self.scalers:
            X_processed = self.scalers['standard'].transform(X_processed)
        elif 'minmax' in self.scalers:
            X_processed = self.scalers['minmax'].transform(X_processed)
        
        # Apply feature selection
        if 'univariate' in self.feature_selectors:
            X_processed = self.feature_selectors['univariate'].transform(X_processed)
        
        return X_processed
    
    def preprocess_text(
        self,
        texts: list,
        method: str = 'tfidf',
        max_features: int = 5000,
        min_df: int = 2,
        max_df: float = 0.95,
        stop_words: str = 'english'
    ) -> np.ndarray:
        """
        Preprocess text data for SVM classification.
        
        Args:
            texts: List of text documents
            method: 'tfidf' or 'count'
            max_features: Maximum number of features
            min_df: Minimum document frequency
            max_df: Maximum document frequency
            stop_words: Stop words to remove
            
        Returns:
            Text features matrix
        """
        # Clean texts
        cleaned_texts = [self._clean_text(text) for text in texts]
        
        # Vectorize
        if method == 'tfidf':
            vectorizer = TfidfVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                ngram_range=(1, 2)
            )
        else:
            vectorizer = CountVectorizer(
                max_features=max_features,
                min_df=min_df,
                max_df=max_df,
                stop_words=stop_words,
                ngram_range=(1, 2)
            )
        
        X_text = vectorizer.fit_transform(cleaned_texts).toarray()
        self.vectorizers[method] = vectorizer
        
        return X_text
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text."""
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation and special characters
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def encode_labels(self, y: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Encode categorical labels to numerical values.
        
        Args:
            y: Target labels
            
        Returns:
            Encoded labels and mapping dictionary
        """
        if y.dtype == 'object' or isinstance(y[0], str):
            encoder = LabelEncoder()
            y_encoded = encoder.fit_transform(y)
            self.encoders['label'] = encoder
            
            # Create mapping dictionary
            mapping = {
                label: encoded for label, encoded in 
                zip(encoder.classes_, encoder.transform(encoder.classes_))
            }
            
            return y_encoded, mapping
        
        return y, {}


class DataSplitter:
    """Utility class for data splitting and cross-validation."""
    
    @staticmethod
    def train_test_split_data(
        X: np.ndarray,
        y: np.ndarray,
        test_size: float = 0.2,
        random_state: int = 42,
        stratify: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split data into training and testing sets.
        
        Args:
            X: Features
            y: Target values
            test_size: Proportion of test set
            random_state: Random seed
            stratify: Whether to stratify split
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        stratify_param = y if stratify and len(np.unique(y)) < len(y) // 2 else None
        
        return train_test_split(
            X, y, 
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_param
        )
    
    @staticmethod
    def create_cv_folds(
        X: np.ndarray,
        y: np.ndarray,
        n_folds: int = 5,
        random_state: int = 42
    ) -> StratifiedKFold:
        """
        Create cross-validation folds.
        
        Args:
            X: Features
            y: Target values
            n_folds: Number of folds
            random_state: Random seed
            
        Returns:
            StratifiedKFold object
        """
        return StratifiedKFold(
            n_splits=n_folds,
            shuffle=True,
            random_state=random_state
        )


class DimensionalityReducer:
    """Dimensionality reduction utilities."""
    
    def __init__(self):
        self.reducers = {}
    
    def apply_pca(
        self,
        X: np.ndarray,
        n_components: Optional[int] = None,
        explained_variance_ratio: float = 0.95
    ) -> Tuple[np.ndarray, PCA]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Input features
            n_components: Number of components (optional)
            explained_variance_ratio: Minimum variance to retain
            
        Returns:
            Transformed features and fitted PCA object
        """
        if n_components is None:
            # Find number of components for desired variance
            pca_temp = PCA()
            pca_temp.fit(X)
            cumsum = np.cumsum(pca_temp.explained_variance_ratio_)
            n_components = np.argmax(cumsum >= explained_variance_ratio) + 1
        
        pca = PCA(n_components=n_components)
        X_reduced = pca.fit_transform(X)
        
        self.reducers['pca'] = pca
        
        return X_reduced, pca
    
    def get_pca_info(self, pca: PCA) -> Dict[str, Any]:
        """Get information about PCA transformation."""
        return {
            'n_components': pca.n_components_,
            'explained_variance_ratio': pca.explained_variance_ratio_,
            'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
            'total_variance_explained': np.sum(pca.explained_variance_ratio_)
        }


def create_synthetic_data(
    n_samples: int = 1000,
    n_features: int = 2,
    n_classes: int = 2,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic classification data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        n_classes: Number of classes
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Features and labels
    """
    np.random.seed(random_state)
    
    # Generate class centers
    centers = np.random.randn(n_classes, n_features) * 3
    
    X = []
    y = []
    
    samples_per_class = n_samples // n_classes
    
    for i in range(n_classes):
        # Generate samples around each center
        class_samples = np.random.multivariate_normal(
            centers[i], 
            np.eye(n_features) * noise,
            samples_per_class
        )
        X.append(class_samples)
        y.extend([i] * samples_per_class)
    
    X = np.vstack(X)
    y = np.array(y)
    
    # Shuffle data
    indices = np.random.permutation(len(X))
    X = X[indices]
    y = y[indices]
    
    return X, y


def create_synthetic_regression_data(
    n_samples: int = 1000,
    n_features: int = 1,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create synthetic regression data for testing.
    
    Args:
        n_samples: Number of samples
        n_features: Number of features
        noise: Noise level
        random_state: Random seed
        
    Returns:
        Features and target values
    """
    np.random.seed(random_state)
    
    X = np.random.randn(n_samples, n_features)
    
    # Create non-linear relationship
    if n_features == 1:
        y = np.sin(2 * np.pi * X.flatten()) + 0.5 * X.flatten()**2
    else:
        y = np.sum(X**2, axis=1) + np.sin(np.sum(X, axis=1))
    
    # Add noise
    y += np.random.randn(n_samples) * noise
    
    return X, y


if __name__ == "__main__":
    # Example usage
    print("Creating synthetic classification data...")
    X_class, y_class = create_synthetic_data(n_samples=200, n_features=2)
    
    print("Creating synthetic regression data...")
    X_reg, y_reg = create_synthetic_regression_data(n_samples=200)
    
    print("Testing preprocessing...")
    preprocessor = DataPreprocessor()
    
    # Test numerical preprocessing
    X_processed = preprocessor.fit_transform_numerical(
        X_class, y_class, 
        scaling_method='standard'
    )
    print(f"Original shape: {X_class.shape}, Processed shape: {X_processed.shape}")
    
    # Test data splitting
    X_train, X_test, y_train, y_test = DataSplitter.train_test_split_data(
        X_processed, y_class
    )
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
    
    print("Preprocessing utilities created successfully!")

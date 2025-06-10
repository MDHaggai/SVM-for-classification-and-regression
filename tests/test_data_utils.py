import pandas as pd
import pytest
from src.utils.data_loader import load_heart_disease_data, load_news_classification_data, load_wine_quality_data

def test_load_heart_disease_data():
    X, y = load_heart_disease_data()
    assert X.shape[0] == 303  # Check number of instances
    assert X.shape[1] == 13   # Check number of features
    assert y.nunique() == 2    # Check binary classification

def test_load_news_classification_data():
    X, y = load_news_classification_data()
    assert X.shape[0] == 2225  # Check number of instances
    assert X.shape[1] == 1000  # Check number of features (TF-IDF vectors)
    assert y.nunique() == 5     # Check number of categories

def test_load_wine_quality_data():
    X, y = load_wine_quality_data()
    assert X.shape[0] == 4898  # Check number of instances
    assert X.shape[1] == 11    # Check number of features
    assert y.nunique() == 7     # Check number of quality scores

def test_data_integrity():
    # Example test to check for NaN values in datasets
    X, y = load_heart_disease_data()
    assert not X.isnull().values.any(), "Heart disease data contains NaN values"
    
    X, y = load_news_classification_data()
    assert not X.isnull().values.any(), "News classification data contains NaN values"
    
    X, y = load_wine_quality_data()
    assert not X.isnull().values.any(), "Wine quality data contains NaN values"
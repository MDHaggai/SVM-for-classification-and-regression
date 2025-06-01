"""
Pytest configuration for the SVM project.
"""
import pytest
import sys
import os

# Add src to path for pytest
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

# Configure pytest markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')")
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "visualization: marks tests that create visualizations")

# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Mark all test_utils.py tests as integration tests
        if "test_utils" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Mark visualization tests
        if "visualization" in item.name or "plot" in item.name:
            item.add_marker(pytest.mark.visualization)
        
        # Mark tests that might be slow
        if any(keyword in item.name for keyword in ["benchmark", "convergence", "large"]):
            item.add_marker(pytest.mark.slow)

# Fixtures
@pytest.fixture
def sample_classification_data():
    """Provide sample classification data for tests."""
    import numpy as np
    from sklearn.datasets import make_classification
    
    X, y = make_classification(
        n_samples=100, n_features=2, n_redundant=0, 
        n_informative=2, random_state=42, n_clusters_per_class=1
    )
    y[y == 0] = -1  # Convert to SVM format
    return X, y

@pytest.fixture
def sample_regression_data():
    """Provide sample regression data for tests."""
    import numpy as np
    from sklearn.datasets import make_regression
    
    X, y = make_regression(
        n_samples=100, n_features=2, noise=0.1, random_state=42
    )
    return X, y

@pytest.fixture
def temp_data_dir(tmp_path):
    """Provide temporary directory for test data."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir()
    return data_dir

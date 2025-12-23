"""
Test configuration and fixtures.
"""

import pytest
import numpy as np
from typing import Tuple

@pytest.fixture
def sample_regression_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate sample regression data for testing."""
    np.random.seed(42)
    
    # Generate synthetic data
    n_samples = 100
    n_features = 5
    
    X = np.random.randn(n_samples, n_features)
    # Create target with known relationship
    y = X[:, 0] + 0.5 * X[:, 1] + 0.3 * np.random.randn(n_samples)
    
    # Split into train/test
    split_idx = int(0.8 * n_samples)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    return X_train, X_test, y_train, y_test


@pytest.fixture
def small_regression_data() -> Tuple[np.ndarray, np.ndarray]:
    """Generate small regression dataset for quick tests."""
    np.random.seed(42)
    
    X = np.array([
        [1.0, 2.0],
        [2.0, 3.0],
        [3.0, 4.0],
        [4.0, 5.0],
        [5.0, 6.0]
    ])
    
    y = np.array([1.5, 2.5, 3.5, 4.5, 5.5])
    
    return X, y
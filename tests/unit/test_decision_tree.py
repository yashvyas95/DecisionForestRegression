"""
Unit tests for DecisionTree class.
"""

import pytest
import numpy as np
from decision_forest.core import DecisionTree


class TestDecisionTree:
    """Test cases for DecisionTree class."""
    
    def test_init(self):
        """Test DecisionTree initialization."""
        tree = DecisionTree(max_depth=10, min_samples_split=5)
        
        assert tree.max_depth == 10
        assert tree.min_samples_split == 5
        assert tree.min_samples_leaf == 1
        assert tree.tree_ is None
        assert tree.n_features_ == 0
    
    def test_fit_simple(self, small_regression_data):
        """Test fitting on simple data."""
        X, y = small_regression_data
        
        tree = DecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        assert tree.tree_ is not None
        assert tree.n_features_ == X.shape[1]
        assert tree.n_samples_ == X.shape[0]
        assert tree.feature_importances_ is not None
    
    def test_predict(self, small_regression_data):
        """Test prediction on simple data."""
        X, y = small_regression_data
        
        tree = DecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        predictions = tree.predict(X)
        
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_predict_not_fitted(self, small_regression_data):
        """Test prediction on unfitted tree raises error."""
        X, y = small_regression_data
        
        tree = DecisionTree()
        
        with pytest.raises(ValueError, match="Tree must be fitted"):
            tree.predict(X)
    
    def test_get_depth(self, small_regression_data):
        """Test getting tree depth."""
        X, y = small_regression_data
        
        tree = DecisionTree(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        depth = tree.get_depth()
        assert isinstance(depth, int)
        assert depth >= 0
        assert depth <= 2  # Should respect max_depth
    
    def test_get_n_leaves(self, small_regression_data):
        """Test getting number of leaves."""
        X, y = small_regression_data
        
        tree = DecisionTree(max_depth=2, random_state=42)
        tree.fit(X, y)
        
        n_leaves = tree.get_n_leaves()
        assert isinstance(n_leaves, int)
        assert n_leaves >= 1
    
    def test_feature_importances(self, small_regression_data):
        """Test feature importance calculation."""
        X, y = small_regression_data
        
        tree = DecisionTree(random_state=42)
        tree.fit(X, y)
        
        importances = tree.feature_importances_
        assert importances is not None
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)
        assert abs(sum(importances) - 1.0) < 1e-6  # Should sum to 1
    
    def test_different_splitters(self, small_regression_data):
        """Test different splitter strategies."""
        X, y = small_regression_data
        
        # Test best splitter
        tree_best = DecisionTree(splitter="best", random_state=42)
        tree_best.fit(X, y)
        
        # Test random splitter
        tree_random = DecisionTree(splitter="random", random_state=42)
        tree_random.fit(X, y)
        
        # Both should work
        pred_best = tree_best.predict(X)
        pred_random = tree_random.predict(X)
        
        assert len(pred_best) == len(y)
        assert len(pred_random) == len(y)
    
    def test_invalid_splitter(self):
        """Test invalid splitter raises error."""
        with pytest.raises(ValueError, match="Unknown splitter"):
            DecisionTree(splitter="invalid")
    
    def test_min_samples_constraints(self, small_regression_data):
        """Test minimum samples constraints."""
        X, y = small_regression_data
        
        # Test min_samples_split
        tree = DecisionTree(min_samples_split=10, random_state=42)  # Large value
        tree.fit(X, y)
        
        # Should create a simple tree (likely just root)
        assert tree.get_n_leaves() >= 1
        
        # Test min_samples_leaf
        tree = DecisionTree(min_samples_leaf=3, random_state=42)
        tree.fit(X, y)
        
        # Should respect minimum leaf samples
        assert tree.get_n_leaves() >= 1
    
    def test_serialization(self, small_regression_data):
        """Test model serialization and deserialization."""
        X, y = small_regression_data
        
        tree = DecisionTree(max_depth=3, random_state=42)
        tree.fit(X, y)
        
        # Test to_dict and from_dict
        tree_dict = tree.to_dict()
        tree_restored = DecisionTree.from_dict(tree_dict)
        
        # Compare predictions
        pred_original = tree.predict(X)
        pred_restored = tree_restored.predict(X)
        
        np.testing.assert_array_almost_equal(pred_original, pred_restored)
    
    def test_invalid_input_dimensions(self, small_regression_data):
        """Test handling of invalid input dimensions."""
        X, y = small_regression_data
        
        tree = DecisionTree(random_state=42)
        tree.fit(X, y)
        
        # Test wrong number of features
        X_wrong = X[:, :1]  # Remove one feature
        
        with pytest.raises(ValueError, match="Expected .* features"):
            tree.predict(X_wrong)
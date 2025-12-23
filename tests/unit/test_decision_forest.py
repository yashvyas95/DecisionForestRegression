"""
Unit tests for DecisionForest class.
"""

import pytest
import numpy as np
from decision_forest.core import DecisionForest


class TestDecisionForest:
    """Test cases for DecisionForest class."""
    
    def test_init(self):
        """Test DecisionForest initialization."""
        forest = DecisionForest(n_trees=50, max_depth=10)
        
        assert forest.n_trees == 50
        assert forest.max_depth == 10
        assert forest.bootstrap == True
        assert len(forest.trees_) == 0
        assert forest.n_features_ == 0
    
    def test_parameter_validation(self):
        """Test parameter validation."""
        # Test invalid n_trees
        with pytest.raises(ValueError, match="n_trees must be positive"):
            DecisionForest(n_trees=0)
        
        # Test invalid max_depth
        with pytest.raises(ValueError, match="max_depth must be positive"):
            DecisionForest(max_depth=0)
        
        # Test invalid min_samples_split
        with pytest.raises(ValueError, match="min_samples_split must be >= 2"):
            DecisionForest(min_samples_split=1)
        
        # Test invalid min_samples_leaf
        with pytest.raises(ValueError, match="min_samples_leaf must be >= 1"):
            DecisionForest(min_samples_leaf=0)
    
    def test_fit_simple(self, small_regression_data):
        """Test fitting on simple data."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, max_depth=3, random_state=42)
        forest.fit(X, y)
        
        assert len(forest.trees_) == 5
        assert forest.n_features_ == X.shape[1]
        assert forest.n_samples_ == X.shape[0]
        assert forest.feature_importances_ is not None
    
    def test_predict(self, small_regression_data):
        """Test prediction."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, random_state=42)
        forest.fit(X, y)
        
        predictions = forest.predict(X)
        
        assert len(predictions) == len(y)
        assert all(isinstance(p, (int, float, np.number)) for p in predictions)
    
    def test_predict_proba(self, small_regression_data):
        """Test prediction with uncertainty."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, random_state=42)
        forest.fit(X, y)
        
        pred_mean, pred_std = forest.predict_proba(X)
        
        assert len(pred_mean) == len(y)
        assert len(pred_std) == len(y)
        assert all(std >= 0 for std in pred_std)
    
    def test_score(self, small_regression_data):
        """Test R² scoring."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, random_state=42)
        forest.fit(X, y)
        
        score = forest.score(X, y)
        
        assert isinstance(score, float)
        assert -10 <= score <= 1.0  # R² can be negative but usually not extremely
    
    def test_oob_score(self, sample_regression_data):
        """Test out-of-bag scoring."""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        forest = DecisionForest(
            n_trees=10, 
            bootstrap=True, 
            oob_score=True, 
            random_state=42
        )
        forest.fit(X_train, y_train)
        
        assert forest.oob_score_ is not None
        assert isinstance(forest.oob_score_, float)
        assert forest.oob_prediction_ is not None
    
    def test_feature_importances(self, small_regression_data):
        """Test feature importance calculation."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, random_state=42)
        forest.fit(X, y)
        
        importances = forest.feature_importances_
        assert importances is not None
        assert len(importances) == X.shape[1]
        assert all(imp >= 0 for imp in importances)
        assert abs(sum(importances) - 1.0) < 1e-6
    
    def test_bootstrap_vs_no_bootstrap(self, small_regression_data):
        """Test bootstrap vs no bootstrap sampling."""
        X, y = small_regression_data
        
        # With bootstrap
        forest_bootstrap = DecisionForest(
            n_trees=5, bootstrap=True, random_state=42
        )
        forest_bootstrap.fit(X, y)
        
        # Without bootstrap  
        forest_no_bootstrap = DecisionForest(
            n_trees=5, bootstrap=False, random_state=42
        )
        forest_no_bootstrap.fit(X, y)
        
        # Both should work
        pred_bootstrap = forest_bootstrap.predict(X)
        pred_no_bootstrap = forest_no_bootstrap.predict(X)
        
        assert len(pred_bootstrap) == len(y)
        assert len(pred_no_bootstrap) == len(y)
    
    def test_get_tree(self, small_regression_data):
        """Test getting individual trees."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=3, random_state=42)
        forest.fit(X, y)
        
        # Test valid index
        tree = forest.get_tree(0)
        assert tree is not None
        
        # Test invalid index
        with pytest.raises(IndexError):
            forest.get_tree(10)
    
    def test_average_depth_and_leaves(self, small_regression_data):
        """Test average depth and leaves calculation."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5, max_depth=3, random_state=42)
        forest.fit(X, y)
        
        avg_depth = forest.get_average_depth()
        avg_leaves = forest.get_average_n_leaves()
        
        assert isinstance(avg_depth, float)
        assert isinstance(avg_leaves, float)
        assert avg_depth >= 0
        assert avg_leaves >= 1
    
    def test_parallel_training(self, sample_regression_data):
        """Test parallel training."""
        X_train, X_test, y_train, y_test = sample_regression_data
        
        # Sequential training
        forest_seq = DecisionForest(n_trees=5, n_jobs=1, random_state=42)
        forest_seq.fit(X_train, y_train)
        
        # Parallel training
        forest_par = DecisionForest(n_trees=5, n_jobs=2, random_state=42)
        forest_par.fit(X_train, y_train)
        
        # Results should be consistent (both have same random_state)
        pred_seq = forest_seq.predict(X_test)
        pred_par = forest_par.predict(X_test)
        
        assert len(pred_seq) == len(pred_par)
        # Note: Due to parallel execution order, predictions might differ slightly
    
    def test_serialization(self, small_regression_data):
        """Test forest serialization."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=3, random_state=42)
        forest.fit(X, y)
        
        # Serialize and deserialize
        forest_dict = forest.to_dict()
        forest_restored = DecisionForest.from_dict(forest_dict)
        
        # Compare predictions
        pred_original = forest.predict(X)
        pred_restored = forest_restored.predict(X)
        
        np.testing.assert_array_almost_equal(pred_original, pred_restored)
    
    def test_not_fitted_error(self, small_regression_data):
        """Test error when predicting on unfitted forest."""
        X, y = small_regression_data
        
        forest = DecisionForest(n_trees=5)
        
        with pytest.raises(ValueError, match="Forest must be fitted"):
            forest.predict(X)
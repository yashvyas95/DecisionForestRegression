"""
Decision Tree implementation for regression.

This module contains the DecisionTree class that implements a single
decision tree for regression tasks with modern Python practices.
"""

from typing import Optional, Union, Dict, Any, List
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
import joblib
import logging

from .node import Node
from .splitters import BestSplitter, RandomSplitter, BaseSplitter


logger = logging.getLogger(__name__)


class DecisionTree(BaseEstimator, RegressorMixin):
    """
    Decision Tree Regressor with modern implementation.
    
    This implementation provides a single decision tree for regression
    with support for different splitting criteria, pruning, and
    feature importance calculation.
    
    Attributes:
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
        min_samples_leaf: Minimum samples required in a leaf node
        max_features: Number of features to consider for best split
        splitter: Splitting strategy ("best" or "random")
        random_state: Random state for reproducibility
        criterion: Splitting criterion ("mse")
        
    Example:
        >>> from decision_forest.core import DecisionTree
        >>> import numpy as np
        >>> 
        >>> # Create sample data
        >>> X = np.random.rand(100, 4)
        >>> y = X[:, 0] + X[:, 1] + np.random.normal(0, 0.1, 100)
        >>> 
        >>> # Create and train tree
        >>> tree = DecisionTree(max_depth=5, random_state=42)
        >>> tree.fit(X, y)
        >>> 
        >>> # Make predictions
        >>> predictions = tree.predict(X)
    """
    
    def __init__(
        self,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = None,
        splitter: str = "best",
        random_state: Optional[int] = None,
        criterion: str = "mse"
    ) -> None:
        """
        Initialize the Decision Tree.
        
        Args:
            max_depth: Maximum depth of the tree
            min_samples_split: Minimum samples required to split
            min_samples_leaf: Minimum samples in a leaf
            max_features: Number of features to consider
            splitter: Splitting strategy ("best" or "random")
            random_state: Random state for reproducibility
            criterion: Splitting criterion ("mse")
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.splitter = splitter
        self.random_state = random_state
        self.criterion = criterion
        
        # Initialize attributes set during training
        self.tree_: Optional[Node] = None
        self.n_features_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.n_samples_: int = 0
        self.n_outputs_: int = 1
        
        # Setup random state
        self.rng = np.random.RandomState(random_state)
        
        # Initialize splitter
        self._init_splitter()
    
    def _init_splitter(self) -> None:
        """Initialize the splitter based on the splitter parameter."""
        if self.splitter == "best":
            self._splitter = BestSplitter(
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        elif self.splitter == "random":
            self._splitter = RandomSplitter(
                criterion=self.criterion,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown splitter: {self.splitter}")
    
    def _get_max_features(self, n_features: int) -> int:
        """
        Get the number of features to consider for each split.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Number of features to consider
        """
        if self.max_features is None:
            return n_features
        elif isinstance(self.max_features, int):
            return min(self.max_features, n_features)
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_features))
        elif self.max_features == "sqrt":
            return max(1, int(np.sqrt(n_features)))
        elif self.max_features == "log2":
            return max(1, int(np.log2(n_features)))
        else:
            raise ValueError(f"Invalid max_features: {self.max_features}")
    
    def _select_features(self, n_features: int) -> np.ndarray:
        """
        Select features to consider for splitting.
        
        Args:
            n_features: Total number of features
            
        Returns:
            Array of selected feature indices
        """
        max_features = self._get_max_features(n_features)
        
        if max_features >= n_features:
            return np.arange(n_features)
        else:
            return self.rng.choice(n_features, max_features, replace=False)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionTree':
        """
        Fit the decision tree to training data.
        
        Args:
            X: Training features of shape (n_samples, n_features)
            y: Training targets of shape (n_samples,)
            
        Returns:
            Self for method chaining
            
        Raises:
            ValueError: If input data is invalid
        """
        # Validate input data
        X, y = check_X_y(X, y, dtype=np.float32)
        
        # Store dataset information
        self.n_features_ = X.shape[1]
        self.n_samples_ = X.shape[0]
        
        logger.info(f"Training decision tree with {self.n_samples_} samples "
                   f"and {self.n_features_} features")
        
        # Build the tree
        self.tree_ = self._build_tree(X, y, depth=0)
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        logger.info(f"Decision tree trained successfully. "
                   f"Tree depth: {self.get_depth()}, "
                   f"Number of leaves: {self.get_n_leaves()}")
        
        return self
    
    def _build_tree(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """
        Recursively build the decision tree.
        
        Args:
            X: Feature matrix
            y: Target values
            depth: Current depth
            
        Returns:
            Root node of the (sub)tree
        """
        n_samples, n_features = X.shape
        
        # Calculate node statistics
        node_value = np.mean(y)
        node_mse = self._splitter.calculate_mse(y)
        
        # Stopping criteria
        should_stop = (
            depth >= (self.max_depth or float('inf')) or
            n_samples < self.min_samples_split or
            n_samples < 2 * self.min_samples_leaf or
            node_mse == 0.0  # Perfect split
        )
        
        if should_stop:
            return Node(
                value=node_value,
                depth=depth,
                n_samples=n_samples,
                mse=node_mse
            )
        
        # Select features for splitting
        feature_indices = self._select_features(n_features)
        
        # Find best split
        best_feature, best_threshold, best_score = self._splitter.find_best_split(
            X, y, feature_indices
        )
        
        # If no valid split found, create leaf
        if best_feature is None:
            return Node(
                value=node_value,
                depth=depth,
                n_samples=n_samples,
                mse=node_mse
            )
        
        # Split the data
        left_mask = X[:, best_feature] <= best_threshold
        right_mask = ~left_mask
        
        X_left, y_left = X[left_mask], y[left_mask]
        X_right, y_right = X[right_mask], y[right_mask]
        
        # Recursively build left and right subtrees
        left_child = self._build_tree(X_left, y_left, depth + 1)
        right_child = self._build_tree(X_right, y_right, depth + 1)
        
        return Node(
            feature=best_feature,
            threshold=best_threshold,
            left=left_child,
            right=right_child,
            depth=depth,
            n_samples=n_samples,
            mse=node_mse
        )
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
            
        Raises:
            ValueError: If tree is not fitted or input is invalid
        """
        # Validate input
        if self.tree_ is None:
            raise ValueError("Tree must be fitted before making predictions")
        
        X = check_array(X, dtype=np.float32)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        
        # Make predictions for each sample
        predictions = np.array([self.tree_.predict(sample) for sample in X])
        
        return predictions
    
    def _calculate_feature_importances(self) -> None:
        """Calculate feature importances based on the trained tree."""
        if self.tree_ is None:
            self.feature_importances_ = np.zeros(self.n_features_)
            return
        
        importances = self.tree_.get_feature_importance(self.n_features_)
        
        # Normalize importances
        total_importance = np.sum(importances)
        if total_importance > 0:
            importances = importances / total_importance
        
        self.feature_importances_ = importances
    
    def get_depth(self) -> int:
        """
        Get the depth of the tree.
        
        Returns:
            Maximum depth of the tree
        """
        if self.tree_ is None:
            return 0
        return self.tree_.get_depth()
    
    def get_n_leaves(self) -> int:
        """
        Get the number of leaves in the tree.
        
        Returns:
            Number of leaf nodes
        """
        if self.tree_ is None:
            return 0
        return self.tree_.get_n_leaves()
    
    def print_tree(self) -> None:
        """Print the tree structure."""
        if self.tree_ is None:
            print("Tree not fitted")
            return
        
        print("Decision Tree Structure:")
        self.tree_.print_tree()
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert tree to dictionary for serialization.
        
        Returns:
            Dictionary representation of the tree
        """
        tree_dict = {
            'params': {
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'splitter': self.splitter,
                'random_state': self.random_state,
                'criterion': self.criterion
            },
            'n_features_': self.n_features_,
            'n_samples_': self.n_samples_,
            'feature_importances_': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'tree_': self.tree_.to_dict() if self.tree_ is not None else None
        }
        
        return tree_dict
    
    @classmethod
    def from_dict(cls, tree_dict: Dict[str, Any]) -> 'DecisionTree':
        """
        Create tree from dictionary representation.
        
        Args:
            tree_dict: Dictionary representation
            
        Returns:
            DecisionTree instance
        """
        # Create tree with original parameters
        tree = cls(**tree_dict['params'])
        
        # Restore attributes
        tree.n_features_ = tree_dict['n_features_']
        tree.n_samples_ = tree_dict['n_samples_']
        
        if tree_dict['feature_importances_'] is not None:
            tree.feature_importances_ = np.array(tree_dict['feature_importances_'])
        
        if tree_dict['tree_'] is not None:
            tree.tree_ = Node.from_dict(tree_dict['tree_'])
        
        return tree
    
    def save(self, filepath: str) -> None:
        """
        Save the tree to a file.
        
        Args:
            filepath: Path to save the tree
        """
        joblib.dump(self.to_dict(), filepath)
        logger.info(f"Decision tree saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DecisionTree':
        """
        Load a tree from a file.
        
        Args:
            filepath: Path to load the tree from
            
        Returns:
            Loaded DecisionTree instance
        """
        tree_dict = joblib.load(filepath)
        tree = cls.from_dict(tree_dict)
        logger.info(f"Decision tree loaded from {filepath}")
        return tree
    
    def __repr__(self) -> str:
        """String representation of the tree."""
        if self.tree_ is None:
            return "DecisionTree(not fitted)"
        
        return (
            f"DecisionTree(max_depth={self.max_depth}, "
            f"min_samples_split={self.min_samples_split}, "
            f"depth={self.get_depth()}, "
            f"n_leaves={self.get_n_leaves()})"
        )
"""
Decision Forest implementation for regression.

This module contains the DecisionForest class that implements an ensemble
of decision trees for regression tasks with modern Python practices.
"""

from typing import Optional, Union, List, Dict, Any, Tuple
import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_X_y, check_array
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

from .decision_tree import DecisionTree


logger = logging.getLogger(__name__)


class DecisionForest(BaseEstimator, RegressorMixin):
    """
    Decision Forest Regressor implementing ensemble learning.
    
    A Decision Forest is an ensemble of decision trees where each tree
    is trained on a bootstrap sample of the training data. Predictions
    are made by averaging the predictions of all trees.
    
    This implementation includes:
    - Bootstrap sampling for training diversity
    - Out-of-bag (OOB) error estimation  
    - Feature importance calculation
    - Parallel training and prediction
    - Model persistence and serialization
    
    Attributes:
        n_trees: Number of trees in the forest
        max_depth: Maximum depth of individual trees
        min_samples_split: Minimum samples to split a node
        min_samples_leaf: Minimum samples in a leaf
        max_features: Number of features to consider per split
        bootstrap: Whether to use bootstrap sampling
        oob_score: Whether to calculate out-of-bag score
        n_jobs: Number of parallel jobs
        random_state: Random state for reproducibility
        verbose: Verbosity level
        
    Example:
        >>> from decision_forest.core import DecisionForest
        >>> import numpy as np
        >>> 
        >>> # Create sample data
        >>> X = np.random.rand(1000, 10)
        >>> y = np.sum(X[:, :3], axis=1) + np.random.normal(0, 0.1, 1000)
        >>> 
        >>> # Create and train forest
        >>> forest = DecisionForest(n_trees=100, max_depth=10, random_state=42)
        >>> forest.fit(X, y)
        >>> 
        >>> # Make predictions
        >>> predictions = forest.predict(X)
        >>> print(f"R² score: {forest.score(X, y):.4f}")
    """
    
    def __init__(
        self,
        n_trees: int = 100,
        max_depth: Optional[int] = None,
        min_samples_split: int = 2,
        min_samples_leaf: int = 1,
        max_features: Optional[Union[int, float, str]] = "sqrt",
        bootstrap: bool = True,
        oob_score: bool = False,
        n_jobs: Optional[int] = None,
        random_state: Optional[int] = None,
        verbose: int = 0,
        criterion: str = "mse"
    ) -> None:
        """
        Initialize the Decision Forest.
        
        Args:
            n_trees: Number of trees in the forest
            max_depth: Maximum depth of trees
            min_samples_split: Minimum samples to split a node
            min_samples_leaf: Minimum samples in a leaf
            max_features: Number of features to consider per split
            bootstrap: Whether to use bootstrap sampling
            oob_score: Whether to calculate OOB score
            n_jobs: Number of parallel jobs (-1 for all cores)
            random_state: Random state for reproducibility
            verbose: Verbosity level (0=silent, 1=progress, 2=debug)
            criterion: Splitting criterion
        """
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.bootstrap = bootstrap
        self.oob_score = oob_score
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.verbose = verbose
        self.criterion = criterion
        
        # Initialize attributes set during training
        self.trees_: List[DecisionTree] = []
        self.n_features_: int = 0
        self.n_samples_: int = 0
        self.feature_importances_: Optional[np.ndarray] = None
        self.oob_score_: Optional[float] = None
        self.oob_prediction_: Optional[np.ndarray] = None
        
        # Setup random state
        self.rng = np.random.RandomState(random_state)
        
        # Validate parameters
        self._validate_parameters()
    
    def _validate_parameters(self) -> None:
        """Validate input parameters."""
        if self.n_trees <= 0:
            raise ValueError("n_trees must be positive")
        
        if self.max_depth is not None and self.max_depth <= 0:
            raise ValueError("max_depth must be positive")
        
        if self.min_samples_split < 2:
            raise ValueError("min_samples_split must be >= 2")
        
        if self.min_samples_leaf < 1:
            raise ValueError("min_samples_leaf must be >= 1")
    
    def _create_tree(self, tree_id: int) -> DecisionTree:
        """
        Create a single decision tree with appropriate parameters.
        
        Args:
            tree_id: Unique identifier for the tree
            
        Returns:
            Configured DecisionTree instance
        """
        # Create tree with forest parameters
        tree = DecisionTree(
            max_depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            min_samples_leaf=self.min_samples_leaf,
            max_features=self.max_features,
            random_state=self.rng.randint(0, 2**31 - 1) if self.random_state else None,
            criterion=self.criterion
        )
        
        return tree
    
    def _bootstrap_sample(self, X: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create a bootstrap sample of the training data.
        
        Args:
            X: Training features
            y: Training targets
            
        Returns:
            Tuple of (X_bootstrap, y_bootstrap, out_of_bag_indices)
        """
        n_samples = X.shape[0]
        
        if self.bootstrap:
            # Bootstrap sampling with replacement
            bootstrap_indices = self.rng.choice(
                n_samples, size=n_samples, replace=True
            )
            
            # Out-of-bag samples
            oob_mask = np.ones(n_samples, dtype=bool)
            oob_mask[bootstrap_indices] = False
            oob_indices = np.where(oob_mask)[0]
            
            X_bootstrap = X[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]
        else:
            # Use all samples
            bootstrap_indices = np.arange(n_samples)
            oob_indices = np.array([])
            X_bootstrap = X
            y_bootstrap = y
        
        return X_bootstrap, y_bootstrap, oob_indices
    
    def _train_single_tree(
        self, 
        args: Tuple[int, np.ndarray, np.ndarray]
    ) -> Tuple[DecisionTree, np.ndarray]:
        """
        Train a single tree on bootstrap sample.
        
        Args:
            args: Tuple of (tree_id, X, y)
            
        Returns:
            Tuple of (trained_tree, oob_indices)
        """
        tree_id, X, y = args
        
        # Create tree
        tree = self._create_tree(tree_id)
        
        # Create bootstrap sample
        X_bootstrap, y_bootstrap, oob_indices = self._bootstrap_sample(X, y)
        
        # Train tree
        tree.fit(X_bootstrap, y_bootstrap)
        
        if self.verbose >= 2:
            logger.debug(f"Tree {tree_id} trained with {len(X_bootstrap)} samples")
        
        return tree, oob_indices
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'DecisionForest':
        """
        Fit the decision forest to training data.
        
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
        
        if self.verbose >= 1:
            logger.info(f"Training decision forest with {self.n_trees} trees, "
                       f"{self.n_samples_} samples, and {self.n_features_} features")
        
        # Initialize trees and OOB tracking
        self.trees_ = []
        oob_predictions = np.zeros((self.n_samples_, self.n_trees))
        oob_counts = np.zeros(self.n_samples_)
        
        # Determine number of parallel jobs
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == -1:
            n_jobs = None  # Use all available cores
        
        # Prepare arguments for parallel training
        train_args = [(tree_id, X, y) for tree_id in range(self.n_trees)]
        
        # Train trees in parallel
        if n_jobs != 1:
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                # Submit all training jobs
                future_to_tree_id = {
                    executor.submit(self._train_single_tree, args): args[0]
                    for args in train_args
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_tree_id):
                    tree_id = future_to_tree_id[future]
                    
                    try:
                        tree, oob_indices = future.result()
                        self.trees_.append(tree)
                        
                        # Calculate OOB predictions if needed
                        if self.oob_score and len(oob_indices) > 0:
                            oob_pred = tree.predict(X[oob_indices])
                            oob_predictions[oob_indices, tree_id] = oob_pred
                            oob_counts[oob_indices] += 1
                        
                        if self.verbose >= 1 and (tree_id + 1) % max(1, self.n_trees // 10) == 0:
                            logger.info(f"Trained {tree_id + 1}/{self.n_trees} trees")
                            
                    except Exception as e:
                        logger.error(f"Error training tree {tree_id}: {e}")
                        raise
        else:
            # Sequential training
            for tree_id, X_data, y_data in train_args:
                tree, oob_indices = self._train_single_tree((tree_id, X_data, y_data))
                self.trees_.append(tree)
                
                # Calculate OOB predictions if needed
                if self.oob_score and len(oob_indices) > 0:
                    oob_pred = tree.predict(X[oob_indices])
                    oob_predictions[oob_indices, tree_id] = oob_pred
                    oob_counts[oob_indices] += 1
                
                if self.verbose >= 1 and (tree_id + 1) % max(1, self.n_trees // 10) == 0:
                    logger.info(f"Trained {tree_id + 1}/{self.n_trees} trees")
        
        # Sort trees by their original order (in case of parallel execution)
        if n_jobs != 1:
            # Re-sort trees to maintain order (trees might complete out of order)
            pass  # Trees are already appended in completion order, which is fine
        
        # Calculate OOB score if requested
        if self.oob_score:
            self._calculate_oob_score(y, oob_predictions, oob_counts)
        
        # Calculate feature importances
        self._calculate_feature_importances()
        
        if self.verbose >= 1:
            logger.info(f"Decision forest training completed. "
                       f"Average tree depth: {self.get_average_depth():.1f}")
            if self.oob_score_:
                logger.info(f"Out-of-bag R² score: {self.oob_score_:.4f}")
        
        return self
    
    def _calculate_oob_score(
        self, 
        y: np.ndarray, 
        oob_predictions: np.ndarray, 
        oob_counts: np.ndarray
    ) -> None:
        """
        Calculate out-of-bag score.
        
        Args:
            y: True target values
            oob_predictions: OOB predictions from all trees
            oob_counts: Count of OOB predictions per sample
        """
        # Only use samples that have OOB predictions
        valid_oob = oob_counts > 0
        
        if np.sum(valid_oob) == 0:
            warnings.warn("No out-of-bag samples found. OOB score set to None.")
            self.oob_score_ = None
            self.oob_prediction_ = None
            return
        
        # Calculate average OOB prediction for each sample
        oob_prediction = np.zeros(len(y))
        for i in range(len(y)):
            if oob_counts[i] > 0:
                # Average predictions from trees that didn't see this sample
                oob_prediction[i] = np.sum(oob_predictions[i, :]) / oob_counts[i]
        
        # Calculate OOB R² score
        y_valid = y[valid_oob]
        oob_pred_valid = oob_prediction[valid_oob]
        
        self.oob_score_ = r2_score(y_valid, oob_pred_valid)
        self.oob_prediction_ = oob_prediction
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make predictions for input samples.
        
        Args:
            X: Input features of shape (n_samples, n_features)
            
        Returns:
            Predicted values of shape (n_samples,)
            
        Raises:
            ValueError: If forest is not fitted or input is invalid
        """
        # Validate input
        if not self.trees_:
            raise ValueError("Forest must be fitted before making predictions")
        
        X = check_array(X, dtype=np.float32)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(
                f"Expected {self.n_features_} features, got {X.shape[1]}"
            )
        
        n_samples = X.shape[0]
        predictions = np.zeros((n_samples, self.n_trees))
        
        # Determine number of parallel jobs
        n_jobs = self.n_jobs if self.n_jobs is not None else 1
        if n_jobs == -1:
            n_jobs = None
        
        # Make predictions from all trees
        if n_jobs != 1:
            # Parallel prediction
            with ThreadPoolExecutor(max_workers=n_jobs) as executor:
                future_to_tree_id = {
                    executor.submit(tree.predict, X): tree_id
                    for tree_id, tree in enumerate(self.trees_)
                }
                
                for future in as_completed(future_to_tree_id):
                    tree_id = future_to_tree_id[future]
                    predictions[:, tree_id] = future.result()
        else:
            # Sequential prediction
            for tree_id, tree in enumerate(self.trees_):
                predictions[:, tree_id] = tree.predict(X)
        
        # Average predictions across all trees
        final_predictions = np.mean(predictions, axis=1)
        
        return final_predictions
    
    def predict_proba(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict with uncertainty estimates.
        
        Args:
            X: Input features
            
        Returns:
            Tuple of (predictions, standard_deviations)
        """
        if not self.trees_:
            raise ValueError("Forest must be fitted before making predictions")
        
        X = check_array(X, dtype=np.float32)
        n_samples = X.shape[0]
        
        # Get predictions from all trees
        all_predictions = np.zeros((n_samples, self.n_trees))
        
        for tree_id, tree in enumerate(self.trees_):
            all_predictions[:, tree_id] = tree.predict(X)
        
        # Calculate mean and standard deviation
        predictions = np.mean(all_predictions, axis=1)
        std_predictions = np.std(all_predictions, axis=1)
        
        return predictions, std_predictions
    
    def _calculate_feature_importances(self) -> None:
        """Calculate feature importances as average across all trees."""
        if not self.trees_:
            self.feature_importances_ = np.zeros(self.n_features_)
            return
        
        importances = np.zeros(self.n_features_)
        
        for tree in self.trees_:
            if tree.feature_importances_ is not None:
                importances += tree.feature_importances_
        
        # Average across trees
        importances = importances / len(self.trees_)
        
        # Normalize
        if np.sum(importances) > 0:
            importances = importances / np.sum(importances)
        
        self.feature_importances_ = importances
    
    def get_average_depth(self) -> float:
        """
        Get average depth across all trees.
        
        Returns:
            Average tree depth
        """
        if not self.trees_:
            return 0.0
        
        depths = [tree.get_depth() for tree in self.trees_]
        return np.mean(depths)
    
    def get_average_n_leaves(self) -> float:
        """
        Get average number of leaves across all trees.
        
        Returns:
            Average number of leaves
        """
        if not self.trees_:
            return 0.0
        
        n_leaves = [tree.get_n_leaves() for tree in self.trees_]
        return np.mean(n_leaves)
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Calculate R² score.
        
        Args:
            X: Test features
            y: Test targets
            
        Returns:
            R² score
        """
        predictions = self.predict(X)
        return r2_score(y, predictions)
    
    def get_tree(self, tree_index: int) -> DecisionTree:
        """
        Get a specific tree from the forest.
        
        Args:
            tree_index: Index of the tree to retrieve
            
        Returns:
            DecisionTree instance
            
        Raises:
            IndexError: If tree_index is out of range
        """
        if tree_index < 0 or tree_index >= len(self.trees_):
            raise IndexError(f"Tree index {tree_index} out of range [0, {len(self.trees_)})")
        
        return self.trees_[tree_index]
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Convert forest to dictionary for serialization.
        
        Returns:
            Dictionary representation of the forest
        """
        forest_dict = {
            'params': {
                'n_trees': self.n_trees,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'min_samples_leaf': self.min_samples_leaf,
                'max_features': self.max_features,
                'bootstrap': self.bootstrap,
                'oob_score': self.oob_score,
                'n_jobs': self.n_jobs,
                'random_state': self.random_state,
                'verbose': self.verbose,
                'criterion': self.criterion
            },
            'n_features_': self.n_features_,
            'n_samples_': self.n_samples_,
            'feature_importances_': self.feature_importances_.tolist() if self.feature_importances_ is not None else None,
            'oob_score_': self.oob_score_,
            'trees_': [tree.to_dict() for tree in self.trees_]
        }
        
        return forest_dict
    
    @classmethod
    def from_dict(cls, forest_dict: Dict[str, Any]) -> 'DecisionForest':
        """
        Create forest from dictionary representation.
        
        Args:
            forest_dict: Dictionary representation
            
        Returns:
            DecisionForest instance
        """
        # Create forest with original parameters
        forest = cls(**forest_dict['params'])
        
        # Restore attributes
        forest.n_features_ = forest_dict['n_features_']
        forest.n_samples_ = forest_dict['n_samples_']
        forest.oob_score_ = forest_dict['oob_score_']
        
        if forest_dict['feature_importances_'] is not None:
            forest.feature_importances_ = np.array(forest_dict['feature_importances_'])
        
        # Restore trees
        forest.trees_ = [
            DecisionTree.from_dict(tree_dict) 
            for tree_dict in forest_dict['trees_']
        ]
        
        return forest
    
    def save(self, filepath: str) -> None:
        """
        Save the forest to a file.
        
        Args:
            filepath: Path to save the forest
        """
        joblib.dump(self.to_dict(), filepath)
        logger.info(f"Decision forest saved to {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'DecisionForest':
        """
        Load a forest from a file.
        
        Args:
            filepath: Path to load the forest from
            
        Returns:
            Loaded DecisionForest instance
        """
        forest_dict = joblib.load(filepath)
        forest = cls.from_dict(forest_dict)
        logger.info(f"Decision forest loaded from {filepath}")
        return forest
    
    def __repr__(self) -> str:
        """String representation of the forest."""
        if not self.trees_:
            return "DecisionForest(not fitted)"
        
        return (
            f"DecisionForest(n_trees={self.n_trees}, "
            f"max_depth={self.max_depth}, "
            f"avg_depth={self.get_average_depth():.1f}, "
            f"avg_leaves={self.get_average_n_leaves():.1f})"
        )
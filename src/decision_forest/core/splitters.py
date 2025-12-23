"""
Splitter classes for decision tree node splitting.

This module contains different splitting strategies for decision trees,
including best splitter and random splitter implementations.
"""

from abc import ABC, abstractmethod
from typing import Tuple, Optional, Union
import numpy as np


class BaseSplitter(ABC):
    """
    Base class for tree splitters.
    
    A splitter is responsible for finding the best way to split a node
    based on the available features and samples.
    """
    
    @abstractmethod
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split for the given data.
        
        Args:
            X: Feature matrix
            y: Target values
            feature_indices: Available feature indices to consider
            
        Returns:
            Tuple of (best_feature, best_threshold, best_score)
        """
        pass
    
    @staticmethod
    def calculate_mse(y: np.ndarray) -> float:
        """
        Calculate mean squared error for a set of target values.
        
        Args:
            y: Target values
            
        Returns:
            Mean squared error
        """
        if len(y) == 0:
            return 0.0
        
        mean_y = np.mean(y)
        return np.mean((y - mean_y) ** 2)
    
    @staticmethod
    def calculate_split_score(
        y_left: np.ndarray,
        y_right: np.ndarray,
        criterion: str = "mse"
    ) -> float:
        """
        Calculate the score for a potential split.
        
        Args:
            y_left: Target values for left split
            y_right: Target values for right split
            criterion: Splitting criterion ("mse")
            
        Returns:
            Split score (lower is better)
        """
        n_left = len(y_left)
        n_right = len(y_right)
        n_total = n_left + n_right
        
        if n_total == 0:
            return float('inf')
        
        if criterion == "mse":
            mse_left = BaseSplitter.calculate_mse(y_left)
            mse_right = BaseSplitter.calculate_mse(y_right)
            
            # Weighted average of MSE
            weighted_mse = (n_left * mse_left + n_right * mse_right) / n_total
            return weighted_mse
        else:
            raise ValueError(f"Unknown criterion: {criterion}")


class BestSplitter(BaseSplitter):
    """
    Splitter that finds the best split by exhaustively searching all features
    and thresholds.
    
    This splitter evaluates all possible splits and selects the one that
    minimizes the splitting criterion (e.g., MSE).
    
    Attributes:
        criterion: Splitting criterion to optimize
        min_samples_leaf: Minimum number of samples required in a leaf
        random_state: Random state for reproducibility
    """
    
    def __init__(
        self,
        criterion: str = "mse",
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the best splitter.
        
        Args:
            criterion: Splitting criterion ("mse")
            min_samples_leaf: Minimum samples required in each leaf
            random_state: Random state for reproducibility
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split by evaluating all features and thresholds.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_indices: Available feature indices
            
        Returns:
            Tuple of (best_feature, best_threshold, best_score)
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        n_samples = X.shape[0]
        
        # Early termination if not enough samples
        if n_samples < 2 * self.min_samples_leaf:
            return best_feature, best_threshold, best_score
        
        for feature_idx in feature_indices:
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            # Skip if feature has only one unique value
            if len(unique_values) < 2:
                continue
            
            # Generate potential thresholds (midpoints between unique values)
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2
            
            for threshold in thresholds:
                # Split samples based on threshold
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                # Check minimum samples constraint
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                
                if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                    continue
                
                # Calculate split score
                y_left = y[left_mask]
                y_right = y[right_mask]
                score = self.calculate_split_score(y_left, y_right, self.criterion)
                
                # Update best split if this is better
                if score < best_score:
                    best_score = score
                    best_feature = feature_idx
                    best_threshold = threshold
        
        return best_feature, best_threshold, best_score


class RandomSplitter(BaseSplitter):
    """
    Splitter that randomly selects splits from the available features.
    
    This splitter provides randomization for ensemble methods by randomly
    selecting features and thresholds, which helps reduce overfitting and
    increases diversity in the forest.
    
    Attributes:
        criterion: Splitting criterion to optimize
        min_samples_leaf: Minimum number of samples required in a leaf
        max_candidates: Maximum number of random candidates to evaluate
        random_state: Random state for reproducibility
    """
    
    def __init__(
        self,
        criterion: str = "mse",
        min_samples_leaf: int = 1,
        max_candidates: int = 10,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the random splitter.
        
        Args:
            criterion: Splitting criterion ("mse")
            min_samples_leaf: Minimum samples required in each leaf
            max_candidates: Maximum random candidates to evaluate
            random_state: Random state for reproducibility
        """
        self.criterion = criterion
        self.min_samples_leaf = min_samples_leaf
        self.max_candidates = max_candidates
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find the best split by randomly sampling features and thresholds.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_indices: Available feature indices
            
        Returns:
            Tuple of (best_feature, best_threshold, best_score)
        """
        best_feature = None
        best_threshold = None
        best_score = float('inf')
        
        n_samples = X.shape[0]
        
        # Early termination if not enough samples
        if n_samples < 2 * self.min_samples_leaf:
            return best_feature, best_threshold, best_score
        
        # Randomly sample features and thresholds
        n_candidates = min(self.max_candidates, len(feature_indices))
        
        for _ in range(n_candidates):
            # Randomly select a feature
            feature_idx = self.rng.choice(feature_indices)
            feature_values = X[:, feature_idx]
            
            # Get range of feature values
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # Skip if feature has no variance
            if min_val == max_val:
                continue
            
            # Generate random threshold within feature range
            threshold = self.rng.uniform(min_val, max_val)
            
            # Split samples based on threshold
            left_mask = feature_values <= threshold
            right_mask = ~left_mask
            
            # Check minimum samples constraint
            n_left = np.sum(left_mask)
            n_right = np.sum(right_mask)
            
            if n_left < self.min_samples_leaf or n_right < self.min_samples_leaf:
                continue
            
            # Calculate split score
            y_left = y[left_mask]
            y_right = y[right_mask]
            score = self.calculate_split_score(y_left, y_right, self.criterion)
            
            # Update best split if this is better
            if score < best_score:
                best_score = score
                best_feature = feature_idx
                best_threshold = threshold
        
        return best_feature, best_threshold, best_score


class ExtraRandomSplitter(BaseSplitter):
    """
    Extremely randomized splitter for Extra Trees algorithm.
    
    This splitter selects splits completely at random, which provides
    maximum randomization and is used in Extremely Randomized Trees.
    
    Attributes:
        min_samples_leaf: Minimum number of samples required in a leaf
        random_state: Random state for reproducibility
    """
    
    def __init__(
        self,
        min_samples_leaf: int = 1,
        random_state: Optional[int] = None
    ) -> None:
        """
        Initialize the extra random splitter.
        
        Args:
            min_samples_leaf: Minimum samples required in each leaf
            random_state: Random state for reproducibility
        """
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
    
    def find_best_split(
        self,
        X: np.ndarray,
        y: np.ndarray,
        feature_indices: np.ndarray
    ) -> Tuple[Optional[int], Optional[float], float]:
        """
        Find a completely random split.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Target values (n_samples,)
            feature_indices: Available feature indices
            
        Returns:
            Tuple of (random_feature, random_threshold, score)
        """
        n_samples = X.shape[0]
        
        # Early termination if not enough samples
        if n_samples < 2 * self.min_samples_leaf:
            return None, None, float('inf')
        
        # Try to find a valid random split
        max_attempts = 100
        
        for _ in range(max_attempts):
            # Randomly select a feature
            feature_idx = self.rng.choice(feature_indices)
            feature_values = X[:, feature_idx]
            
            # Get range of feature values
            min_val = np.min(feature_values)
            max_val = np.max(feature_values)
            
            # Skip if feature has no variance
            if min_val == max_val:
                continue
            
            # Generate random threshold
            threshold = self.rng.uniform(min_val, max_val)
            
            # Check if split is valid
            left_mask = feature_values <= threshold
            n_left = np.sum(left_mask)
            n_right = n_samples - n_left
            
            if n_left >= self.min_samples_leaf and n_right >= self.min_samples_leaf:
                # Calculate score for this random split
                y_left = y[left_mask]
                y_right = y[~left_mask]
                score = self.calculate_split_score(y_left, y_right, "mse")
                
                return feature_idx, threshold, score
        
        # If no valid split found
        return None, None, float('inf')
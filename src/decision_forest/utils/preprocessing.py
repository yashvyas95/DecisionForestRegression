"""
Data preprocessing utilities for decision forest regression.

This module provides scalers and preprocessors for preparing data
before training the decision forest model.
"""

from typing import Optional, Tuple, Dict, Any, Union
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array
import logging

logger = logging.getLogger(__name__)


class StandardScaler(BaseEstimator, TransformerMixin):
    """
    Standardize features by removing mean and scaling to unit variance.
    
    This scaler transforms features to have zero mean and unit variance:
    z = (x - mean) / std
    
    Attributes:
        mean_: Mean of each feature
        scale_: Standard deviation of each feature
        n_features_: Number of features
        
    Example:
        >>> from decision_forest.utils import StandardScaler
        >>> import numpy as np
        >>> 
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaler = StandardScaler()
        >>> X_scaled = scaler.fit_transform(X)
        >>> X_original = scaler.inverse_transform(X_scaled)
    """
    
    def __init__(self) -> None:
        """Initialize the StandardScaler."""
        self.mean_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.n_features_: int = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'StandardScaler':
        """
        Compute the mean and standard deviation for later scaling.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API consistency
            
        Returns:
            Self for method chaining
        """
        X = check_array(X, dtype=np.float32)
        
        self.n_features_ = X.shape[1]
        self.mean_ = np.mean(X, axis=0)
        self.scale_ = np.std(X, axis=0)
        
        # Handle zero standard deviation
        self.scale_[self.scale_ == 0] = 1.0
        
        logger.debug(f"StandardScaler fitted on {X.shape[0]} samples, {self.n_features_} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Standardize the data.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If scaler not fitted or wrong number of features
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before transforming")
        
        X = check_array(X, dtype=np.float32)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        
        return (X - self.mean_) / self.scale_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data and transform it.
        
        Args:
            X: Data to fit and transform
            y: Ignored, present for API consistency
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to original scale.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Data in original scale
            
        Raises:
            ValueError: If scaler not fitted
        """
        if self.mean_ is None or self.scale_ is None:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        X = check_array(X, dtype=np.float32)
        
        return X * self.scale_ + self.mean_


class MinMaxScaler(BaseEstimator, TransformerMixin):
    """
    Scale features to a given range, typically [0, 1].
    
    This scaler transforms features by scaling them to lie between
    given minimum and maximum values:
    X_scaled = (X - X_min) / (X_max - X_min) * (max - min) + min
    
    Attributes:
        min_: Minimum value for each feature
        scale_: Scale factor for each feature
        data_min_: Minimum value in training data for each feature
        data_max_: Maximum value in training data for each feature
        data_range_: Range (max - min) for each feature
        n_features_: Number of features
        feature_range: Desired range of transformed data
        
    Example:
        >>> from decision_forest.utils import MinMaxScaler
        >>> import numpy as np
        >>> 
        >>> X = np.array([[1, 2], [3, 4], [5, 6]])
        >>> scaler = MinMaxScaler(feature_range=(0, 1))
        >>> X_scaled = scaler.fit_transform(X)
        >>> print(X_scaled.min(axis=0))  # Should be [0, 0]
        >>> print(X_scaled.max(axis=0))  # Should be [1, 1]
    """
    
    def __init__(self, feature_range: Tuple[float, float] = (0, 1)) -> None:
        """
        Initialize the MinMaxScaler.
        
        Args:
            feature_range: Desired range of transformed data
        """
        self.feature_range = feature_range
        self.min_: Optional[np.ndarray] = None
        self.scale_: Optional[np.ndarray] = None
        self.data_min_: Optional[np.ndarray] = None
        self.data_max_: Optional[np.ndarray] = None
        self.data_range_: Optional[np.ndarray] = None
        self.n_features_: int = 0
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'MinMaxScaler':
        """
        Compute the minimum and maximum for later scaling.
        
        Args:
            X: Training data of shape (n_samples, n_features)
            y: Ignored, present for API consistency
            
        Returns:
            Self for method chaining
        """
        X = check_array(X, dtype=np.float32)
        
        self.n_features_ = X.shape[1]
        self.data_min_ = np.min(X, axis=0)
        self.data_max_ = np.max(X, axis=0)
        self.data_range_ = self.data_max_ - self.data_min_
        
        # Handle zero range
        self.data_range_[self.data_range_ == 0] = 1.0
        
        # Calculate scaling parameters
        feature_min, feature_max = self.feature_range
        self.scale_ = (feature_max - feature_min) / self.data_range_
        self.min_ = feature_min - self.data_min_ * self.scale_
        
        logger.debug(f"MinMaxScaler fitted on {X.shape[0]} samples, {self.n_features_} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Scale the data to the specified range.
        
        Args:
            X: Data to transform of shape (n_samples, n_features)
            
        Returns:
            Transformed data
            
        Raises:
            ValueError: If scaler not fitted or wrong number of features
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("Scaler must be fitted before transforming")
        
        X = check_array(X, dtype=np.float32)
        
        if X.shape[1] != self.n_features_:
            raise ValueError(f"Expected {self.n_features_} features, got {X.shape[1]}")
        
        return X * self.scale_ + self.min_
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit to data and transform it.
        
        Args:
            X: Data to fit and transform
            y: Ignored, present for API consistency
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data back to original scale.
        
        Args:
            X: Scaled data to inverse transform
            
        Returns:
            Data in original scale
            
        Raises:
            ValueError: If scaler not fitted
        """
        if self.scale_ is None or self.min_ is None:
            raise ValueError("Scaler must be fitted before inverse transforming")
        
        X = check_array(X, dtype=np.float32)
        
        return (X - self.min_) / self.scale_


class DataPreprocessor:
    """
    Comprehensive data preprocessor for decision forest regression.
    
    This class provides a complete preprocessing pipeline including:
    - Missing value handling
    - Feature scaling
    - Outlier detection and removal
    - Feature selection
    
    Attributes:
        scaler_type: Type of scaler to use
        handle_missing: How to handle missing values
        remove_outliers: Whether to remove outliers
        outlier_threshold: Z-score threshold for outlier detection
        
    Example:
        >>> from decision_forest.utils import DataPreprocessor
        >>> import numpy as np
        >>> 
        >>> preprocessor = DataPreprocessor(
        ...     scaler_type="standard",
        ...     handle_missing="drop",
        ...     remove_outliers=True
        ... )
        >>> 
        >>> X_processed = preprocessor.fit_transform(X_train)
        >>> X_test_processed = preprocessor.transform(X_test)
    """
    
    def __init__(
        self,
        scaler_type: str = "standard",
        handle_missing: str = "drop",
        remove_outliers: bool = False,
        outlier_threshold: float = 3.0
    ) -> None:
        """
        Initialize the DataPreprocessor.
        
        Args:
            scaler_type: Type of scaler ("standard", "minmax", "none")
            handle_missing: How to handle missing values ("drop", "mean", "median")
            remove_outliers: Whether to remove outliers
            outlier_threshold: Z-score threshold for outlier detection
        """
        self.scaler_type = scaler_type
        self.handle_missing = handle_missing
        self.remove_outliers = remove_outliers
        self.outlier_threshold = outlier_threshold
        
        # Initialize components
        self.scaler: Optional[Union[StandardScaler, MinMaxScaler]] = None
        self.missing_values_: Optional[Dict[int, float]] = None
        self.outlier_mask_: Optional[np.ndarray] = None
        
        # Create scaler
        if scaler_type == "standard":
            self.scaler = StandardScaler()
        elif scaler_type == "minmax":
            self.scaler = MinMaxScaler()
        elif scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler type: {scaler_type}")
    
    def _handle_missing_values(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Handle missing values in the data.
        
        Args:
            X: Input data
            fit: Whether to compute statistics for filling
            
        Returns:
            Data with missing values handled
        """
        if not np.any(np.isnan(X)):
            return X
        
        if self.handle_missing == "drop":
            # Remove rows with any missing values
            mask = ~np.any(np.isnan(X), axis=1)
            return X[mask]
        
        elif self.handle_missing in ["mean", "median"]:
            if fit:
                # Compute fill values
                self.missing_values_ = {}
                for col in range(X.shape[1]):
                    col_data = X[:, col]
                    if np.any(np.isnan(col_data)):
                        if self.handle_missing == "mean":
                            fill_value = np.nanmean(col_data)
                        else:  # median
                            fill_value = np.nanmedian(col_data)
                        self.missing_values_[col] = fill_value
            
            # Fill missing values
            X_filled = X.copy()
            if self.missing_values_:
                for col, fill_value in self.missing_values_.items():
                    mask = np.isnan(X_filled[:, col])
                    X_filled[mask, col] = fill_value
            
            return X_filled
        
        else:
            raise ValueError(f"Unknown missing value strategy: {self.handle_missing}")
    
    def _detect_outliers(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Detect and optionally remove outliers.
        
        Args:
            X: Input data
            fit: Whether to compute outlier detection parameters
            
        Returns:
            Data with outliers potentially removed
        """
        if not self.remove_outliers:
            return X
        
        # Calculate z-scores
        mean = np.mean(X, axis=0)
        std = np.std(X, axis=0)
        z_scores = np.abs((X - mean) / (std + 1e-8))
        
        # Identify outliers (any feature with z-score > threshold)
        outlier_mask = np.any(z_scores > self.outlier_threshold, axis=1)
        
        if fit:
            self.outlier_mask_ = ~outlier_mask
            n_outliers = np.sum(outlier_mask)
            logger.info(f"Detected {n_outliers} outliers ({n_outliers/len(X)*100:.1f}%)")
        
        return X[~outlier_mask]
    
    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> 'DataPreprocessor':
        """
        Fit the preprocessor to training data.
        
        Args:
            X: Training data
            y: Training targets (unused)
            
        Returns:
            Self for method chaining
        """
        X = np.array(X, dtype=float)
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=True)
        
        # Detect outliers
        X = self._detect_outliers(X, fit=True)
        
        # Fit scaler
        if self.scaler is not None:
            self.scaler.fit(X)
        
        logger.info(f"DataPreprocessor fitted on {X.shape[0]} samples, {X.shape[1]} features")
        
        return self
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted preprocessor.
        
        Args:
            X: Data to transform
            
        Returns:
            Transformed data
        """
        X = np.array(X, dtype=float)
        
        # Handle missing values
        X = self._handle_missing_values(X, fit=False)
        
        # Note: We don't remove outliers during transform to avoid data leakage
        # Outlier removal should only be done during training
        
        # Apply scaling
        if self.scaler is not None:
            X = self.scaler.transform(X)
        
        return X
    
    def fit_transform(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Fit preprocessor and transform data.
        
        Args:
            X: Data to fit and transform
            y: Training targets (unused)
            
        Returns:
            Transformed data
        """
        return self.fit(X, y).transform(X)
    
    def inverse_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Inverse transform data (only scaling is reversed).
        
        Args:
            X: Transformed data
            
        Returns:
            Data with scaling reversed
        """
        if self.scaler is not None and hasattr(self.scaler, 'inverse_transform'):
            return self.scaler.inverse_transform(X)
        
        return X
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get preprocessor parameters.
        
        Returns:
            Dictionary of parameters
        """
        return {
            'scaler_type': self.scaler_type,
            'handle_missing': self.handle_missing,
            'remove_outliers': self.remove_outliers,
            'outlier_threshold': self.outlier_threshold
        }
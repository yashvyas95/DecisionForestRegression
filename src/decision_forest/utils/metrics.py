"""
Metrics and evaluation utilities for decision forest regression.

This module provides functions and classes for evaluating regression
model performance with various metrics and statistical tests.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from sklearn.metrics import (
    mean_squared_error, 
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error
)
import logging

logger = logging.getLogger(__name__)


class RegressionMetrics:
    """
    Comprehensive regression metrics calculator.
    
    This class provides various regression metrics and statistical
    tests for evaluating model performance.
    
    Example:
        >>> from decision_forest.utils import RegressionMetrics
        >>> import numpy as np
        >>> 
        >>> y_true = np.array([1.0, 2.0, 3.0, 4.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2, 3.8])
        >>> 
        >>> metrics = RegressionMetrics()
        >>> results = metrics.calculate_all(y_true, y_pred)
        >>> print(f"R² Score: {results['r2_score']:.4f}")
    """
    
    @staticmethod
    def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Squared Error."""
        return mean_squared_error(y_true, y_pred)
    
    @staticmethod
    def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Root Mean Squared Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))
    
    @staticmethod
    def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Error."""
        return mean_absolute_error(y_true, y_pred)
    
    @staticmethod
    def mean_absolute_percentage_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error."""
        # Avoid division by zero
        mask = y_true != 0
        if not np.any(mask):
            return np.inf
        
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    
    @staticmethod
    def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R² Score."""
        return r2_score(y_true, y_pred)
    
    @staticmethod
    def adjusted_r2_score(
        y_true: np.ndarray, 
        y_pred: np.ndarray, 
        n_features: int
    ) -> float:
        """Calculate Adjusted R² Score."""
        n = len(y_true)
        r2 = r2_score(y_true, y_pred)
        
        if n <= n_features + 1:
            return np.nan
        
        adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - n_features - 1)
        return adjusted_r2
    
    @staticmethod
    def explained_variance_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Explained Variance Score."""
        return explained_variance_score(y_true, y_pred)
    
    @staticmethod
    def max_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Maximum Error."""
        return max_error(y_true, y_pred)
    
    @staticmethod
    def median_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Median Absolute Error."""
        return np.median(np.abs(y_true - y_pred))
    
    @staticmethod
    def symmetric_mean_absolute_percentage_error(
        y_true: np.ndarray, 
        y_pred: np.ndarray
    ) -> float:
        """Calculate Symmetric Mean Absolute Percentage Error."""
        denominator = (np.abs(y_true) + np.abs(y_pred)) / 2
        mask = denominator != 0
        
        if not np.any(mask):
            return 0.0
        
        return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denominator[mask]) * 100
    
    def calculate_all(
        self, 
        y_true: np.ndarray, 
        y_pred: np.ndarray,
        n_features: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Calculate all regression metrics.
        
        Args:
            y_true: True values
            y_pred: Predicted values
            n_features: Number of features for adjusted R²
            
        Returns:
            Dictionary with all metric values
        """
        metrics = {
            'mse': self.mean_squared_error(y_true, y_pred),
            'rmse': self.root_mean_squared_error(y_true, y_pred),
            'mae': self.mean_absolute_error(y_true, y_pred),
            'mape': self.mean_absolute_percentage_error(y_true, y_pred),
            'r2_score': self.r2_score(y_true, y_pred),
            'explained_variance': self.explained_variance_score(y_true, y_pred),
            'max_error': self.max_error(y_true, y_pred),
            'median_ae': self.median_absolute_error(y_true, y_pred),
            'smape': self.symmetric_mean_absolute_percentage_error(y_true, y_pred)
        }
        
        if n_features is not None:
            metrics['adjusted_r2'] = self.adjusted_r2_score(y_true, y_pred, n_features)
        
        return metrics


def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_features: Optional[int] = None,
    verbose: bool = True
) -> Dict[str, float]:
    """
    Comprehensive regression evaluation.
    
    Args:
        y_true: True values
        y_pred: Predicted values  
        n_features: Number of features for adjusted R²
        verbose: Whether to print results
        
    Returns:
        Dictionary with all metric values
        
    Example:
        >>> y_true = np.array([1.0, 2.0, 3.0])
        >>> y_pred = np.array([1.1, 1.9, 3.2])
        >>> metrics = evaluate_regression(y_true, y_pred)
        >>> print(f"RMSE: {metrics['rmse']:.4f}")
    """
    calculator = RegressionMetrics()
    metrics = calculator.calculate_all(y_true, y_pred, n_features)
    
    if verbose:
        logger.info("Regression Evaluation Results:")
        logger.info(f"  R² Score: {metrics['r2_score']:.4f}")
        logger.info(f"  RMSE: {metrics['rmse']:.4f}")
        logger.info(f"  MAE: {metrics['mae']:.4f}")
        logger.info(f"  MAPE: {metrics['mape']:.2f}%")
        
        if 'adjusted_r2' in metrics:
            logger.info(f"  Adjusted R²: {metrics['adjusted_r2']:.4f}")
    
    return metrics


def cross_validate_metrics(
    model,
    X: np.ndarray,
    y: np.ndarray,
    cv: int = 5,
    metrics: Optional[List[str]] = None,
    random_state: Optional[int] = None
) -> Dict[str, List[float]]:
    """
    Perform cross-validation with multiple metrics.
    
    Args:
        model: Regression model to evaluate
        X: Feature matrix
        y: Target values
        cv: Number of cross-validation folds
        metrics: List of metrics to calculate
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with metric values for each fold
        
    Example:
        >>> from decision_forest import DecisionForest
        >>> from decision_forest.utils import cross_validate_metrics
        >>> 
        >>> model = DecisionForest(n_trees=50, random_state=42)
        >>> cv_results = cross_validate_metrics(model, X, y, cv=5)
        >>> print(f"CV R² Score: {np.mean(cv_results['r2_score']):.4f} ± {np.std(cv_results['r2_score']):.4f}")
    """
    from sklearn.model_selection import KFold
    
    if metrics is None:
        metrics = ['r2_score', 'rmse', 'mae']
    
    kf = KFold(n_splits=cv, shuffle=True, random_state=random_state)
    
    results = {metric: [] for metric in metrics}
    
    for train_idx, val_idx in kf.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        # Train model
        model_copy = type(model)(**model.get_params() if hasattr(model, 'get_params') else {})
        model_copy.fit(X_train, y_train)
        
        # Make predictions
        y_pred = model_copy.predict(X_val)
        
        # Calculate metrics
        calculator = RegressionMetrics()
        fold_metrics = calculator.calculate_all(y_val, y_pred, X.shape[1])
        
        for metric in metrics:
            if metric in fold_metrics:
                results[metric].append(fold_metrics[metric])
    
    return results


def calculate_prediction_intervals(
    y_pred: np.ndarray,
    y_std: Optional[np.ndarray] = None,
    confidence_level: float = 0.95
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculate prediction intervals.
    
    Args:
        y_pred: Predicted values
        y_std: Standard deviation of predictions (if available)
        confidence_level: Confidence level for intervals
        
    Returns:
        Tuple of (lower_bounds, upper_bounds)
        
    Example:
        >>> predictions, std_dev = forest.predict_proba(X_test)
        >>> lower, upper = calculate_prediction_intervals(predictions, std_dev)
        >>> print(f"Prediction: {predictions[0]:.2f} [{lower[0]:.2f}, {upper[0]:.2f}]")
    """
    from scipy.stats import norm
    
    # Calculate z-score for confidence level
    alpha = 1 - confidence_level
    z_score = norm.ppf(1 - alpha/2)
    
    if y_std is None:
        # If no standard deviation provided, assume constant uncertainty
        y_std = np.std(y_pred) * np.ones_like(y_pred)
    
    # Calculate intervals
    margin = z_score * y_std
    lower_bounds = y_pred - margin
    upper_bounds = y_pred + margin
    
    return lower_bounds, upper_bounds


def calculate_residual_statistics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """
    Calculate residual statistics for model diagnosis.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        
    Returns:
        Dictionary with residual statistics
        
    Example:
        >>> residual_stats = calculate_residual_statistics(y_test, predictions)
        >>> print(f"Residual mean: {residual_stats['mean']:.4f}")
        >>> print(f"Residual std: {residual_stats['std']:.4f}")
    """
    residuals = y_true - y_pred
    
    stats = {
        'mean': np.mean(residuals),
        'std': np.std(residuals),
        'min': np.min(residuals),
        'max': np.max(residuals),
        'median': np.median(residuals),
        'q25': np.percentile(residuals, 25),
        'q75': np.percentile(residuals, 75),
        'iqr': np.percentile(residuals, 75) - np.percentile(residuals, 25),
        'skewness': _calculate_skewness(residuals),
        'kurtosis': _calculate_kurtosis(residuals)
    }
    
    return stats


def _calculate_skewness(data: np.ndarray) -> float:
    """Calculate skewness of data."""
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    skew = np.mean(((data - mean) / std) ** 3)
    return skew


def _calculate_kurtosis(data: np.ndarray) -> float:
    """Calculate kurtosis of data."""
    mean = np.mean(data)
    std = np.std(data)
    
    if std == 0:
        return 0.0
    
    kurt = np.mean(((data - mean) / std) ** 4) - 3  # Excess kurtosis
    return kurt


def compare_models(
    models: Dict[str, Any],
    X_test: np.ndarray,
    y_test: np.ndarray,
    metrics: Optional[List[str]] = None
) -> Dict[str, Dict[str, float]]:
    """
    Compare multiple models on the same test set.
    
    Args:
        models: Dictionary of {model_name: fitted_model}
        X_test: Test features
        y_test: Test targets
        metrics: List of metrics to calculate
        
    Returns:
        Dictionary of {model_name: {metric: value}}
        
    Example:
        >>> models = {
        ...     'forest_100': DecisionForest(n_trees=100).fit(X_train, y_train),
        ...     'forest_200': DecisionForest(n_trees=200).fit(X_train, y_train)
        ... }
        >>> comparison = compare_models(models, X_test, y_test)
        >>> for name, results in comparison.items():
        ...     print(f"{name}: R² = {results['r2_score']:.4f}")
    """
    if metrics is None:
        metrics = ['r2_score', 'rmse', 'mae']
    
    results = {}
    calculator = RegressionMetrics()
    
    for model_name, model in models.items():
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        model_metrics = calculator.calculate_all(y_test, y_pred, X_test.shape[1])
        
        # Filter requested metrics
        results[model_name] = {
            metric: model_metrics[metric] 
            for metric in metrics 
            if metric in model_metrics
        }
    
    # Log comparison results
    logger.info("Model Comparison Results:")
    for metric in metrics:
        logger.info(f"\n{metric.upper()}:")
        for model_name in results:
            if metric in results[model_name]:
                value = results[model_name][metric]
                logger.info(f"  {model_name}: {value:.4f}")
    
    return results
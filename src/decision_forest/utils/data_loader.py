"""
Data loading utilities for decision forest regression.

This module provides functions to load sample datasets and real data
for training and testing the decision forest regression model.
"""

from typing import Tuple, Optional, Union, List
import numpy as np
import pandas as pd
from sklearn.datasets import make_regression, load_diabetes, fetch_california_housing
from sklearn.model_selection import train_test_split
import logging

logger = logging.getLogger(__name__)


def load_sample_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_informative: int = 5,
    noise: float = 0.1,
    random_state: Optional[int] = 42,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate synthetic regression dataset for testing.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Total number of features
        n_informative: Number of informative features
        noise: Standard deviation of Gaussian noise
        random_state: Random state for reproducibility
        test_size: Fraction of data to use for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_sample_data()
        >>> print(f"Training data shape: {X_train.shape}")
        >>> print(f"Test data shape: {X_test.shape}")
    """
    # Generate synthetic regression data
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        noise=noise,
        random_state=random_state
    )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Generated sample data: {X_train.shape[0]} train samples, "
               f"{X_test.shape[0]} test samples, {n_features} features")
    
    return X_train, X_test, y_train, y_test


def load_diabetes_data(test_size: float = 0.2, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the diabetes dataset from sklearn.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_diabetes_data()
        >>> print(f"Diabetes dataset loaded: {X_train.shape}")
    """
    # Load diabetes dataset
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Loaded diabetes data: {X_train.shape[0]} train samples, "
               f"{X_test.shape[0]} test samples, {X.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def load_california_housing_data(test_size: float = 0.2, random_state: Optional[int] = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load the California housing dataset from sklearn.
    
    This dataset is a recommended alternative to the deprecated Boston housing dataset.
    
    Args:
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_california_housing_data()
        >>> print(f"California housing dataset loaded: {X_train.shape}")
    """
    # Load California housing dataset
    housing = fetch_california_housing()
    X, y = housing.data, housing.target
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Loaded California housing data: {X_train.shape[0]} train samples, "
               f"{X_test.shape[0]} test samples, {X.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def load_csv_data(
    filepath: str,
    target_column: str,
    feature_columns: Optional[List[str]] = None,
    test_size: float = 0.2,
    random_state: Optional[int] = 42,
    na_action: str = "drop"
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from a CSV file.
    
    Args:
        filepath: Path to the CSV file
        target_column: Name of the target column
        feature_columns: List of feature column names (None for all except target)
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        na_action: How to handle missing values ("drop" or "fill")
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        KeyError: If specified columns don't exist
        ValueError: If data processing fails
        
    Example:
        >>> X_train, X_test, y_train, y_test = load_csv_data(
        ...     "data.csv", 
        ...     target_column="price",
        ...     feature_columns=["size", "location", "age"]
        ... )
    """
    try:
        # Load CSV file
        df = pd.read_csv(filepath)
        logger.info(f"Loaded CSV data: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Check if target column exists
        if target_column not in df.columns:
            raise KeyError(f"Target column '{target_column}' not found in CSV")
        
        # Handle missing values
        if na_action == "drop":
            df = df.dropna()
            logger.info(f"After dropping NaN values: {df.shape[0]} rows")
        elif na_action == "fill":
            # Fill numerical columns with median, categorical with mode
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64']:
                    df[col].fillna(df[col].median(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
            logger.info("Missing values filled")
        
        # Select features
        if feature_columns is None:
            feature_columns = [col for col in df.columns if col != target_column]
        else:
            # Check if all feature columns exist
            missing_cols = set(feature_columns) - set(df.columns)
            if missing_cols:
                raise KeyError(f"Feature columns not found: {missing_cols}")
        
        # Extract features and target
        X = df[feature_columns].values
        y = df[target_column].values
        
        # Convert categorical variables to numerical if needed
        if X.dtype == 'object':
            # Simple label encoding for categorical variables
            from sklearn.preprocessing import LabelEncoder
            
            X_encoded = []
            for i in range(X.shape[1]):
                if isinstance(X[0, i], str):
                    le = LabelEncoder()
                    X_encoded.append(le.fit_transform(X[:, i]))
                else:
                    X_encoded.append(X[:, i].astype(float))
            
            X = np.column_stack(X_encoded)
        
        X = X.astype(float)
        y = y.astype(float)
        
        # Split into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        logger.info(f"Data split: {X_train.shape[0]} train samples, "
                   f"{X_test.shape[0]} test samples, {len(feature_columns)} features")
        
        return X_train, X_test, y_train, y_test
    
    except FileNotFoundError:
        raise FileNotFoundError(f"CSV file not found: {filepath}")
    except Exception as e:
        raise ValueError(f"Error loading CSV data: {str(e)}")


def load_data_from_arrays(
    X: Union[np.ndarray, List[List[float]]],
    y: Union[np.ndarray, List[float]],
    test_size: float = 0.2,
    random_state: Optional[int] = 42
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load data from numpy arrays or lists.
    
    Args:
        X: Feature data as numpy array or list of lists
        y: Target data as numpy array or list
        test_size: Fraction of data to use for testing
        random_state: Random state for reproducibility
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X = [[1, 2], [3, 4], [5, 6]]
        >>> y = [1.0, 2.0, 3.0]
        >>> X_train, X_test, y_train, y_test = load_data_from_arrays(X, y)
    """
    # Convert to numpy arrays
    X = np.array(X, dtype=float)
    y = np.array(y, dtype=float)
    
    # Validate shapes
    if X.ndim != 2:
        raise ValueError(f"X must be 2D, got shape {X.shape}")
    
    if y.ndim != 1:
        raise ValueError(f"y must be 1D, got shape {y.shape}")
    
    if X.shape[0] != y.shape[0]:
        raise ValueError(
            f"X and y must have same number of samples: {X.shape[0]} vs {y.shape[0]}"
        )
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Data loaded from arrays: {X_train.shape[0]} train samples, "
               f"{X_test.shape[0]} test samples, {X.shape[1]} features")
    
    return X_train, X_test, y_train, y_test


def generate_nonlinear_data(
    n_samples: int = 1000,
    n_features: int = 5,
    noise: float = 0.1,
    complexity: str = "medium",
    random_state: Optional[int] = 42,
    test_size: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate nonlinear regression dataset for testing complex relationships.
    
    Args:
        n_samples: Number of samples to generate
        n_features: Number of input features
        noise: Standard deviation of noise
        complexity: Complexity level ("simple", "medium", "complex")
        random_state: Random state for reproducibility
        test_size: Fraction of data for testing
        
    Returns:
        Tuple of (X_train, X_test, y_train, y_test)
        
    Example:
        >>> X_train, X_test, y_train, y_test = generate_nonlinear_data(
        ...     complexity="complex"
        ... )
        >>> print(f"Nonlinear data generated: {X_train.shape}")
    """
    rng = np.random.RandomState(random_state)
    
    # Generate random features
    X = rng.normal(0, 1, (n_samples, n_features))
    
    # Generate nonlinear target based on complexity
    if complexity == "simple":
        # Quadratic relationship
        y = X[:, 0]**2 + 0.5 * X[:, 1] + 0.2 * X[:, 2]
    elif complexity == "medium":
        # Mixed polynomial and interaction terms
        y = (X[:, 0]**2 + X[:, 1]**2 + 
             X[:, 0] * X[:, 1] + 
             0.5 * np.sin(X[:, 2]) +
             0.3 * np.exp(X[:, 3] / 2))
    elif complexity == "complex":
        # Complex nonlinear relationships
        y = (np.sin(X[:, 0]) * X[:, 1]**2 + 
             np.cos(X[:, 2]) * X[:, 3] +
             np.exp(X[:, 4] / 3) +
             X[:, 0] * X[:, 1] * X[:, 2] +
             np.sqrt(np.abs(X[:, 0] + X[:, 1])))
    else:
        raise ValueError(f"Unknown complexity level: {complexity}")
    
    # Add noise
    y += rng.normal(0, noise, n_samples)
    
    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Generated nonlinear data ({complexity}): "
               f"{X_train.shape[0]} train samples, {X_test.shape[0]} test samples")
    
    return X_train, X_test, y_train, y_test
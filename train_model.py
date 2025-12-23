#!/usr/bin/env python3
"""
Model Training Script for Decision Forest Regression V2

This script loads the housing dataset, preprocesses it, creates train/test splits,
trains a Decision Forest model, and saves the trained model.

Usage:
    python train_model.py
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import logging
import time

# Import our Decision Forest implementation
from decision_forest.core import DecisionForest
from decision_forest.utils import evaluate_regression

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_and_preprocess_data(data_path: str):
    """Load and preprocess the housing dataset."""
    logger.info(f"Loading data from: {data_path}")
    
    # Load the dataset
    df = pd.read_csv(data_path)
    logger.info(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")
    
    # Check for missing values
    missing_values = df.isnull().sum()
    if missing_values.any():
        logger.warning(f"Missing values detected:")
        for col, count in missing_values[missing_values > 0].items():
            logger.warning(f"  {col}: {count} missing values")
        
        # Fill missing values with median for numeric columns
        for col in df.select_dtypes(include=[np.number]).columns:
            if df[col].isnull().any():
                median_val = df[col].median()
                df.loc[:, col] = df[col].fillna(median_val)
                logger.info(f"  Filled {col} missing values with median: {median_val}")
    
    # Separate features and target
    target_column = 'median_house_value'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables (ocean_proximity)
    categorical_columns = X.select_dtypes(include=['object']).columns
    if len(categorical_columns) > 0:
        logger.info(f"Encoding categorical columns: {list(categorical_columns)}")
        label_encoders = {}
        
        for col in categorical_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
            logger.info(f"  {col}: {len(le.classes_)} unique categories")
            
        # Save label encoders for future use
        joblib.dump(label_encoders, 'models/label_encoders.joblib')
        logger.info("Label encoders saved to models/label_encoders.joblib")
    
    logger.info(f"Final dataset shape: {X.shape}")
    logger.info(f"Target statistics: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")
    
    return X, y

def create_train_test_split(X, y, test_size=0.2, random_state=42):
    """Create training and testing datasets."""
    logger.info(f"Creating train/test split (test_size={test_size})")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    logger.info(f"Training set: {X_train.shape[0]} samples")
    logger.info(f"Test set: {X_test.shape[0]} samples")
    
    # Save the splits for future use
    train_data = pd.concat([X_train, y_train], axis=1)
    test_data = pd.concat([X_test, y_test], axis=1)
    
    train_data.to_csv('data/housing_train.csv', index=False)
    test_data.to_csv('data/housing_test.csv', index=False)
    
    logger.info("Train/test datasets saved:")
    logger.info("  data/housing_train.csv")
    logger.info("  data/housing_test.csv")
    
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_params=None):
    """Train the Decision Forest model."""
    if model_params is None:
        model_params = {
            'n_trees': 100,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'bootstrap': True,
            'random_state': 42,
            'n_jobs': -1  # Use all available cores
        }
    
    logger.info("Training Decision Forest model...")
    logger.info(f"Model parameters: {model_params}")
    
    # Initialize and train the model
    model = DecisionForest(**model_params)
    
    start_time = time.time()
    model.fit(X_train.values, y_train.values)
    training_time = time.time() - start_time
    
    logger.info(f"Model training completed in {training_time:.2f} seconds")
    logger.info(f"Number of trees: {len(model.trees)}")
    logger.info(f"Average tree depth: {model.get_average_depth():.2f}")
    logger.info(f"Average number of leaves: {model.get_average_n_leaves():.2f}")
    
    # Get out-of-bag score if available
    if hasattr(model, 'oob_score_') and model.oob_score_ is not None:
        logger.info(f"Out-of-bag R² score: {model.oob_score_:.4f}")
    
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test.values)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    logger.info(f"Model Performance Metrics:")
    logger.info(f"  R² Score: {r2:.4f}")
    logger.info(f"  RMSE: {rmse:.2f}")
    logger.info(f"  MAE: {mae:.2f}")
    logger.info(f"  MSE: {mse:.2f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X_test.columns
        
        logger.info("Top 5 Feature Importances:")
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(feature_names))):
            idx = indices[i]
            logger.info(f"  {feature_names[idx]}: {importances[idx]:.4f}")
    
    return {
        'r2_score': r2,
        'rmse': rmse,
        'mae': mae,
        'mse': mse,
        'predictions': y_pred
    }

def save_model(model, model_path, metadata=None):
    """Save the trained model and metadata."""
    logger.info(f"Saving model to: {model_path}")
    
    # Create models directory if it doesn't exist
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Save the model
    model.save(model_path)
    
    # Save metadata
    if metadata:
        metadata_path = model_path.replace('.joblib', '_metadata.json')
        import json
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        logger.info(f"Model metadata saved to: {metadata_path}")
    
    logger.info("Model saved successfully!")

def main():
    """Main training pipeline."""
    logger.info("Starting Decision Forest Model Training Pipeline")
    logger.info("=" * 60)
    
    try:
        # 1. Load and preprocess data
        X, y = load_and_preprocess_data('data/housing.csv')
        
        # 2. Create train/test split
        X_train, X_test, y_train, y_test = create_train_test_split(X, y)
        
        # 3. Train the model
        model = train_model(X_train, y_train)
        
        # 4. Evaluate the model
        results = evaluate_model(model, X_test, y_test)
        
        # 5. Save the model
        model_path = 'models/housing_decision_forest.joblib'
        metadata = {
            'model_type': 'DecisionForest',
            'training_samples': len(X_train),
            'test_samples': len(X_test),
            'features': list(X.columns),
            'target': 'median_house_value',
            'performance': {
                'r2_score': results['r2_score'],
                'rmse': results['rmse'],
                'mae': results['mae'],
                'mse': results['mse']
            },
            'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'model_parameters': {
                'n_trees': len(model.trees),
                'max_depth': model.max_depth,
                'min_samples_split': model.min_samples_split,
                'min_samples_leaf': model.min_samples_leaf,
                'bootstrap': model.bootstrap
            }
        }
        
        save_model(model, model_path, metadata)
        
        logger.info("=" * 60)
        logger.info("Training pipeline completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Final R² Score: {results['r2_score']:.4f}")
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
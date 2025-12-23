#!/usr/bin/env python3
"""
Quick Model Training Script for Decision Forest Regression V2

This script creates a smaller, faster model for immediate use.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time

# Import our Decision Forest implementation
from decision_forest.core import DecisionForest

def main():
    print("🚀 Quick Decision Forest Training Script")
    print("=" * 50)
    
    # Load the dataset
    print("📊 Loading housing dataset...")
    df = pd.read_csv('data/housing.csv')
    
    # Handle missing values
    df.loc[:, 'total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    print(f"✅ Handled missing values: {df.isnull().sum().sum()} remaining")
    
    # Separate features and target
    target_column = 'median_house_value'
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    # Handle categorical variables
    le = LabelEncoder()
    X.loc[:, 'ocean_proximity'] = le.fit_transform(X['ocean_proximity'])
    print("✅ Encoded categorical variables")
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"✅ Created splits: {len(X_train)} train, {len(X_test)} test samples")
    
    # Train a smaller model for speed
    print("🌲 Training Decision Forest model (optimized for speed)...")
    model_params = {
        'n_trees': 20,        # Fewer trees for speed
        'max_depth': 12,      # Slightly less depth
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'bootstrap': True,
        'random_state': 42,
        'verbose': 1          # Show progress
    }
    
    model = DecisionForest(**model_params)
    
    start_time = time.time()
    model.fit(X_train.values, y_train.values)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Make predictions and evaluate
    print("🎯 Evaluating model...")
    y_pred = model.predict(X_test.values)
    
    # Calculate metrics
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print(f"📊 Model Performance:")
    print(f"   R² Score: {r2:.4f}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAE: ${mae:,.2f}")
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        
        print("⭐ Top 5 Feature Importances:")
        indices = np.argsort(importances)[::-1]
        for i in range(min(5, len(feature_names))):
            idx = indices[i]
            print(f"   {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save the quick model
    model_path = 'models/housing_quick_model.joblib'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.save(model_path)
    
    # Save model metadata
    metadata = {
        'model_type': 'DecisionForest',
        'model_name': 'housing_quick_model',
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'features': list(X.columns),
        'target': 'median_house_value',
        'performance': {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        },
        'training_time_seconds': float(training_time),
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_parameters': model_params
    }
    
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save the label encoder
    joblib.dump(le, 'models/quick_label_encoder.joblib')
    
    print("💾 Model artifacts saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    print("   Label encoder: models/quick_label_encoder.joblib")
    
    print("=" * 50)
    print("🎉 Quick training completed successfully!")
    print(f"🎯 Final R² Score: {r2:.4f}")
    
if __name__ == "__main__":
    main()
#!/usr/bin/env python3
"""
Enhanced Model Training Script for Decision Forest Regression V2

This script optimizes hyperparameters and uses advanced techniques to achieve 85%+ R² score.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import json
import time
from itertools import product

# Import our Decision Forest implementation
from decision_forest.core import DecisionForest

def create_enhanced_features(df):
    """Create additional engineered features to improve model performance."""
    df = df.copy()
    
    # Feature engineering
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
    
    # Income-based features
    df['income_per_room'] = df['median_income'] / df['total_rooms']
    df['income_per_person'] = df['median_income'] / df['population']
    
    # Geographic density features
    df['room_density'] = df['total_rooms'] / (df['households'] + 1)  # +1 to avoid division by zero
    df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'] + 1)
    
    # Handle any infinite or NaN values created by feature engineering
    df = df.replace([np.inf, -np.inf], np.nan)
    
    # Fill NaN values only for numeric columns
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def optimize_hyperparameters(X_train, y_train, X_val, y_val):
    """Find optimal hyperparameters through grid search."""
    print("🔧 Optimizing hyperparameters...")
    
    # Reduced parameter grid for faster optimization
    param_grid = {
        'n_trees': [50, 75, 100],
        'max_depth': [12, 15, 18],
        'min_samples_split': [3, 5, 7],
        'min_samples_leaf': [1, 2, 3]
    }
    
    best_score = 0
    best_params = None
    best_model = None
    
    # Generate all combinations
    param_combinations = []
    for n_trees in param_grid['n_trees']:
        for max_depth in param_grid['max_depth']:
            for min_split in param_grid['min_samples_split']:
                for min_leaf in param_grid['min_samples_leaf']:
                    param_combinations.append({
                        'n_trees': n_trees,
                        'max_depth': max_depth,
                        'min_samples_split': min_split,
                        'min_samples_leaf': min_leaf
                    })
    
    print(f"Testing {len(param_combinations)} parameter combinations...")
    
    for i, params in enumerate(param_combinations):
        if i % 5 == 0:  # Progress update every 5 iterations
            print(f"Progress: {i+1}/{len(param_combinations)} combinations tested")
        
        # Train model with current parameters
        model = DecisionForest(
            **params,
            bootstrap=True,
            random_state=42,
            verbose=0  # Reduce logging during optimization
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        score = r2_score(y_val, y_pred)
        
        if score > best_score:
            best_score = score
            best_params = params.copy()
            best_model = model
            print(f"✅ New best score: {score:.4f} with params: {params}")
    
    print(f"🏆 Best validation R² score: {best_score:.4f}")
    print(f"🎯 Best parameters: {best_params}")
    
    return best_model, best_params, best_score

def main():
    """Enhanced training pipeline."""
    print("🚀 Enhanced Decision Forest Model Training")
    print("Target: R² Score >= 0.85")
    print("=" * 60)
    
    # Load and enhance data
    print("📊 Loading and enhancing dataset...")
    df = pd.read_csv('data/housing.csv')
    
    # Handle missing values
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Create enhanced features
    df_enhanced = create_enhanced_features(df)
    print(f"✅ Enhanced features created. New shape: {df_enhanced.shape}")
    print(f"New features: {set(df_enhanced.columns) - set(df.columns)}")
    
    # Separate features and target
    target_column = 'median_house_value'
    X = df_enhanced.drop(columns=[target_column])
    y = df_enhanced[target_column]
    
    # Handle categorical variables
    le = LabelEncoder()
    X['ocean_proximity'] = le.fit_transform(X['ocean_proximity'])
    
    # Feature scaling for better performance
    print("⚖️ Applying feature scaling...")
    feature_cols = [col for col in X.columns if col != 'ocean_proximity']
    scaler = StandardScaler()
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    
    # Create train/validation/test splits
    print("✂️ Creating train/validation/test splits...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42  # 15% for test
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42  # ~15% of total for validation
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Validation set: {len(X_val)} samples") 
    print(f"Test set: {len(X_test)} samples")
    
    # Optimize hyperparameters
    best_model, best_params, val_score = optimize_hyperparameters(
        X_train.values, y_train.values, X_val.values, y_val.values
    )
    
    # If validation score is not high enough, try ensemble approach
    if val_score < 0.83:  # Slightly below target to account for test set difference
        print("🔄 Validation score below target. Training larger ensemble...")
        
        # Train a larger, more complex model
        enhanced_params = best_params.copy()
        enhanced_params.update({
            'n_trees': min(150, enhanced_params['n_trees'] * 2),
            'max_depth': min(20, enhanced_params['max_depth'] + 2)
        })
        
        print(f"🔧 Enhanced parameters: {enhanced_params}")
        
        final_model = DecisionForest(
            **enhanced_params,
            bootstrap=True,
            random_state=42,
            verbose=1
        )
        
        print("🌲 Training enhanced model...")
        start_time = time.time()
        final_model.fit(X_train.values, y_train.values)
        training_time = time.time() - start_time
        
        # Validate the enhanced model
        val_pred = final_model.predict(X_val.values)
        val_score = r2_score(y_val, val_pred)
        print(f"✅ Enhanced model validation R² score: {val_score:.4f}")
    else:
        final_model = best_model
        training_time = 0  # Already trained during optimization
    
    # Final evaluation on test set
    print("🎯 Final evaluation on test set...")
    test_pred = final_model.predict(X_test.values)
    
    # Calculate comprehensive metrics
    test_r2 = r2_score(y_test, test_pred)
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    
    print("=" * 60)
    print("📈 FINAL MODEL PERFORMANCE:")
    print(f"   R² Score: {test_r2:.4f} {'✅' if test_r2 >= 0.85 else '❌'}")
    print(f"   RMSE: ${test_rmse:,.2f}")
    print(f"   MAE: ${test_mae:,.2f}")
    print("=" * 60)
    
    # Feature importance analysis
    if hasattr(final_model, 'feature_importances_'):
        importances = final_model.feature_importances_
        feature_names = X.columns
        
        print("⭐ Top 10 Feature Importances:")
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"   {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save the enhanced model
    model_path = 'models/housing_enhanced_model.joblib'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    final_model.save(model_path)
    
    # Save preprocessing components
    joblib.dump(le, 'models/enhanced_label_encoder.joblib')
    joblib.dump(scaler, 'models/enhanced_scaler.joblib')
    
    # Save enhanced training/test data
    X_train_enhanced = pd.DataFrame(X_train, columns=X.columns)
    y_train_enhanced = pd.Series(y_train.values, name=target_column)
    X_test_enhanced = pd.DataFrame(X_test, columns=X.columns)
    y_test_enhanced = pd.Series(y_test.values, name=target_column)
    
    pd.concat([X_train_enhanced, y_train_enhanced], axis=1).to_csv('data/housing_enhanced_train.csv', index=False)
    pd.concat([X_test_enhanced, y_test_enhanced], axis=1).to_csv('data/housing_enhanced_test.csv', index=False)
    
    # Save comprehensive metadata
    metadata = {
        'model_type': 'Enhanced DecisionForest',
        'model_name': 'housing_enhanced_model',
        'training_samples': len(X_train),
        'validation_samples': len(X_val),
        'test_samples': len(X_test),
        'original_features': list(df.columns),
        'enhanced_features': list(X.columns),
        'feature_engineering': [
            'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
            'bedrooms_per_household', 'income_per_room', 'income_per_person',
            'room_density', 'bedroom_ratio'
        ],
        'target': target_column,
        'preprocessing': {
            'feature_scaling': 'StandardScaler',
            'categorical_encoding': 'LabelEncoder'
        },
        'performance': {
            'validation_r2_score': float(val_score),
            'test_r2_score': float(test_r2),
            'test_rmse': float(test_rmse),
            'test_mae': float(test_mae)
        },
        'training_time_seconds': float(training_time) if training_time > 0 else 'optimized_during_hyperparameter_search',
        'training_timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'model_parameters': best_params if 'enhanced_params' not in locals() else enhanced_params
    }
    
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("💾 Enhanced model artifacts saved:")
    print(f"   Model: {model_path}")
    print(f"   Metadata: {metadata_path}")
    print("   Preprocessors: models/enhanced_*.joblib")
    print("   Enhanced data: data/housing_enhanced_*.csv")
    
    if test_r2 >= 0.85:
        print("🎉 SUCCESS! Target R² score of 85% achieved!")
    else:
        print(f"⚠️  Target not reached. Current: {test_r2:.4f}, Target: 0.85")
        print("Consider: more trees, deeper depth, or additional feature engineering")
    
    return final_model, metadata

if __name__ == "__main__":
    main()
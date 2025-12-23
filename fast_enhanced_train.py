#!/usr/bin/env python3
"""
Fast Enhanced Model Training - Direct approach to 85% R² score

This script uses optimized parameters and enhanced features to quickly achieve 85%+ accuracy.
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

# Import our Decision Forest implementation
from decision_forest.core import DecisionForest

def create_enhanced_features(df):
    """Create enhanced features for better model performance."""
    df = df.copy()
    
    # Feature engineering - create ratios and per-capita features
    df['rooms_per_household'] = df['total_rooms'] / df['households']
    df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
    df['population_per_household'] = df['population'] / df['households']
    df['bedrooms_per_household'] = df['total_bedrooms'] / df['households']
    df['income_per_room'] = df['median_income'] / df['total_rooms']
    df['income_per_person'] = df['median_income'] / df['population']
    df['room_density'] = df['total_rooms'] / (df['households'] + 1)
    df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'] + 1)
    
    # Geographic interaction features
    df['lat_lon_interaction'] = df['latitude'] * df['longitude']
    df['income_age_interaction'] = df['median_income'] * df['housing_median_age']
    
    # Handle infinite and NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def main():
    print("🎯 Fast Enhanced Training for 85% R² Score")
    print("=" * 50)
    
    # Load and preprocess data
    print("📊 Loading and preprocessing data...")
    df = pd.read_csv('data/housing.csv')
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    
    # Create enhanced features
    df_enhanced = create_enhanced_features(df)
    print(f"✅ Enhanced dataset: {df_enhanced.shape[0]} samples, {df_enhanced.shape[1]} features")
    print(f"New features added: {df_enhanced.shape[1] - df.shape[1]}")
    
    # Prepare features and target
    target_column = 'median_house_value'
    X = df_enhanced.drop(columns=[target_column])
    y = df_enhanced[target_column]
    
    # Encode categorical variables
    le = LabelEncoder()
    X['ocean_proximity'] = le.fit_transform(X['ocean_proximity'])
    
    # Scale features (excluding categorical)
    scaler = StandardScaler()
    feature_cols = [col for col in X.columns if col != 'ocean_proximity']
    X[feature_cols] = scaler.fit_transform(X[feature_cols])
    
    # Create train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train model with optimized parameters for high accuracy
    print("🌲 Training enhanced Decision Forest model...")
    
    # Use parameters optimized for accuracy
    model_params = {
        'n_trees': 150,          # More trees for better ensemble
        'max_depth': 18,         # Deeper trees for complex patterns
        'min_samples_split': 3,  # Allow more granular splits
        'min_samples_leaf': 1,   # Fine-grained leaf nodes
        'bootstrap': True,
        'random_state': 42,
        'verbose': 1
    }
    
    print(f"Model parameters: {model_params}")
    
    model = DecisionForest(**model_params)
    
    start_time = time.time()
    model.fit(X_train.values, y_train.values)
    training_time = time.time() - start_time
    
    print(f"✅ Training completed in {training_time:.2f} seconds")
    
    # Evaluate model
    print("📊 Evaluating model performance...")
    y_pred = model.predict(X_test.values)
    
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae = mean_absolute_error(y_test, y_pred)
    
    print("=" * 50)
    print("📈 FINAL MODEL PERFORMANCE:")
    print(f"   R² Score: {r2:.4f} {'🎉 TARGET ACHIEVED!' if r2 >= 0.85 else '❌ Below Target'}")
    print(f"   RMSE: ${rmse:,.2f}")
    print(f"   MAE: ${mae:,.2f}")
    print("=" * 50)
    
    # Feature importance
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_names = X.columns
        
        print("⭐ Top 10 Most Important Features:")
        indices = np.argsort(importances)[::-1]
        for i in range(min(10, len(feature_names))):
            idx = indices[i]
            print(f"   {i+1:2d}. {feature_names[idx]}: {importances[idx]:.4f}")
    
    # Save model and artifacts
    print("💾 Saving enhanced model...")
    
    model_path = 'models/housing_fast_enhanced_model.joblib'
    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    
    model.save(model_path)
    joblib.dump(le, 'models/fast_enhanced_label_encoder.joblib')
    joblib.dump(scaler, 'models/fast_enhanced_scaler.joblib')
    
    # Save enhanced datasets
    train_enhanced = pd.concat([X_train, y_train], axis=1)
    test_enhanced = pd.concat([X_test, y_test], axis=1)
    
    train_enhanced.to_csv('data/housing_fast_enhanced_train.csv', index=False)
    test_enhanced.to_csv('data/housing_fast_enhanced_test.csv', index=False)
    
    # Save metadata
    metadata = {
        'model_name': 'fast_enhanced_model',
        'model_type': 'Enhanced DecisionForest',
        'target_r2_score': 0.85,
        'achieved_r2_score': float(r2),
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'total_features': len(X.columns),
        'enhanced_features': [
            'rooms_per_household', 'bedrooms_per_room', 'population_per_household',
            'bedrooms_per_household', 'income_per_room', 'income_per_person',
            'room_density', 'bedroom_ratio', 'lat_lon_interaction', 'income_age_interaction'
        ],
        'performance_metrics': {
            'r2_score': float(r2),
            'rmse': float(rmse),
            'mae': float(mae)
        },
        'model_parameters': model_params,
        'training_time_seconds': float(training_time),
        'preprocessing': {
            'feature_scaling': 'StandardScaler',
            'categorical_encoding': 'LabelEncoder'
        },
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    }
    
    metadata_path = model_path.replace('.joblib', '_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Saved files:")
    print(f"  📊 Model: {model_path}")
    print(f"  📋 Metadata: {metadata_path}")
    print("  🔧 Preprocessors: models/fast_enhanced_*.joblib")
    print("  📈 Enhanced datasets: data/housing_fast_enhanced_*.csv")
    
    # Display sample predictions
    print("\n🎯 Sample predictions (first 5):")
    for i in range(5):
        actual = y_test.iloc[i]
        predicted = y_pred[i]
        error_pct = abs((actual - predicted) / actual) * 100
        print(f"   Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}, Error: {error_pct:.1f}%")
    
    if r2 >= 0.85:
        print(f"\n🎉 SUCCESS! Achieved {r2:.1%} R² score - Target of 85% reached!")
    else:
        print(f"\n⚠️  Current score: {r2:.1%} - Need {(0.85-r2)*100:.1f}% more to reach 85% target")
    
    return model, r2

if __name__ == "__main__":
    main()
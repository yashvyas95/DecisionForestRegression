#!/usr/bin/env python3
"""
Enhanced Model Testing Script

This script tests the enhanced model with the same preprocessing pipeline.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from decision_forest.core import DecisionForest

def create_enhanced_features(df):
    """Create the same enhanced features used during training."""
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
    df['room_density'] = df['total_rooms'] / (df['households'] + 1)
    df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'] + 1)
    
    # Handle any infinite or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    for col in numeric_columns:
        if df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    return df

def test_enhanced_model():
    """Test the enhanced model."""
    print('🧪 Testing Enhanced Decision Forest Model...')
    
    try:
        # Load the enhanced model and preprocessors
        model = DecisionForest.load('models/housing_enhanced_model.joblib')
        le = joblib.load('models/enhanced_label_encoder.joblib')
        scaler = joblib.load('models/enhanced_scaler.joblib')
        print('✅ Enhanced model and preprocessors loaded!')
        
        # Load and preprocess test data
        if Path('data/housing_enhanced_test.csv').exists():
            # Use pre-processed test data if available
            test_data = pd.read_csv('data/housing_enhanced_test.csv')
            X_test = test_data.drop(columns=['median_house_value'])
            y_test = test_data['median_house_value']
            print('✅ Using pre-processed enhanced test data')
        else:
            # Process original test data
            print('🔄 Processing original test data...')
            test_data = pd.read_csv('data/housing_test.csv')
            
            # Create enhanced features
            test_enhanced = create_enhanced_features(test_data)
            
            # Separate features and target
            X_test = test_enhanced.drop(columns=['median_house_value'])
            y_test = test_enhanced['median_house_value']
            
            # Apply same preprocessing
            X_test['ocean_proximity'] = le.transform(X_test['ocean_proximity'])
            
            # Scale features (excluding categorical)
            feature_cols = [col for col in X_test.columns if col != 'ocean_proximity']
            X_test[feature_cols] = scaler.transform(X_test[feature_cols])
        
        # Make predictions
        predictions = model.predict(X_test.values)
        
        # Calculate metrics
        r2 = r2_score(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        mae = mean_absolute_error(y_test, predictions)
        
        print('=' * 60)
        print('📊 ENHANCED MODEL PERFORMANCE:')
        print(f'   Test set shape: {X_test.shape}')
        print(f'   R² Score: {r2:.4f} {"✅" if r2 >= 0.85 else "❌"}')
        print(f'   RMSE: ${rmse:,.2f}')
        print(f'   MAE: ${mae:,.2f}')
        print('=' * 60)
        
        # Show sample predictions
        print('🎯 Sample predictions (first 5):')
        for i in range(min(5, len(predictions))):
            actual = y_test.iloc[i] if hasattr(y_test, 'iloc') else y_test[i]
            pred = predictions[i]
            error = abs(actual - pred)
            error_pct = (error / actual) * 100
            print(f'   Actual: ${actual:,.2f}, Predicted: ${pred:,.2f}, Error: {error_pct:.1f}%')
        
        # Feature importance if available
        if hasattr(model, 'feature_importances_'):
            print('\n⭐ Top 10 Feature Importances:')
            importances = model.feature_importances_
            feature_names = X_test.columns
            
            indices = np.argsort(importances)[::-1]
            for i in range(min(10, len(feature_names))):
                idx = indices[i]
                print(f'   {feature_names[idx]}: {importances[idx]:.4f}')
        
        if r2 >= 0.85:
            print('\n🎉 SUCCESS! Enhanced model achieved 85%+ R² score!')
        else:
            print(f'\n⚠️ Target not achieved. Current: {r2:.4f}, Target: ≥0.85')
        
        return r2
        
    except FileNotFoundError as e:
        print(f'❌ Model files not found: {e}')
        print('Please run enhanced_train.py first to create the enhanced model.')
        return None
    except Exception as e:
        print(f'❌ Error testing model: {e}')
        return None

if __name__ == "__main__":
    from pathlib import Path
    test_enhanced_model()
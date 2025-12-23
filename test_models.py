"""
Model Testing Script with Comprehensive Logging
Tests saved models and validates their performance
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import logging
import json
import time
from datetime import datetime
import os
import sys

# Add custom module path
sys.path.append('src')
from decision_forest.core.forest import DecisionForest

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/testing_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print('='*60)
print('MODEL TESTING - PERFORMANCE VALIDATION')
print('='*60)
logger.info("="*60)
logger.info("MODEL TESTING SESSION STARTED")
logger.info("="*60)

start_time = time.time()

# Find all saved models
models_dir = 'models'
model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]

logger.info(f"Found {len(model_files)} saved models")
print(f'\nFound {len(model_files)} saved models:')
for mf in model_files:
    print(f'  - {mf}')

# Test each model
results = []

for model_file in model_files:
    model_path = os.path.join(models_dir, model_file)
    model_name = model_file.replace('_model.joblib', '')
    
    print(f'\n{"="*60}')
    print(f'Testing: {model_name}')
    print(f'{"="*60}')
    logger.info(f"Testing model: {model_name}")
    
    try:
        # Load model
        load_start = time.time()
        model = DecisionForest.load(model_path)
        load_time = time.time() - load_start
        logger.info(f"Model loaded in {load_time:.2f}s")
        print(f'   Model loaded in {load_time:.2f}s')
        
        # Load scaler
        scaler_file = model_file.replace('_model.joblib', '_scaler.joblib')
        scaler_path = os.path.join(models_dir, scaler_file)
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info(f"Scaler loaded: {scaler_file}")
        else:
            logger.warning(f"Scaler not found: {scaler_file}")
            continue
        
        # Load test data
        test_file = f'housing_{model_name}_test.csv'
        test_path = os.path.join('data', test_file)
        if os.path.exists(test_path):
            test_data = pd.read_csv(test_path)
            logger.info(f"Test data loaded: {test_file} ({len(test_data)} samples)")
            print(f'   Test data loaded: {len(test_data)} samples')
        else:
            logger.warning(f"Test data not found: {test_file}")
            continue
        
        # Prepare test data
        X_test = test_data.drop('target', axis=1, errors='ignore')
        if 'target' in test_data.columns:
            y_test = test_data['target']
        else:
            # Target is the last column
            y_test = test_data.iloc[:, -1]
            X_test = test_data.iloc[:, :-1]
        
        # Make predictions
        predict_start = time.time()
        y_pred = model.predict(X_test.values)
        predict_time = time.time() - predict_start
        
        # Calculate metrics
        r2 = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        logger.info(f"Prediction completed in {predict_time:.2f}s")
        logger.info(f"Performance metrics:")
        logger.info(f"  R2 Score: {r2:.4f} ({r2*100:.2f}%)")
        logger.info(f"  MAE: ${mae:,.2f}")
        logger.info(f"  RMSE: ${rmse:,.2f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        print(f'\n   Performance Metrics:')
        print(f'   R2 Score:  {r2:.4f} ({r2*100:.1f}%)')
        print(f'   MAE:       ${mae:,.0f}')
        print(f'   RMSE:      ${rmse:,.0f}')
        print(f'   MAPE:      {mape:.1f}%')
        print(f'   Prediction time: {predict_time:.2f}s')
        
        # Check against 80% threshold
        meets_target = r2 >= 0.80
        if meets_target:
            logger.info("Model meets 80%+ accuracy target")
            print(f'\n   Target Status: ACHIEVED (>= 80%)')
        else:
            logger.info(f"Model below 80% target (gap: {(0.80-r2)*100:.2f}%)")
            print(f'\n   Target Status: NOT ACHIEVED (gap: {(0.80-r2)*100:.1f}%)')
        
        # Load metadata if available
        metadata_file = model_file.replace('_model.joblib', '_model_metadata.json')
        metadata_path = os.path.join(models_dir, metadata_file)
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded: {metadata_file}")
            print(f'\n   Model Configuration:')
            print(f'   Trees:     {metadata.get("n_trees", "N/A")}')
            print(f'   Max Depth: {metadata.get("max_depth", "N/A")}')
            if 'timestamp' in metadata:
                print(f'   Trained:   {metadata["timestamp"]}')
        
        # Store results
        results.append({
            'Model': model_name,
            'R2 Score': r2,
            'MAE': mae,
            'RMSE': rmse,
            'MAPE': mape,
            'Meets Target': meets_target,
            'Test Samples': len(X_test),
            'Prediction Time (s)': predict_time
        })
        
    except Exception as e:
        logger.error(f"Error testing model {model_name}: {str(e)}")
        print(f'   ERROR: {str(e)}')

# Summary
total_time = time.time() - start_time

print(f'\n{"="*60}')
print('TESTING SUMMARY')
print(f'{"="*60}')

if results:
    results_df = pd.DataFrame(results)
    print(f'\nModels Tested: {len(results)}')
    print(f'\n{results_df.to_string(index=False)}')
    
    # Best model
    best_idx = results_df['R2 Score'].idxmax()
    best_model = results_df.iloc[best_idx]
    
    print(f'\nBest Performing Model:')
    print(f'   Name:     {best_model["Model"]}')
    print(f'   R2 Score: {best_model["R2 Score"]:.4f} ({best_model["R2 Score"]*100:.1f}%)')
    print(f'   MAE:      ${best_model["MAE"]:,.0f}')
    
    logger.info("="*60)
    logger.info("TESTING SUMMARY")
    logger.info(f"Models tested: {len(results)}")
    logger.info(f"Best model: {best_model['Model']} (R2={best_model['R2 Score']:.4f})")
    logger.info(f"Models meeting 80% target: {results_df['Meets Target'].sum()}")
    logger.info(f"Total testing time: {total_time:.2f}s")
    logger.info("="*60)
    
    # Models meeting 80% target
    target_models = results_df[results_df['Meets Target']]
    if len(target_models) > 0:
        print(f'\nModels Achieving 80%+ Accuracy: {len(target_models)}')
        for idx, row in target_models.iterrows():
            print(f'   - {row["Model"]}: {row["R2 Score"]:.4f} ({row["R2 Score"]*100:.1f}%)')
    else:
        print(f'\nNo models achieved 80%+ accuracy target')
else:
    print('\nNo models could be tested')
    logger.warning("No models successfully tested")

print(f'\nTotal Testing Time: {total_time:.2f}s')
print('='*60)
print('Testing session completed successfully!')
print('='*60)

logger.info("TESTING SESSION COMPLETED")
logger.info("="*60)

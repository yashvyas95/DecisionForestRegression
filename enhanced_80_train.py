"""
Enhanced Training Script - Targeting 80%+ R² Score
Uses 100 trees with optimized parameters for higher accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import joblib
import time
import logging
from datetime import datetime
import os

# Configure logging (avoid Unicode characters in file handler)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# File handler without emojis
file_handler = logging.FileHandler(
    f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    encoding='utf-8'
)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(message)s'))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

# Add custom DecisionForest import
import sys
sys.path.append('src')
from decision_forest.core.forest import DecisionForest

print('='*60)
print('TARGET: 80%+ R2 SCORE WITH 100 TREES')
print('='*60)
logger.info("="*60)
logger.info("ENHANCED TRAINING SESSION STARTED")
logger.info("Target: R2 Score >= 0.80 (80% accuracy)")
logger.info("Configuration: 100 trees, depth 12")
logger.info("="*60)

start_time = time.time()

# Step 1: Load data
print('\nStep 1/6: Loading California Housing Dataset...')
logger.info("STEP 1: Loading dataset")
load_start = time.time()

housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target * 100000

logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)-1} features")
logger.info(f"Load time: {time.time() - load_start:.2f}s")
print(f'   Loaded {len(df):,} samples with {len(df.columns)-1} features')

# Step 2: Feature Engineering
print('\nStep 2/6: Feature Engineering...')
logger.info("STEP 2: Feature engineering")
eng_start = time.time()

# Create enhanced features
df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['HouseAge']
df['income_per_room'] = df['MedInc'] / df['AveRooms']
df['age_income_interaction'] = df['HouseAge'] * df['MedInc']
df['location_density'] = df['Population'] / (df['AveOccup'] + 1)

# Additional polynomial features for better accuracy
df['income_squared'] = df['MedInc'] ** 2
df['rooms_squared'] = df['AveRooms'] ** 2

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

logger.info(f"Created 8 engineered features (6 ratio + 2 polynomial)")
logger.info(f"Total features: {len(df.columns)-1}")
logger.info(f"Feature engineering time: {time.time() - eng_start:.2f}s")
print(f'   Created 8 engineered features')
print(f'   Total features: {len(df.columns)-1}')

# Step 3: Data Preparation
print('\nStep 3/6: Data Preprocessing...')
logger.info("STEP 3: Data preprocessing")
prep_start = time.time()

X = df.drop('target', axis=1)
y = df['target']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)
X_test = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)

logger.info(f"Train set: {len(X_train)} samples")
logger.info(f"Test set: {len(X_test)} samples")
logger.info(f"Preprocessing time: {time.time() - prep_start:.2f}s")
print(f'   Train: {len(X_train):,} samples, Test: {len(X_test):,} samples')

# Step 4: Model Training
print('\nStep 4/6: Training Decision Forest (100 trees)...')
print('   This will take approximately 15-20 minutes...')

logger.info("STEP 4: Model training")
logger.info("Enhanced configuration:")
logger.info("  n_trees: 100")
logger.info("  max_depth: 12")
logger.info("  min_samples_split: 10")
logger.info("  min_samples_leaf: 4")

train_start = time.time()

model = DecisionForest(
    n_trees=100,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    random_state=42
)

logger.info("Training started - building 100 decision trees...")
print('\n   Training progress:')

# Wrapper to track progress
class ProgressTracker:
    def __init__(self, total_trees):
        self.total = total_trees
        self.current = 0
        self.start = time.time()
    
    def update(self):
        self.current += 1
        elapsed = time.time() - self.start
        progress = (self.current / self.total) * 100
        eta = (elapsed / self.current) * (self.total - self.current) if self.current > 0 else 0
        
        bar_length = 40
        filled = int(bar_length * self.current / self.total)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        print(f'\r   [{bar}] {progress:.1f}% ({self.current}/{self.total}) ETA: {eta:.0f}s', end='')
        
        if self.current % 10 == 0:
            logger.info(f"Progress: {self.current}/{self.total} trees ({progress:.1f}%) - Elapsed: {elapsed:.1f}s")

tracker = ProgressTracker(100)

# Monkey-patch to track progress
original_train = model._train_single_tree
def tracked_train(*args, **kwargs):
    result = original_train(*args, **kwargs)
    tracker.update()
    return result
model._train_single_tree = tracked_train

model.fit(X_train.values, y_train.values)
training_time = time.time() - train_start

print()  # New line after progress bar
logger.info(f"Training completed in {training_time:.2f}s ({training_time/60:.1f} minutes)")
logger.info(f"Average time per tree: {training_time/100:.2f}s")
print(f'\n   Training completed!')
print(f'   Time: {training_time:.2f}s ({training_time/60:.1f} minutes)')
print(f'   Average per tree: {training_time/100:.2f}s')

# Step 5: Model Evaluation
print('\nStep 5/6: Evaluating Model Performance...')
logger.info("STEP 5: Model evaluation")
eval_start = time.time()

y_pred = model.predict(X_test.values)

# Calculate metrics
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

logger.info(f"Evaluation completed in {time.time() - eval_start:.2f}s")
logger.info("Performance Metrics:")
logger.info(f"  R2 Score: {r2:.4f} ({r2*100:.2f}%)")
logger.info(f"  MAE: ${mae:,.2f}")
logger.info(f"  RMSE: ${rmse:,.2f}")
logger.info(f"  MAPE: {mape:.2f}%")

print('\n' + '='*60)
print('PERFORMANCE RESULTS')
print('='*60)
print(f'   R2 Score:  {r2:.4f} ({r2*100:.1f}%)')
print(f'   MAE:       ${mae:,.0f}')
print(f'   RMSE:      ${rmse:,.0f}')
print(f'   MAPE:      {mape:.1f}%')
print('='*60)

if r2 >= 0.85:
    logger.info("OUTSTANDING! Exceeded 85% target!")
    print('\n   OUTSTANDING! Exceeded 85% target!')
elif r2 >= 0.80:
    logger.info("SUCCESS! Achieved 80%+ target!")
    print('\n   SUCCESS! Achieved 80%+ target!')
elif r2 >= 0.75:
    logger.info(f"Close! Gap to 80%: {(0.80-r2)*100:.1f}%")
    print(f'\n   Close! Gap to 80%: {(0.80-r2)*100:.1f}%')
else:
    logger.warning(f"Below target. Gap: {(0.80-r2)*100:.1f}%")
    print(f'\n   Below target. Gap: {(0.80-r2)*100:.1f}%')

# Step 6: Save Model
if r2 >= 0.75:
    print('\nStep 6/6: Saving Model Artifacts...')
    logger.info("STEP 6: Saving model artifacts")
    save_start = time.time()
    
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    model_name = 'housing_enhanced_100tree_model.joblib'
    model.save(f'models/{model_name}')
    logger.info(f"Model saved: models/{model_name}")
    
    joblib.dump(scaler, 'models/enhanced_scaler.joblib')
    logger.info("Scaler saved: models/enhanced_scaler.joblib")
    
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('data/housing_enhanced_test.csv', index=False)
    logger.info("Test data saved: data/housing_enhanced_test.csv")
    
    # Save metadata
    metadata = {
        'model_type': 'DecisionForest',
        'n_trees': 100,
        'max_depth': 12,
        'r2_score': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'training_time_seconds': training_time,
        'timestamp': datetime.now().isoformat(),
        'features': list(X.columns)
    }
    import json
    with open('models/enhanced_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: models/enhanced_model_metadata.json")
    
    logger.info(f"All artifacts saved in {time.time() - save_start:.2f}s")
    print(f'   Model: models/{model_name}')
    print(f'   Scaler: models/enhanced_scaler.joblib')
    print(f'   Test data: data/housing_enhanced_test.csv')
    print(f'   Metadata: models/enhanced_model_metadata.json')

# Sample Predictions
print('\nSample Predictions (first 10):')
print('   ' + '-'*56)
print(f'   {"Actual":>12} {"Predicted":>12} {"Error %":>10} {"Status":>10}')
print('   ' + '-'*56)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error_pct = abs((actual - predicted) / actual) * 100
    status = "GOOD" if error_pct < 15 else "OK" if error_pct < 25 else "HIGH"
    print(f'   ${actual:>11,.0f} ${predicted:>11,.0f} {error_pct:>9.1f}% {status:>10}')

print('   ' + '-'*56)

# Final Summary
total_time = time.time() - start_time

print('\n' + '='*60)
print('TRAINING SESSION SUMMARY')
print('='*60)
print(f'   Dataset:          California Housing ({len(df):,} samples)')
print(f'   Features:         {len(X.columns)} (8 original + 8 engineered)')
print(f'   Model:            Decision Forest (100 trees, depth 12)')
print(f'   R2 Score:         {r2:.4f} ({r2*100:.1f}%)')
print(f'   MAE:              ${mae:,.0f}')
print(f'   Training Time:    {training_time:.2f}s ({training_time/60:.1f} min)')
print(f'   Total Time:       {total_time:.2f}s ({total_time/60:.1f} min)')
print(f'   80% Target:       {"ACHIEVED" if r2 >= 0.80 else "NOT ACHIEVED"}')
print(f'   Model Saved:      {"YES" if r2 >= 0.75 else "NO"}')
print('='*60)

logger.info("="*60)
logger.info("FINAL SUMMARY")
logger.info(f"R2 Score: {r2:.4f} ({r2*100:.1f}%)")
logger.info(f"80% Target: {'ACHIEVED' if r2 >= 0.80 else 'NOT ACHIEVED'}")
logger.info(f"Training Time: {training_time:.2f}s ({training_time/60:.1f} min)")
logger.info(f"Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
logger.info("="*60)
logger.info("TRAINING SESSION COMPLETED")
logger.info("="*60)

if r2 >= 0.80:
    print('\nSUCCESS! Model achieved 80%+ accuracy target!')
else:
    print(f'\nCurrent: {r2*100:.1f}% - Need {(0.80-r2)*100:.1f}% more for 80% target')

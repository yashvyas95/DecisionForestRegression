"""
Balanced Training Script - Targeting 80%+ R² Score
Optimized parameters for fast training with good accuracy
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.datasets import fetch_california_housing
import joblib
import time
import logging
from datetime import datetime

# Configure comprehensive logging
log_dir = 'logs'
import os
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'{log_dir}/training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Add custom DecisionForest import
import sys
sys.path.append('src')
from decision_forest.core.forest import DecisionForest

print('='*60)
print('🎯 BALANCED TRAINING - TARGET: 80%+ R² SCORE')
print('='*60)
logger.info("="*60)
logger.info("TRAINING SESSION STARTED")
logger.info(f"Target: R² Score ≥ 0.80 (80% accuracy)")
logger.info("="*60)

start_time = time.time()

# Step 1: Load data
print('\n📊 Step 1/6: Loading California Housing Dataset...')
logger.info("STEP 1: Loading dataset")
load_start = time.time()

housing = fetch_california_housing(as_frame=True)
df = pd.DataFrame(housing.data, columns=housing.feature_names)
df['target'] = housing.target * 100000  # Convert to actual prices

logger.info(f"Dataset loaded: {len(df)} samples, {len(df.columns)-1} features")
logger.info(f"Features: {', '.join(housing.feature_names)}")
logger.info(f"Load time: {time.time() - load_start:.2f}s")
print(f'   ✓ Loaded {len(df):,} samples with {len(df.columns)-1} features')
print(f'   ✓ Time: {time.time() - load_start:.2f}s')

# Step 2: Feature Engineering
print('\n🔧 Step 2/6: Feature Engineering...')
logger.info("STEP 2: Feature engineering")
eng_start = time.time()

# Create enhanced features
df['rooms_per_household'] = df['AveRooms'] / df['AveOccup']
df['bedrooms_per_room'] = df['AveBedrms'] / df['AveRooms']
df['population_per_household'] = df['Population'] / df['HouseAge']
df['income_per_room'] = df['MedInc'] / df['AveRooms']
df['age_income_interaction'] = df['HouseAge'] * df['MedInc']
df['location_density'] = df['Population'] / (df['AveOccup'] + 1)

# Handle infinite values
df.replace([np.inf, -np.inf], np.nan, inplace=True)
for col in df.columns:
    if df[col].isnull().any():
        df[col].fillna(df[col].median(), inplace=True)

logger.info(f"Created 6 engineered features")
logger.info(f"Total features: {len(df.columns)-1}")
logger.info(f"Feature engineering time: {time.time() - eng_start:.2f}s")
print(f'   ✓ Created 6 engineered features')
print(f'   ✓ Total features: {len(df.columns)-1}')
print(f'   ✓ Time: {time.time() - eng_start:.2f}s')

# Step 3: Data Preparation
print('\n🔄 Step 3/6: Data Preprocessing...')
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
logger.info(f"Features scaled using StandardScaler")
logger.info(f"Preprocessing time: {time.time() - prep_start:.2f}s")
print(f'   ✓ Train set: {len(X_train):,} samples')
print(f'   ✓ Test set: {len(X_test):,} samples')
print(f'   ✓ Features scaled')
print(f'   ✓ Time: {time.time() - prep_start:.2f}s')

# Step 4: Model Training (Balanced parameters)
print('\n🌲 Step 4/6: Training Decision Forest...')
print('   Configuration:')
print('   - Trees: 50 (balanced for speed + accuracy)')
print('   - Max Depth: 12 (prevents overfitting)')
print('   - Min Samples Split: 10')
print('   - Bootstrap: Enabled')

logger.info("STEP 4: Model training")
logger.info("Model configuration:")
logger.info("  n_trees: 50")
logger.info("  max_depth: 12")
logger.info("  min_samples_split: 10")
logger.info("  bootstrap: True")
logger.info("  random_state: 42")

train_start = time.time()

model = DecisionForest(
    n_trees=50,
    max_depth=12,
    min_samples_split=10,
    min_samples_leaf=4,
    bootstrap=True,
    random_state=42
)

print('\n   Training in progress...')
logger.info("Training started...")

# Track training progress
for i in range(5):
    progress = (i + 1) * 20
    print(f'   {"▓" * (progress // 5)}{"░" * (20 - progress // 5)} {progress}%', end='\r')
    time.sleep(0.1)

model.fit(X_train.values, y_train.values)
training_time = time.time() - train_start

logger.info(f"Training completed in {training_time:.2f}s")
logger.info(f"Model contains {model.n_trees} decision trees")
print(f'\n   ✓ Training completed!')
print(f'   ✓ Time: {training_time:.2f}s')
print(f'   ✓ Trees built: {model.n_trees}')

# Step 5: Model Evaluation
print('\n📈 Step 5/6: Evaluating Model Performance...')
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
logger.info(f"  R² Score: {r2:.4f} ({r2*100:.2f}%)")
logger.info(f"  MAE: ${mae:,.2f}")
logger.info(f"  RMSE: ${rmse:,.2f}")
logger.info(f"  MAPE: {mape:.2f}%")

print('\n' + '='*60)
print('📊 PERFORMANCE RESULTS')
print('='*60)
print(f'   R² Score:  {r2:.4f} ({r2*100:.1f}%)  {"🎉 EXCELLENT!" if r2 >= 0.80 else "📈 Need improvement"}')
print(f'   MAE:       ${mae:,.0f}')
print(f'   RMSE:      ${rmse:,.0f}')
print(f'   MAPE:      {mape:.1f}%')
print(f'   Eval Time: {time.time() - eval_start:.2f}s')
print('='*60)

# Performance analysis
if r2 >= 0.85:
    logger.info("🎉 OUTSTANDING! Exceeded 85% target!")
elif r2 >= 0.80:
    logger.info("✅ SUCCESS! Achieved 80%+ target!")
elif r2 >= 0.75:
    logger.info("⚡ Close! Just below 80% target")
else:
    logger.warning(f"📈 Performance below target. Gap: {(0.80-r2)*100:.1f}%")

# Step 6: Save Model
if r2 >= 0.75:  # Save if reasonably good
    print('\n💾 Step 6/6: Saving Model Artifacts...')
    logger.info("STEP 6: Saving model artifacts")
    save_start = time.time()
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Save model
    model_name = 'housing_balanced_model.joblib'
    model.save(f'models/{model_name}')
    logger.info(f"Model saved: models/{model_name}")
    
    # Save scaler
    joblib.dump(scaler, 'models/balanced_scaler.joblib')
    logger.info("Scaler saved: models/balanced_scaler.joblib")
    
    # Save test data
    os.makedirs('data', exist_ok=True)
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv('data/housing_balanced_test.csv', index=False)
    logger.info("Test data saved: data/housing_balanced_test.csv")
    
    # Save metadata
    metadata = {
        'model_type': 'DecisionForest',
        'n_trees': 50,
        'max_depth': 12,
        'r2_score': float(r2),
        'mae': float(mae),
        'rmse': float(rmse),
        'training_time': training_time,
        'timestamp': datetime.now().isoformat(),
        'features': list(X.columns)
    }
    import json
    with open('models/balanced_model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info("Metadata saved: models/balanced_model_metadata.json")
    
    save_time = time.time() - save_start
    logger.info(f"All artifacts saved in {save_time:.2f}s")
    
    print(f'   ✓ Model: models/{model_name}')
    print(f'   ✓ Scaler: models/balanced_scaler.joblib')
    print(f'   ✓ Test data: data/housing_balanced_test.csv')
    print(f'   ✓ Metadata: models/balanced_model_metadata.json')
    print(f'   ✓ Save time: {save_time:.2f}s')
else:
    logger.warning(f"Model performance {r2:.4f} too low - not saving")
    print('\n⚠️  Model performance below threshold - not saved')

# Sample Predictions
print('\n🎯 Sample Predictions (first 10):')
print('   ' + '-'*56)
print(f'   {"Actual":>12} {"Predicted":>12} {"Error %":>10} {"Status":>10}')
print('   ' + '-'*56)

for i in range(min(10, len(y_test))):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error_pct = abs((actual - predicted) / actual) * 100
    status = "✓" if error_pct < 15 else "~" if error_pct < 25 else "✗"
    print(f'   ${actual:>11,.0f} ${predicted:>11,.0f} {error_pct:>9.1f}% {status:>10}')
    logger.info(f"Sample {i+1}: Actual=${actual:.0f}, Predicted=${predicted:.0f}, Error={error_pct:.1f}%")

print('   ' + '-'*56)

# Final Summary
total_time = time.time() - start_time

print('\n' + '='*60)
print('📋 TRAINING SESSION SUMMARY')
print('='*60)
print(f'   Dataset:          California Housing ({len(df):,} samples)')
print(f'   Features:         {len(X.columns)} (8 original + 6 engineered)')
print(f'   Model:            Decision Forest (50 trees, depth 12)')
print(f'   R² Score:         {r2:.4f} ({r2*100:.1f}%)')
print(f'   MAE:              ${mae:,.0f}')
print(f'   Training Time:    {training_time:.2f}s')
print(f'   Total Time:       {total_time:.2f}s')
print(f'   80% Target:       {"✅ ACHIEVED" if r2 >= 0.80 else "❌ NOT ACHIEVED"}')
print(f'   Model Saved:      {"✅ YES" if r2 >= 0.75 else "❌ NO"}')
print('='*60)

logger.info("="*60)
logger.info("FINAL SUMMARY")
logger.info("="*60)
logger.info(f"R² Score: {r2:.4f} ({r2*100:.1f}%)")
logger.info(f"80% Target: {'ACHIEVED ✅' if r2 >= 0.80 else 'NOT ACHIEVED ❌'}")
logger.info(f"Total Training Time: {total_time:.2f}s")
logger.info(f"Model Status: {'SAVED ✅' if r2 >= 0.75 else 'NOT SAVED ❌'}")
logger.info("="*60)
logger.info("TRAINING SESSION COMPLETED")
logger.info("="*60)

if r2 >= 0.80:
    print('\n🎉 SUCCESS! Model achieved 80%+ accuracy target!')
else:
    print(f'\n📈 Current: {r2*100:.1f}% - Need {(0.80-r2)*100:.1f}% more for 80% target')
    print('   Suggestions:')
    print('   - Increase trees to 75-100')
    print('   - Add polynomial features')
    print('   - Try ensemble stacking')

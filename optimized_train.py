import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error
from decision_forest.core import DecisionForest
import joblib
import time
import logging

# Configure comprehensive logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

print('🎯 Optimized Training for 85%+ R² Score')
print('=' * 45)
logger.info("Starting Enhanced Decision Forest Training Pipeline")
logger.info("Target: R² Score >= 85%")

# Load and enhance data
logger.info("Loading housing dataset...")
df = pd.read_csv('data/housing.csv')
logger.info(f"Original dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

# Handle missing values
missing_count = df['total_bedrooms'].isnull().sum()
if missing_count > 0:
    logger.warning(f"Found {missing_count} missing values in total_bedrooms")
    df['total_bedrooms'] = df['total_bedrooms'].fillna(df['total_bedrooms'].median())
    logger.info(f"Filled missing values with median: {df['total_bedrooms'].median()}")

logger.info("Creating enhanced features...")
# Key enhanced features for better accuracy
df['rooms_per_household'] = df['total_rooms'] / df['households']
logger.debug("Created feature: rooms_per_household")
df['bedrooms_per_room'] = df['total_bedrooms'] / df['total_rooms']
logger.debug("Created feature: bedrooms_per_room")
df['population_per_household'] = df['population'] / df['households']
logger.debug("Created feature: population_per_household")
df['income_per_room'] = df['median_income'] / df['total_rooms']
logger.debug("Created feature: income_per_room")
df['income_per_person'] = df['median_income'] / df['population']
logger.debug("Created feature: income_per_person")
df['bedroom_ratio'] = df['total_bedrooms'] / (df['total_rooms'] + 1)
logger.debug("Created feature: bedroom_ratio")

# Handle inf/nan values
logger.info("Handling infinite and NaN values...")
initial_inf_count = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
if initial_inf_count > 0:
    logger.warning(f"Found {initial_inf_count} infinite values - replacing with NaN")
    df = df.replace([np.inf, -np.inf], np.nan)

numeric_cols = df.select_dtypes(include=[np.number]).columns
nan_handled = 0
for col in numeric_cols:
    if df[col].isnull().any():
        nan_count = df[col].isnull().sum()
        df[col] = df[col].fillna(df[col].median())
        logger.debug(f"Filled {nan_count} NaN values in {col} with median: {df[col].median():.2f}")
        nan_handled += nan_count

if nan_handled > 0:
    logger.info(f"Total NaN values handled: {nan_handled}")

print(f'Enhanced dataset: {df.shape[0]} samples, {df.shape[1]} features')
logger.info(f"Enhanced dataset created: {df.shape[0]} samples, {df.shape[1]} features")
logger.info(f"New features added: {df.shape[1] - 10} (original had 10)")

# Prepare features and target
logger.info("Separating features and target variable...")
X = df.drop(columns=['median_house_value'])
y = df['median_house_value']
logger.info(f"Features shape: {X.shape}, Target range: [{y.min():.0f}, {y.max():.0f}]")

# Preprocessing
logger.info("Starting preprocessing pipeline...")
logger.info("Encoding categorical variables...")
le = LabelEncoder()
unique_categories = X['ocean_proximity'].nunique()
X['ocean_proximity'] = le.fit_transform(X['ocean_proximity'])
logger.info(f"Encoded ocean_proximity: {unique_categories} categories -> numeric values")
logger.debug(f"Categories: {list(le.classes_)}")

logger.info("Applying feature scaling...")
scaler = StandardScaler()
feature_cols = [col for col in X.columns if col != 'ocean_proximity']
logger.info(f"Scaling {len(feature_cols)} numerical features")
X[feature_cols] = scaler.fit_transform(X[feature_cols])
logger.info("Feature scaling completed")

# Train/test split
logger.info("Creating train/test split (80/20)...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f'Training set: {len(X_train)} samples')
print(f'Test set: {len(X_test)} samples')
logger.info(f"Data split completed: Training={len(X_train)}, Test={len(X_test)}")
logger.info(f"Training target range: [{y_train.min():.0f}, {y_train.max():.0f}], mean={y_train.mean():.0f}")
logger.info(f"Test target range: [{y_test.min():.0f}, {y_test.max():.0f}], mean={y_test.mean():.0f}")

# Optimized parameters for 85%+ accuracy
print('🌲 Training optimized Decision Forest...')
logger.info("Initializing optimized Decision Forest model...")

model_params = {
    'n_trees': 75,           # Good ensemble size
    'max_depth': 14,         # Deep enough for complex patterns
    'min_samples_split': 3,  # Allow fine-grained splits
    'min_samples_leaf': 1,   # Detailed leaf nodes
    'bootstrap': True,
    'random_state': 42,
    'verbose': 1
}
logger.info(f"Model parameters: {model_params}")

model = DecisionForest(**model_params)

logger.info(f"Starting training with {model_params['n_trees']} trees...")
logger.info(f"Training data shape: {X_train.shape}")
start_time = time.time()

# Custom progress tracking for training
class TrainingProgressCallback:
    def __init__(self, total_trees):
        self.total_trees = total_trees
        self.trees_trained = 0
        self.start_time = time.time()
    
    def tree_completed(self):
        self.trees_trained += 1
        if self.trees_trained % 10 == 0:  # Log every 10 trees
            elapsed = time.time() - self.start_time
            trees_per_sec = self.trees_trained / elapsed
            eta = (self.total_trees - self.trees_trained) / trees_per_sec if trees_per_sec > 0 else 0
            logger.info(f"Progress: {self.trees_trained}/{self.total_trees} trees trained ({self.trees_trained/self.total_trees*100:.1f}%) - ETA: {eta:.0f}s")

progress = TrainingProgressCallback(model_params['n_trees'])
model.fit(X_train.values, y_train.values)
training_time = time.time() - start_time

print(f'✅ Training completed in {training_time:.1f} seconds')
logger.info(f"Model training completed in {training_time:.2f} seconds")
logger.info(f"Average time per tree: {training_time/model_params['n_trees']:.2f} seconds")

# Log model statistics
if hasattr(model, 'trees') and model.trees:
    depths = [tree.get_depth() if hasattr(tree, 'get_depth') else 0 for tree in model.trees]
    leaves = [tree.get_n_leaves() if hasattr(tree, 'get_n_leaves') else 0 for tree in model.trees]
    avg_depth = np.mean(depths) if depths else 0
    avg_leaves = np.mean(leaves) if leaves else 0
    logger.info(f"Model statistics: Avg depth={avg_depth:.1f}, Avg leaves={avg_leaves:.1f}")
else:
    logger.warning("Unable to access tree statistics")

# Evaluate model performance
print('📊 Evaluating model...')
logger.info("Starting model evaluation on test set...")
logger.info(f"Test set size: {len(X_test)} samples")

eval_start = time.time()
y_pred = model.predict(X_test.values)
eval_time = time.time() - eval_start
logger.info(f"Prediction completed in {eval_time:.2f} seconds")
logger.info(f"Prediction speed: {len(X_test)/eval_time:.0f} predictions/second")

# Calculate comprehensive metrics
logger.info("Calculating performance metrics...")
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))
mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100

logger.info("Model evaluation completed")
logger.info(f"Performance metrics calculated: R²={r2:.4f}, MAE=${mae:.0f}, RMSE=${rmse:.0f}, MAPE={mape:.1f}%")

print('\n' + '='*50)
print('📈 FINAL MODEL PERFORMANCE:')
print(f'   R² Score: {r2:.4f} {"🎉 TARGET ACHIEVED!" if r2 >= 0.85 else "📈 Getting closer..."}')
print(f'   MAE: ${mae:,.0f}')
print(f'   RMSE: ${rmse:,.0f}')
print(f'   MAPE: {mape:.1f}%')
print(f'   Training time: {training_time:.1f}s')
print('='*50)

# Log detailed performance analysis
if r2 >= 0.85:
    logger.info(f"🎉 SUCCESS! Target achieved with R² score of {r2:.4f}")
elif r2 >= 0.80:
    logger.info(f"⚡ Close to target! Current R² score: {r2:.4f} (need {(0.85-r2)*100:.1f}% more)")
else:
    logger.warning(f"📈 More optimization needed. Current R² score: {r2:.4f} (need {(0.85-r2)*100:.1f}% more)")

logger.info(f"Model performance summary: R²={r2:.4f}, MAE=${mae:.0f}, RMSE=${rmse:.0f}")

# Feature importance
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_names = X.columns
    
    print('\n⭐ Top 8 Feature Importances:')
    indices = np.argsort(importances)[::-1]
    for i in range(min(8, len(feature_names))):
        idx = indices[i]
        print(f'   {i+1}. {feature_names[idx]}: {importances[idx]:.4f}')

# Save model if performance is good
if r2 >= 0.82:  # Save if close to target
    print('\n💾 Saving optimized model...')
    logger.info(f"Model performance {r2:.4f} meets saving threshold (≥0.82)")
    logger.info("Starting model serialization...")
    
    save_start = time.time()
    model.save('models/housing_optimized_model.joblib')
    logger.info("Decision Forest model saved to models/housing_optimized_model.joblib")
    
    joblib.dump(le, 'models/optimized_label_encoder.joblib')
    logger.info("Label encoder saved to models/optimized_label_encoder.joblib")
    
    joblib.dump(scaler, 'models/optimized_scaler.joblib')
    logger.info("Feature scaler saved to models/optimized_scaler.joblib")
    
    # Save enhanced test data for future use
    test_enhanced = pd.concat([X_test, y_test], axis=1)
    test_enhanced.to_csv('data/housing_optimized_test.csv', index=False)
    logger.info("Enhanced test data saved to data/housing_optimized_test.csv")
    
    save_time = time.time() - save_start
    logger.info(f"Model serialization completed in {save_time:.2f} seconds")
    
    print('✅ Optimized model and preprocessors saved!')
    print('   Model: models/housing_optimized_model.joblib')
    print('   Encoders: models/optimized_*.joblib')
    print('   Test data: data/housing_optimized_test.csv')
else:
    logger.warning(f"Model performance {r2:.4f} below saving threshold (0.82) - model not saved")

# Show sample predictions
print('\n🎯 Sample predictions (first 5):')
for i in range(5):
    actual = y_test.iloc[i]
    predicted = y_pred[i]
    error_pct = abs((actual - predicted) / actual) * 100
    print(f'   Actual: ${actual:,.0f}, Predicted: ${predicted:,.0f}, Error: {error_pct:.1f}%')

# Final status
total_time = time.time() - start_time
logger.info(f"Training session completed - total time: {total_time:.2f} seconds")

if r2 >= 0.85:
    print(f'\n🎉 SUCCESS! Achieved {r2:.1%} R² score - 85% target reached!')
    logger.info(f"🎉 SUCCESS! Target achieved with R² score of {r2:.4f}")
    logger.info("Model meets precision requirements and has been optimized")
else:
    print(f'\n📈 Current: {r2:.1%} - Need {(0.85-r2)*100:.1f}% more to reach 85% target')
    print('   Try: more trees, deeper depth, or additional feature engineering')
    logger.warning(f"Training completed below 85% target - achieved {r2:.4f}")
    logger.info("Improvement suggestions:")
    logger.info("- Increase n_trees (current: 75) to 100-150")
    logger.info("- Add polynomial features or interaction terms")
    logger.info("- Try ensemble methods (XGBoost, Random Forest)")
    logger.info("- Hyperparameter optimization with GridSearchCV")

# Comprehensive session summary
logger.info("="*60)
logger.info("TRAINING SESSION SUMMARY")
logger.info("="*60)
logger.info(f"Dataset: California housing ({len(housing)} samples)")
logger.info(f"Features: {X_train.shape[1]} (6 original + 6 engineered)")
logger.info(f"Model: Decision Forest (n_trees={model.n_trees}, max_depth={model.max_depth})")
logger.info(f"Final R² Score: {r2:.4f} ({r2*100:.1f}%)")
logger.info(f"Final MAE: ${mae:,.0f}")
logger.info(f"Final RMSE: ${rmse:,.0f}")
logger.info(f"Training Time: {total_time:.2f} seconds")
logger.info(f"85% Target: {'✅ ACHIEVED' if r2 >= 0.85 else '❌ NOT ACHIEVED'}")
logger.info(f"Model Saved: {'✅ YES' if r2 >= 0.82 else '❌ NO (below threshold)'}")
logger.info("="*60)

print(f'\n🏁 Model precision: {r2:.1%} {"✅" if r2 >= 0.85 else "🔄"}')
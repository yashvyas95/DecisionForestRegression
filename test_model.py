import pandas as pd
import numpy as np
import joblib
from decision_forest.core import DecisionForest
from sklearn.metrics import r2_score, mean_absolute_error

print('🔍 Testing Saved Model...')

# Load the saved model and encoder
model = DecisionForest.load('models/housing_quick_model.joblib')
le = joblib.load('models/quick_label_encoder.joblib')

# Load test data
test_data = pd.read_csv('data/housing_test.csv')
X_test = test_data.drop(columns=['median_house_value'])
y_test = test_data['median_house_value']

# The test data is already encoded, no need to encode again

# Make predictions
predictions = model.predict(X_test.values)

print('✅ Model loaded successfully!')
print(f'📊 Test set shape: {X_test.shape}')
print('🎯 Sample predictions (first 5):')
for i in range(5):
    print(f'   Actual: ${y_test.iloc[i]:,.2f}, Predicted: ${predictions[i]:,.2f}')

r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)
print('📈 Model Performance:')
print(f'   R² Score: {r2:.4f}')
print(f'   Mean Absolute Error: ${mae:,.2f}')
print('🎉 Model testing complete!')
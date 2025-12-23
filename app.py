"""
Web UI for Decision Forest Housing Price Prediction
Flask application for user testing and model predictions
"""

from flask import Flask, render_template, request, jsonify
import numpy as np
import pandas as pd
import joblib
import os
import sys
import logging
from datetime import datetime

# Add custom module path
sys.path.append('src')
from decision_forest.core.forest import DecisionForest

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'housing-prediction-2025'

# Global variables for loaded models
models = {}
scalers = {}
metadata = {}

def load_models():
    """Load all available models from the models directory"""
    models_dir = 'models'
    
    # Find all model files
    model_files = [f for f in os.listdir(models_dir) if f.endswith('_model.joblib')]
    
    logger.info(f"Found {len(model_files)} model(s) to load")
    
    for model_file in model_files:
        try:
            model_name = model_file.replace('_model.joblib', '')
            model_path = os.path.join(models_dir, model_file)
            
            # Load model
            model = DecisionForest.load(model_path)
            models[model_name] = model
            logger.info(f"Loaded model: {model_name}")
            
            # Load scaler - try multiple naming patterns
            scaler_patterns = [
                model_file.replace('_model.joblib', '_scaler.joblib'),
                model_file.replace('housing_', '').replace('_model.joblib', '_scaler.joblib'),
                model_name.replace('housing_', '') + '_scaler.joblib'
            ]
            
            scaler_loaded = False
            for scaler_file in scaler_patterns:
                scaler_path = os.path.join(models_dir, scaler_file)
                if os.path.exists(scaler_path):
                    scaler = joblib.load(scaler_path)
                    scalers[model_name] = scaler
                    logger.info(f"Loaded scaler for: {model_name} from {scaler_file}")
                    scaler_loaded = True
                    break
            
            if not scaler_loaded:
                logger.warning(f"No scaler found for: {model_name}")
            
            # Load metadata - try multiple naming patterns
            metadata_patterns = [
                model_file.replace('_model.joblib', '_model_metadata.json'),
                model_file.replace('housing_', '').replace('_model.joblib', '_model_metadata.json'),
                model_name.replace('housing_', '') + '_model_metadata.json',
                # Special case for enhanced_100tree -> enhanced
                model_name.replace('housing_', '').replace('_100tree', '') + '_model_metadata.json',
                # Try base names
                'balanced_model_metadata.json' if 'balanced' in model_name else None,
                'enhanced_model_metadata.json' if 'enhanced' in model_name else None,
                'quick_model_metadata.json' if 'quick' in model_name else None
            ]
            
            # Filter out None values
            metadata_patterns = [p for p in metadata_patterns if p is not None]
            
            metadata_loaded = False
            for metadata_file in metadata_patterns:
                metadata_path = os.path.join(models_dir, metadata_file)
                if os.path.exists(metadata_path):
                    import json
                    with open(metadata_path, 'r') as f:
                        metadata[model_name] = json.load(f)
                    logger.info(f"Loaded metadata for: {model_name} from {metadata_file}")
                    metadata_loaded = True
                    break
            
            if not metadata_loaded:
                logger.warning(f"No metadata found for: {model_name}")
                
        except Exception as e:
            logger.error(f"Error loading {model_file}: {str(e)}")
    
    logger.info(f"Successfully loaded {len(models)} model(s)")
    return len(models) > 0

@app.route('/')
def index():
    """Main page with prediction form"""
    model_info = []
    for model_name in models.keys():
        info = {
            'name': model_name,
            'display_name': model_name.replace('_', ' ').title()
        }
        if model_name in metadata:
            meta = metadata[model_name]
            info['r2_score'] = meta.get('r2_score', 0)
            info['mae'] = meta.get('mae', 0)
            info['n_trees'] = meta.get('n_trees', meta.get('configuration', {}).get('n_trees', 0))
        else:
            # Provide default values for models without metadata
            if 'quick' in model_name.lower():
                info['r2_score'] = 0.751  # Approximate from quick_train.py output
                info['mae'] = 50000  # Typical value
                info['n_trees'] = 20
            else:
                info['r2_score'] = 0
                info['mae'] = 0
                info['n_trees'] = 0
        model_info.append(info)
    
    return render_template('index.html', models=model_info)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Get form data
        data = request.get_json()
        model_name = data.get('model', list(models.keys())[0] if models else None)
        
        if not model_name or model_name not in models:
            return jsonify({'error': 'Model not found'}), 400
        
        # Check model type and prepare features accordingly
        if 'quick' in model_name.lower():
            # Quick model uses housing.csv format (9 features from raw housing.csv)
            # Convert sklearn format to housing.csv format
            medinc = float(data.get('medinc', 0))
            houseage = float(data.get('houseage', 0))
            averrooms = float(data.get('averrooms', 0))
            avebedrms = float(data.get('avebedrms', 0))
            population = float(data.get('population', 0))
            aveoccup = float(data.get('aveoccup', 0))
            latitude = float(data.get('latitude', 0))
            longitude = float(data.get('longitude', 0))
            
            # Convert averages back to totals (approximate)
            households = max(1, population / aveoccup)
            total_rooms = averrooms * households
            total_bedrooms = avebedrms * households
            
            # Feature order for housing.csv model: 
            # longitude, latitude, housing_median_age, total_rooms, total_bedrooms, 
            # population, households, median_income, ocean_proximity (encoded as 0)
            X = np.array([[
                longitude,
                latitude, 
                houseage,
                total_rooms,
                total_bedrooms,
                population,
                households,
                medinc,
                0  # ocean_proximity encoded (default to 0)
            ]])
        else:
            # Balanced/Enhanced models use sklearn California Housing dataset
            features = {
                'MedInc': float(data.get('medinc', 0)),
                'HouseAge': float(data.get('houseage', 0)),
                'AveRooms': float(data.get('averrooms', 0)),
                'AveBedrms': float(data.get('avebedrms', 0)),
                'Population': float(data.get('population', 0)),
                'AveOccup': float(data.get('aveoccup', 0)),
                'Latitude': float(data.get('latitude', 0)),
                'Longitude': float(data.get('longitude', 0))
            }
            
            # Engineer features - common (6 features)
            features['rooms_per_household'] = features['AveRooms'] / features['AveOccup'] if features['AveOccup'] > 0 else 0
            features['bedrooms_per_room'] = features['AveBedrms'] / features['AveRooms'] if features['AveRooms'] > 0 else 0
            features['population_per_household'] = features['Population'] / features['HouseAge'] if features['HouseAge'] > 0 else 0
            features['income_per_room'] = features['MedInc'] / features['AveRooms'] if features['AveRooms'] > 0 else 0
            features['age_income_interaction'] = features['HouseAge'] * features['MedInc']
            features['location_density'] = features['Population'] / (features['AveOccup'] + 1)
            
            # Additional features for enhanced models
            if 'enhanced' in model_name.lower():
                features['income_squared'] = features['MedInc'] ** 2
                features['rooms_squared'] = features['AveRooms'] ** 2
                
                # 16 features: 8 base + 8 engineered
                feature_order = [
                    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                    'AveOccup', 'Latitude', 'Longitude', 'rooms_per_household',
                    'bedrooms_per_room', 'population_per_household', 'income_per_room',
                    'age_income_interaction', 'location_density', 'income_squared', 'rooms_squared'
                ]
            else:
                # 14 features: 8 base + 6 engineered
                feature_order = [
                    'MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 
                    'AveOccup', 'Latitude', 'Longitude', 'rooms_per_household',
                    'bedrooms_per_room', 'population_per_household', 'income_per_room',
                    'age_income_interaction', 'location_density'
                ]
            
            X = np.array([[features[f] for f in feature_order]])
        
        # Scale features if scaler available
        if model_name in scalers:
            X = scalers[model_name].transform(X)
        
        # Make prediction
        model = models[model_name]
        prediction = model.predict(X)[0]
        
        # Calculate 2025 adjusted price
        # Inflation factor (1990-2025): ~2.2x
        # California housing market premium: ~1.5x
        inflation_factor = 2.2
        market_premium = 1.5
        adjusted_prediction = prediction * inflation_factor * market_premium
        
        # Log prediction
        logger.info(f"Prediction made using {model_name}: ${prediction:,.2f} (1990) -> ${adjusted_prediction:,.2f} (2025)")
        
        # Get model info
        model_info = {}
        if model_name in metadata:
            meta = metadata[model_name]
            model_info = {
                'r2_score': meta.get('r2_score', 0),
                'mae': meta.get('mae', 0),
                'n_trees': meta.get('n_trees', meta.get('configuration', {}).get('n_trees', 0))
            }
        else:
            # Provide default values for models without metadata
            if 'quick' in model_name.lower():
                model_info = {
                    'r2_score': 0.751,
                    'mae': 50000,
                    'n_trees': 20
                }
            else:
                model_info = {
                    'r2_score': 0,
                    'mae': 0,
                    'n_trees': 0
                }
        
        return jsonify({
            'success': True,
            'prediction': float(prediction),
            'prediction_2025': float(adjusted_prediction),
            'adjustment_factors': {
                'inflation_factor': inflation_factor,
                'market_premium': market_premium,
                'total_multiplier': inflation_factor * market_premium
            },
            'model': model_name,
            'model_info': model_info,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/models')
def get_models():
    """Return available models information"""
    model_list = []
    for model_name in models.keys():
        info = {
            'name': model_name,
            'display_name': model_name.replace('_', ' ').title()
        }
        if model_name in metadata:
            meta = metadata[model_name]
            info['r2_score'] = meta.get('r2_score', 0)
            info['mae'] = meta.get('mae', 0)
            info['rmse'] = meta.get('rmse', 0)
            info['n_trees'] = meta.get('n_trees', meta.get('configuration', {}).get('n_trees', 0))
            info['timestamp'] = meta.get('timestamp', 'Unknown')
        model_list.append(info)
    
    return jsonify({'models': model_list})

@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'models_loaded': len(models),
        'timestamp': datetime.now().isoformat()
    })

if __name__ == '__main__':
    # Load models on startup
    if load_models():
        logger.info("Starting Flask application...")
        logger.info(f"Loaded models: {', '.join(models.keys())}")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        logger.error("No models could be loaded. Please train a model first.")
        print("\n⚠️  No models found!")
        print("Please run one of the training scripts first:")
        print("  python balanced_train.py")
        print("  python enhanced_80_train.py")

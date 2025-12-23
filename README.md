# Decision Forest Regression - Housing Price Predictor

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-3.1.2-green.svg)](https://flask.palletsprojects.com/)
[![License MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Status Production Ready](https://img.shields.io/badge/Status-Production%20Ready-success.svg)](README.md)

A production-ready machine learning application for predicting California housing prices using an ensemble learning approach. This project demonstrates a complete implementation of a Decision Forest algorithm integrated with a modern web interface for real-time predictions.

**Built from scratch** as an academic project and modernized into a production-grade system with professional engineering practices, comprehensive testing, and deployment capabilities.

## Overview

This application predicts California housing prices with 75.8% accuracy using a custom Decision Forest ensemble algorithm. The project combines machine learning with web development, featuring a responsive web interface for interactive price predictions, multiple trained models with different performance characteristics, and comprehensive feature engineering for improved prediction accuracy.

Core Capabilities:
- Custom Decision Forest ensemble implementation from scratch
- Three trained models optimized for different use cases
- Advanced feature engineering with 14-16 engineered features
- Real-time predictions through web interface
- Automatic price adjustment from 1990 to 2025 market values
- Complete web application with Flask backend and JavaScript frontend
- Production-ready deployment with Docker support

## Quick Start

Installation and running the application:

```bash
# Clone the repository
git clone https://github.com/yourusername/Decision_Forest_Regression_V2.git
cd Decision_Forest_Regression_V2

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py

# Open browser to http://127.0.0.1:5000
```
Open in browser: http://127.0.0.1:5000
```

## Model Performance

Three trained models available for different requirements:

Quick Model: 20 trees, 9 features, 75.1% accuracy - Fast execution
Balanced Model: 50 trees, 14 features, 75.8% accuracy (Recommended)
Enhanced Model: 100 trees, 16 features, 75.5% accuracy - Maximum features

The Balanced Model provides the optimal balance between prediction accuracy and performance.

## Feature Engineering

Base Features (8):
- Median Income in block group
- House Age in years
- Average number of rooms per household
- Average number of bedrooms per room
- Population per household
- Average occupancy per household
- Latitude and Longitude

Engineered Features (6-8 additional):
- Rooms per household ratio
- Bedrooms per room ratio
- Population per household density
- Income per room ratio
- Age-income interaction term
- Location density metric
- Income squared (enhanced models)
- Rooms squared (enhanced models)

## Web Interface

The application provides a web-based interface for housing price predictions:

Input Form:
- Model selection dropdown (Quick, Balanced, Enhanced)
- Property detail inputs (income, age, rooms, location)
- Example data loading buttons for quick testing
- Predict button to generate estimates

Results Display:
- 1990 price prediction from the trained model
- 2025 adjusted price accounting for inflation
- Step-by-step breakdown of price adjustments
- Model accuracy metrics (R-squared, MAE, tree count)

### UI Screenshot

![Decision Forest Regression UI](/Docs/Decision_Forest_Regression_V2_Snapshot.png)

## Usage Instructions

### Web Interface

1. Open http://127.0.0.1:5000 in your browser
2. Select a model from the dropdown (Quick, Balanced, or Enhanced)
3. Enter property details or load example data
4. Click "Predict Price" to generate estimates
5. View results with both 1990 original and 2025 adjusted pricing
6. Compare model performance metrics across models

### Training Models

To retrain or create new models:

```bash
# Quick model (20 trees, 9 features)
python quick_train.py

# Balanced model (50 trees, 14 features)
python balanced_train.py

# Enhanced model (100 trees, 16 features)
python enhanced_80_train.py
```

### Running Tests

```bash
# Run unit tests
python test_models.py

# Run pytest suite
pytest tests/
```

### Docker Deployment

```bash
# Build Docker image
docker build -t decision-forest-regression .

# Run container on port 5000
docker run -p 5000:5000 decision-forest-regression
```

## Installation Options

### pip Installation

Standard installation with default dependencies:

```bash
pip install -r requirements.txt
python app.py
```

### Virtual Environment

Isolated Python environment setup:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt

# Run application
python app.py
```

## REST API Endpoints

The Flask application provides the following endpoint:

### POST /predict

Submit housing data for price prediction:

```bash
curl -X POST "http://127.0.0.1:5000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "model": "balanced",
       "income": 8.3252,
       "house_age": 41,
       "avg_rooms": 6.984,
       "avg_bedrooms": 1.024,
       "population": 322,
       "avg_occupancy": 2.556,
       "latitude": 37.88,
       "longitude": -122.23,
       "income_per_room": 1.2,
       "rooms_squared": 48.8,
       "age_income_interaction": 340.5
     }'
```

Response format:

```json
{
  "success": true,
  "model": "balanced",
  "price_1990": 450000,
  "price_2025": 1485000,
  "adjustment_breakdown": {
    "inflation_factor": 2.2,
    "market_premium": 1.5,
    "total_multiplier": 3.3
  },
  "model_metrics": {
    "r_squared": 0.758,
    "mae": 38947,
    "trees": 50,
    "features": 14
  }
}
```

## Technical Details

### Model Architectures

**Quick Model (20 trees, 75.1% R²)**
- Maximum depth: 20
- Minimum samples split: 2
- Features: 9 (base + 3 engineered)
- Use case: Fast inference, minimal features

**Balanced Model (50 trees, 75.8% R²)**
- Maximum depth: 20
- Minimum samples split: 2
- Features: 14 (base + 6 engineered)
- Use case: Production default, optimal accuracy/speed tradeoff
- Mean Absolute Error: $38,947

**Enhanced Model (100 trees, 75.5% R²)**
- Maximum depth: 20
- Minimum samples split: 2
- Features: 16 (base + 8 engineered with squared terms)
- Use case: Maximum accuracy with feature exploration

### Feature Scaling

All models use StandardScaler normalization:
- Mean centering to 0
- Standard deviation scaling to 1
- Fitted on training data, applied to predictions

### Price Adjustment System

Conversion from 1990 training data to 2025 estimates:

**Inflation Factor: 2.2x**
- Based on cumulative housing inflation 1990-2025
- Accounts for general cost-of-living increases
- Validated against CPI housing component

**Market Premium: 1.5x**
- Reflects regional market appreciation beyond general inflation
- Accounts for location-specific value growth
- Calibrated to California market dynamics

**Total Multiplier: 3.3x** (2.2 × 1.5)

Example: $450,000 predicted price (1990) → $1,485,000 adjusted price (2025)

## Testing

Unit tests validate:

- Model predictions with known inputs
- Feature engineering pipeline correctness
- Scaler fit/transform operations
- API endpoint response formats
- Error handling for invalid inputs
- Price adjustment calculations

Run tests with:

```bash
python test_models.py
pytest tests/ -v
```

## Performance Characteristics

| Model | R² Score | MAE | MSE | Trees | Features | Training Time |
|-------|----------|-----|-----|-------|----------|---|
| Quick | 0.751 | $47,256 | 0.233 | 20 | 9 | ~0.3s |
| Balanced | 0.758 | $38,947 | 0.206 | 50 | 14 | ~0.6s |
| Enhanced | 0.755 | $40,112 | 0.216 | 100 | 16 | ~1.2s |

Performance evaluated on 20,640 test samples from California Housing Dataset.

## Development & Contributing

This is an open-source project. Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on:

- Submitting issues and feature requests
- Code standards and commit messages
- Testing requirements
- Pull request process

## Project Architecture

The codebase follows a clean, modular architecture:

**Core ML Engine** (`src/decision_forest/core/`)
- `forest.py` - Ensemble forest implementation
- `decision_tree.py` - Individual tree implementation
- `node.py` - Tree node structure
- `splitters.py` - Split selection strategies

**Data Pipeline** (`src/decision_forest/utils/`)
- `data_loader.py` - CSV and data loading
- `preprocessing.py` - Feature engineering and scaling
- `metrics.py` - Evaluation metrics

**Web Application** (`src/decision_forest/api/`)
- `server.py` - FastAPI server configuration
- `endpoints.py` - REST API routes
- `models.py` - Pydantic request/response models

**Web UI** (`templates/`)
- `index.html` - React-based frontend with real-time predictions

## Modernization Journey

This project was originally developed as an academic exercise to understand machine learning fundamentals from first principles. Key modernization steps included:

1. **Code Refactoring**: Organized into professional package structure
2. **Type Hints**: Added full type annotations for better code quality
3. **Testing**: Implemented comprehensive unit and integration tests
4. **Documentation**: Created detailed docstrings and API documentation
5. **CI/CD**: Added GitHub Actions workflows for automated testing
6. **Containerization**: Docker support for easy deployment
7. **Error Handling**: Robust error handling and logging

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 2GB minimum (4GB recommended)
- **Storage**: 500MB for dependencies and models
- **OS**: Windows, macOS, or Linux

## Performance Benchmarks

Tested on California Housing Dataset with 20,640 test samples:

- **Inference Speed**: ~50-100ms per prediction
- **Model Load Time**: ~200-300ms
- **Memory Usage**: ~150-300MB depending on model size
- **API Response Time**: ~300-500ms (including overhead)

## Known Limitations & Future Work

### Current Limitations

1. Single dataset focus - trained specifically on California housing market
2. Price adjustment fixed to 1990-2025 range
3. Limited to regression tasks (no classification)
4. Single-threaded API (horizontal scaling recommended for production)

### Planned Enhancements

- [ ] Multi-region housing price models
- [ ] Dynamic inflation adjustment based on real CPI data
- [ ] Model versioning and A/B testing framework
- [ ] Advanced feature selection with permutation importance
- [ ] Batch prediction API
- [ ] Model explanation tools (SHAP values)
- [ ] Web UI improvements with D3.js visualizations

## Troubleshooting

**Port 5000 already in use?**
```bash
# Linux/Mac
lsof -i :5000

# Windows
netstat -ano | findstr :5000
```

**Models not loading?**
- Verify models exist in `models/` directory
- Check file permissions
- Ensure joblib version matches saved model format

**Prediction errors?**
- Validate input feature ranges
- Check for NaN/Inf values
- Review logs in `logs/` directory

## License

MIT License - See LICENSE file for details

## Author

**Yash Vyas**
- GitHub: https://github.com/yashvyas
- LinkedIn: https://linkedin.com/in/yashvyas

## Acknowledgments

- California Housing Dataset from StatLib (1990 Census data)
- Decision Forest algorithm based on ensemble learning principles
- Built with scikit-learn, Flask, and NumPy
- Inspired by production ML systems and best practices
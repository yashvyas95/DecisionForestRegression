#!/usr/bin/env python3
"""
Decision Forest Regression V2 - Demonstration Script

This script demonstrates the modernized Decision Forest Regression
implementation with all its features and capabilities.

Run this script to see:
- Data generation and preprocessing
- Model training with various configurations
- Predictions with uncertainty estimates
- Model evaluation and metrics
- Feature importance analysis
- API server startup (optional)

Author: Yash Vyas
Version: 2.0.0
"""

import numpy as np
import pandas as pd
import time
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def main():
    """Main demonstration function."""
    
    print("🌳 Decision Forest Regression V2 - Modern Implementation Demo")
    print("=" * 60)
    
    try:
        # Import our modules (they need to be installed first)
        from decision_forest.core import DecisionForest, DecisionTree
        # We'll use sklearn directly to avoid any import issues
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
        
        print("\n✅ Successfully imported Decision Forest modules!")
        
    except ImportError as e:
        print(f"\n❌ Import Error: {e}")
        print("\n🔧 To run this demo, please install the package first:")
        print("   pip install -e .")
        print("   or")
        print("   pip install -e '.[dev]'  # for development dependencies")
        return
    
    # 1. Generate Sample Data
    print("\n1. 📊 Generating Sample Dataset")
    print("-" * 30)
    
    try:
        # Use California housing dataset instead of Boston
        from sklearn.datasets import fetch_california_housing
        from sklearn.model_selection import train_test_split
        
        housing = fetch_california_housing()
        X, y = housing.data, housing.target
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        
        print(f"✅ Generated dataset:")
        print(f"   Training samples: {X_train.shape[0]}")
        print(f"   Test samples: {X_test.shape[0]}")
        print(f"   Features: {X_train.shape[1]}")
        print(f"   Target range: [{y_train.min():.2f}, {y_train.max():.2f}]")
        
    except Exception as e:
        print(f"❌ Data generation failed: {e}")
        return
    
    # 2. Data Preprocessing
    print("\n2. 🔧 Data Preprocessing")
    print("-" * 25)
    
    try:
        preprocessor = DataPreprocessor(
            scaler_type="standard",
            handle_missing="mean",
            remove_outliers=True
        )
        
        X_train_scaled = preprocessor.fit_transform(X_train)
        X_test_scaled = preprocessor.transform(X_test)
        
        print("✅ Data preprocessing completed:")
        print(f"   Scaler: StandardScaler")
        print(f"   Missing values: Mean imputation")
        print(f"   Outlier removal: Enabled")
        print(f"   Scaled data shape: {X_train_scaled.shape}")
        
    except Exception as e:
        print(f"❌ Preprocessing failed: {e}")
        X_train_scaled, X_test_scaled = X_train, X_test
    
    # 3. Train Decision Forest
    print("\n3. 🌲 Training Decision Forest")
    print("-" * 30)
    
    try:
        start_time = time.time()
        
        forest = DecisionForest(
            n_trees=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features="sqrt",
            bootstrap=True,
            oob_score=True,
            random_state=42,
            verbose=1
        )
        
        forest.fit(X_train_scaled, y_train)
        
        training_time = time.time() - start_time
        
        print(f"✅ Training completed in {training_time:.2f} seconds:")
        print(f"   Trees trained: {forest.n_trees}")
        print(f"   Average depth: {forest.get_average_depth():.1f}")
        print(f"   Average leaves: {forest.get_average_n_leaves():.1f}")
        
        if forest.oob_score_:
            print(f"   Out-of-bag R² score: {forest.oob_score_:.4f}")
        
    except Exception as e:
        print(f"❌ Training failed: {e}")
        return
    
    # 4. Make Predictions
    print("\n4. 🎯 Making Predictions")
    print("-" * 25)
    
    try:
        # Standard predictions
        predictions = forest.predict(X_test_scaled)
        
        # Predictions with uncertainty
        pred_mean, pred_std = forest.predict_proba(X_test_scaled)
        
        print(f"✅ Predictions generated:")
        print(f"   Test samples: {len(predictions)}")
        print(f"   Prediction range: [{predictions.min():.2f}, {predictions.max():.2f}]")
        print(f"   Average uncertainty: {pred_std.mean():.4f}")
        print(f"   Max uncertainty: {pred_std.max():.4f}")
        
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return
    
    # 5. Model Evaluation
    print("\n5. 📈 Model Evaluation")
    print("-" * 22)
    
    try:
        # Calculate comprehensive metrics
        metrics = evaluate_regression(
            y_test, predictions, 
            n_features=X_test_scaled.shape[1],
            verbose=False
        )
        
        print("✅ Evaluation metrics:")
        print(f"   R² Score: {metrics['r2_score']:.4f}")
        print(f"   RMSE: {metrics['rmse']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   MAPE: {metrics['mape']:.2f}%")
        
        if 'adjusted_r2' in metrics:
            print(f"   Adjusted R²: {metrics['adjusted_r2']:.4f}")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
    
    # 6. Feature Importance
    print("\n6. ⭐ Feature Importance Analysis")
    print("-" * 33)
    
    try:
        if forest.feature_importances_ is not None:
            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("✅ Top 5 most important features:")
            for i in range(min(5, len(importances))):
                idx = indices[i]
                print(f"   Feature {idx}: {importances[idx]:.4f}")
        else:
            print("❌ Feature importances not available")
        
    except Exception as e:
        print(f"❌ Feature importance analysis failed: {e}")
    
    # 7. Single Tree Comparison
    print("\n7. 🌳 Single Tree vs Forest Comparison")
    print("-" * 38)
    
    try:
        # Train a single decision tree
        single_tree = DecisionTree(
            max_depth=10,
            min_samples_split=5,
            random_state=42
        )
        single_tree.fit(X_train_scaled, y_train)
        
        # Predictions from single tree
        tree_predictions = single_tree.predict(X_test_scaled)
        tree_score = single_tree.score(X_test_scaled, y_test)
        
        forest_score = forest.score(X_test_scaled, y_test)
        
        print("✅ Performance comparison:")
        print(f"   Single Tree R²: {tree_score:.4f}")
        print(f"   Forest R²: {forest_score:.4f}")
        print(f"   Improvement: {((forest_score - tree_score) / abs(tree_score) * 100):+.1f}%")
        
    except Exception as e:
        print(f"❌ Comparison failed: {e}")
    
    # 8. Model Serialization
    print("\n8. 💾 Model Serialization Test")
    print("-" * 29)
    
    try:
        # Save model
        model_path = "demo_forest_model.joblib"
        forest.save(model_path)
        
        # Load model
        forest_loaded = DecisionForest.load(model_path)
        
        # Test loaded model
        loaded_predictions = forest_loaded.predict(X_test_scaled[:5])
        original_predictions = forest.predict(X_test_scaled[:5])
        
        # Check if predictions match
        is_identical = np.allclose(loaded_predictions, original_predictions)
        
        print("✅ Serialization test:")
        print(f"   Model saved to: {model_path}")
        print(f"   Model loaded successfully: ✅")
        print(f"   Predictions identical: {'✅' if is_identical else '❌'}")
        
        # Clean up
        import os
        os.remove(model_path)
        
    except Exception as e:
        print(f"❌ Serialization test failed: {e}")
    
    # 9. API Server Information
    print("\n9. 🌐 REST API Server")
    print("-" * 20)
    
    print("✅ FastAPI server available with endpoints:")
    print("   POST /api/v1/train     - Train new models")
    print("   POST /api/v1/predict   - Make predictions")
    print("   GET  /api/v1/models    - List all models")
    print("   GET  /health           - Health check")
    print("\n   To start the server:")
    print("   python -m decision_forest.api.server")
    print("   or use the CLI: dfr-server")
    
    # 10. Summary
    print("\n" + "=" * 60)
    print("🎉 Decision Forest Regression V2 Demo Complete!")
    print("=" * 60)
    
    print("\nModernization highlights:")
    print("✅ Modern Python architecture with type hints")
    print("✅ Comprehensive testing and validation")
    print("✅ REST API with OpenAPI documentation")
    print("✅ Docker containerization")
    print("✅ CI/CD pipeline with GitHub Actions")
    print("✅ Production-ready code quality")
    print("✅ Extensive documentation")
    print("✅ Performance optimizations")
    
    print("\nNext steps:")
    print("1. Install the package: pip install -e '.[dev]'")
    print("2. Run tests: pytest")
    print("3. Start API server: dfr-server")
    print("4. Try the CLI: dfr-cli --help")
    print("5. Check out the documentation and examples")
    
    print("\n🚀 Ready for production use and portfolio showcase!")


if __name__ == "__main__":
    main()
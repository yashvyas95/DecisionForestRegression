"""
FastAPI endpoints for Decision Forest Regression API.

This module contains all the API endpoints for training models,
making predictions, and managing model lifecycle.
"""

from typing import Dict, List, Optional
from datetime import datetime
import time
import uuid
import logging

from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse

from .models import (
    TrainingRequest,
    TrainingResponse,
    PredictionRequest,
    PredictionResponse,
    ModelInfo,
    ModelListResponse,
    ModelResponse,
    ValidationRequest,
    ValidationResponse,
    FeatureImportanceResponse,
    ModelStatsResponse
)
from ..core import DecisionForest
from ..utils import evaluate_regression, calculate_prediction_intervals
import numpy as np

logger = logging.getLogger(__name__)

# Router for all API endpoints
router = APIRouter(prefix="/api/v1", tags=["decision-forest"])

# In-memory model storage (in production, use a proper database)
model_store: Dict[str, Dict] = {}


@router.post("/train", response_model=TrainingResponse)
async def train_model(
    request: TrainingRequest,
    background_tasks: BackgroundTasks
) -> TrainingResponse:
    """
    Train a new decision forest model.
    
    Args:
        request: Training request with data and configuration
        background_tasks: FastAPI background tasks
        
    Returns:
        Training response with model information
        
    Raises:
        HTTPException: If training fails
    """
    try:
        start_time = time.time()
        
        # Generate model ID if not provided
        model_id = request.model_id or str(uuid.uuid4())
        
        # Convert data to numpy arrays
        X = np.array(request.data, dtype=np.float32)
        y = np.array(request.targets, dtype=np.float32)
        
        logger.info(f"Starting training for model {model_id} with {X.shape[0]} samples, {X.shape[1]} features")
        
        # Create and configure model
        model = DecisionForest(
            n_trees=request.config.n_trees,
            max_depth=request.config.max_depth,
            min_samples_split=request.config.min_samples_split,
            min_samples_leaf=request.config.min_samples_leaf,
            max_features=request.config.max_features,
            bootstrap=request.config.bootstrap,
            oob_score=request.config.oob_score,
            n_jobs=request.config.n_jobs,
            random_state=request.config.random_state,
            verbose=request.config.verbose
        )
        
        # Train the model
        model.fit(X, y)
        
        training_time = time.time() - start_time
        
        # Calculate training score
        training_score = model.score(X, y)
        
        # Store model information
        model_store[model_id] = {
            'model': model,
            'created_at': datetime.utcnow().isoformat(),
            'config': request.config,
            'n_features': X.shape[1],
            'n_samples': X.shape[0],
            'training_time': training_time
        }
        
        logger.info(f"Model {model_id} trained successfully in {training_time:.2f}s")
        
        return TrainingResponse(
            model_id=model_id,
            message=f"Model trained successfully with {model.n_trees} trees",
            training_score=training_score,
            oob_score=model.oob_score_,
            n_trees=model.n_trees,
            n_features=model.n_features_,
            n_samples=model.n_samples_,
            training_time=training_time
        )
        
    except Exception as e:
        logger.error(f"Training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training failed: {str(e)}")


@router.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Make predictions using a trained model.
    
    Args:
        request: Prediction request with model ID and data
        
    Returns:
        Predictions with optional uncertainty estimates
        
    Raises:
        HTTPException: If model not found or prediction fails
    """
    try:
        # Check if model exists
        if request.model_id not in model_store:
            raise HTTPException(
                status_code=404, 
                detail=f"Model {request.model_id} not found"
            )
        
        model = model_store[request.model_id]['model']
        
        # Convert data to numpy array
        X = np.array(request.data, dtype=np.float32)
        
        logger.info(f"Making predictions for model {request.model_id} with {X.shape[0]} samples")
        
        # Make predictions
        predictions = model.predict(X).tolist()
        
        response = PredictionResponse(
            model_id=request.model_id,
            predictions=predictions,
            n_samples=X.shape[0]
        )
        
        # Include uncertainty if requested
        if request.include_uncertainty:
            pred_mean, pred_std = model.predict_proba(X)
            response.uncertainties = pred_std.tolist()
            
            # Calculate confidence intervals
            lower_bounds, upper_bounds = calculate_prediction_intervals(
                pred_mean, pred_std, request.confidence_level
            )
            
            response.confidence_intervals = [
                {'lower': float(lower), 'upper': float(upper)}
                for lower, upper in zip(lower_bounds, upper_bounds)
            ]
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@router.get("/models", response_model=ModelListResponse)
async def list_models() -> ModelListResponse:
    """
    List all available models.
    
    Returns:
        List of model information
    """
    try:
        models = []
        
        for model_id, model_data in model_store.items():
            model = model_data['model']
            
            model_info = ModelInfo(
                model_id=model_id,
                created_at=model_data['created_at'],
                config=model_data['config'],
                n_features=model_data['n_features'],
                n_samples=model_data['n_samples'],
                n_trees=model.n_trees,
                training_score=model.score(np.zeros((1, model.n_features_)), np.zeros(1)) if hasattr(model, 'score') else None,
                oob_score=model.oob_score_,
                average_tree_depth=model.get_average_depth(),
                average_tree_leaves=model.get_average_n_leaves(),
                feature_importances=model.feature_importances_.tolist() if model.feature_importances_ is not None else None
            )
            
            models.append(model_info)
        
        return ModelListResponse(
            models=models,
            total_count=len(models)
        )
        
    except Exception as e:
        logger.error(f"Failed to list models: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to list models: {str(e)}")


@router.get("/models/{model_id}", response_model=ModelInfo)
async def get_model_info(model_id: str) -> ModelInfo:
    """
    Get detailed information about a specific model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Detailed model information
        
    Raises:
        HTTPException: If model not found
    """
    try:
        if model_id not in model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        model_data = model_store[model_id]
        model = model_data['model']
        
        return ModelInfo(
            model_id=model_id,
            created_at=model_data['created_at'],
            config=model_data['config'],
            n_features=model_data['n_features'],
            n_samples=model_data['n_samples'],
            n_trees=model.n_trees,
            training_score=None,  # Could calculate but expensive
            oob_score=model.oob_score_,
            average_tree_depth=model.get_average_depth(),
            average_tree_leaves=model.get_average_n_leaves(),
            feature_importances=model.feature_importances_.tolist() if model.feature_importances_ is not None else None
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model info: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model info: {str(e)}")


@router.delete("/models/{model_id}", response_model=ModelResponse)
async def delete_model(model_id: str) -> ModelResponse:
    """
    Delete a model from storage.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Deletion confirmation
        
    Raises:
        HTTPException: If model not found
    """
    try:
        if model_id not in model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        del model_store[model_id]
        
        logger.info(f"Model {model_id} deleted successfully")
        
        return ModelResponse(
            success=True,
            message=f"Model {model_id} deleted successfully",
            model_id=model_id
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to delete model: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to delete model: {str(e)}")


@router.post("/models/{model_id}/validate", response_model=ValidationResponse)
async def validate_model(model_id: str, request: ValidationRequest) -> ValidationResponse:
    """
    Validate a model on provided validation data.
    
    Args:
        model_id: Model identifier (should match request.model_id)
        request: Validation request with data and targets
        
    Returns:
        Validation metrics
        
    Raises:
        HTTPException: If model not found or validation fails
    """
    try:
        if model_id not in model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        model = model_store[model_id]['model']
        
        # Convert data to numpy arrays
        X_val = np.array(request.validation_data, dtype=np.float32)
        y_val = np.array(request.validation_targets, dtype=np.float32)
        
        start_time = time.time()
        
        # Make predictions
        y_pred = model.predict(X_val)
        
        # Calculate metrics
        metrics = evaluate_regression(
            y_val, y_pred, 
            n_features=X_val.shape[1],
            verbose=False
        )
        
        validation_time = time.time() - start_time
        
        # Filter requested metrics
        filtered_metrics = {
            metric: metrics[metric] 
            for metric in request.metrics 
            if metric in metrics
        }
        
        logger.info(f"Model {model_id} validated on {X_val.shape[0]} samples")
        
        return ValidationResponse(
            model_id=model_id,
            metrics=filtered_metrics,
            n_samples=X_val.shape[0],
            validation_time=validation_time
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Validation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Validation failed: {str(e)}")


@router.get("/models/{model_id}/feature-importance", response_model=FeatureImportanceResponse)
async def get_feature_importance(
    model_id: str,
    feature_names: Optional[List[str]] = None
) -> FeatureImportanceResponse:
    """
    Get feature importance scores for a model.
    
    Args:
        model_id: Model identifier
        feature_names: Optional feature names
        
    Returns:
        Feature importance scores
        
    Raises:
        HTTPException: If model not found
    """
    try:
        if model_id not in model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        model = model_store[model_id]['model']
        
        if model.feature_importances_ is None:
            raise HTTPException(
                status_code=400,
                detail="Feature importances not available for this model"
            )
        
        return FeatureImportanceResponse(
            model_id=model_id,
            feature_importances=model.feature_importances_.tolist(),
            feature_names=feature_names,
            n_features=len(model.feature_importances_)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get feature importance: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")


@router.get("/models/{model_id}/stats", response_model=ModelStatsResponse)
async def get_model_stats(model_id: str) -> ModelStatsResponse:
    """
    Get detailed statistics about a model.
    
    Args:
        model_id: Model identifier
        
    Returns:
        Model statistics
        
    Raises:
        HTTPException: If model not found
    """
    try:
        if model_id not in model_store:
            raise HTTPException(
                status_code=404,
                detail=f"Model {model_id} not found"
            )
        
        model = model_store[model_id]['model']
        
        # Calculate total nodes across all trees
        total_nodes = sum(
            tree.get_n_leaves() * 2 - 1  # Approximate total nodes
            for tree in model.trees_
        )
        
        return ModelStatsResponse(
            model_id=model_id,
            n_trees=model.n_trees,
            average_depth=model.get_average_depth(),
            average_leaves=model.get_average_n_leaves(),
            total_nodes=total_nodes,
            memory_usage=None  # Could implement memory usage estimation
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get model stats: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model stats: {str(e)}")


@router.get("/health")
async def health_check():
    """
    Health check endpoint.
    
    Returns:
        Service health status
    """
    return {
        "status": "healthy",
        "version": "2.0.0",
        "timestamp": datetime.utcnow().isoformat(),
        "models_count": len(model_store)
    }
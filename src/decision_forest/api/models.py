"""
Pydantic models for API request/response schemas.

This module defines the data models used by the FastAPI endpoints
for request validation and response serialization.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, validator
import numpy as np


class TrainingConfig(BaseModel):
    """Configuration for training a decision forest model."""
    
    n_trees: int = Field(default=100, ge=1, le=1000, description="Number of trees in the forest")
    max_depth: Optional[int] = Field(default=None, ge=1, le=100, description="Maximum depth of trees")
    min_samples_split: int = Field(default=2, ge=2, description="Minimum samples to split a node")
    min_samples_leaf: int = Field(default=1, ge=1, description="Minimum samples in a leaf")
    max_features: Optional[Union[int, float, str]] = Field(default="sqrt", description="Number of features to consider")
    bootstrap: bool = Field(default=True, description="Whether to use bootstrap sampling")
    oob_score: bool = Field(default=False, description="Whether to calculate out-of-bag score")
    n_jobs: Optional[int] = Field(default=None, description="Number of parallel jobs")
    random_state: Optional[int] = Field(default=None, description="Random state for reproducibility")
    verbose: int = Field(default=0, ge=0, le=2, description="Verbosity level")
    
    @validator('max_features')
    def validate_max_features(cls, v):
        if isinstance(v, str) and v not in ['sqrt', 'log2']:
            raise ValueError("max_features string must be 'sqrt' or 'log2'")
        if isinstance(v, float) and (v <= 0 or v > 1):
            raise ValueError("max_features float must be between 0 and 1")
        return v


class TrainingRequest(BaseModel):
    """Request model for training a decision forest."""
    
    data: List[List[float]] = Field(description="Training feature data")
    targets: List[float] = Field(description="Training target values")
    config: TrainingConfig = Field(default_factory=TrainingConfig, description="Training configuration")
    model_id: Optional[str] = Field(default=None, description="Optional model identifier")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Training data cannot be empty")
        
        # Check that all rows have the same length
        row_lengths = [len(row) for row in v]
        if len(set(row_lengths)) > 1:
            raise ValueError("All training samples must have the same number of features")
        
        return v
    
    @validator('targets')
    def validate_targets(cls, v, values):
        if not v:
            raise ValueError("Training targets cannot be empty")
        
        # Check that number of targets matches number of samples
        if 'data' in values and len(v) != len(values['data']):
            raise ValueError("Number of targets must match number of training samples")
        
        return v


class PredictionRequest(BaseModel):
    """Request model for making predictions."""
    
    model_id: str = Field(description="Model identifier")
    data: List[List[float]] = Field(description="Feature data for prediction")
    include_uncertainty: bool = Field(default=False, description="Whether to include prediction uncertainty")
    confidence_level: float = Field(default=0.95, ge=0.01, le=0.99, description="Confidence level for intervals")
    
    @validator('data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Prediction data cannot be empty")
        
        # Check that all rows have the same length
        row_lengths = [len(row) for row in v]
        if len(set(row_lengths)) > 1:
            raise ValueError("All prediction samples must have the same number of features")
        
        return v


class PredictionResponse(BaseModel):
    """Response model for predictions."""
    
    model_id: str = Field(description="Model identifier")
    predictions: List[float] = Field(description="Predicted values")
    uncertainties: Optional[List[float]] = Field(default=None, description="Prediction uncertainties")
    confidence_intervals: Optional[List[Dict[str, float]]] = Field(
        default=None, 
        description="Confidence intervals with 'lower' and 'upper' bounds"
    )
    n_samples: int = Field(description="Number of samples predicted")


class TrainingResponse(BaseModel):
    """Response model for training completion."""
    
    model_id: str = Field(description="Model identifier")
    message: str = Field(description="Training completion message")
    training_score: Optional[float] = Field(default=None, description="Training R² score")
    oob_score: Optional[float] = Field(default=None, description="Out-of-bag score")
    n_trees: int = Field(description="Number of trees trained")
    n_features: int = Field(description="Number of features")
    n_samples: int = Field(description="Number of training samples")
    training_time: float = Field(description="Training time in seconds")


class ModelInfo(BaseModel):
    """Information about a trained model."""
    
    model_id: str = Field(description="Model identifier")
    created_at: str = Field(description="Model creation timestamp")
    config: TrainingConfig = Field(description="Training configuration")
    n_features: int = Field(description="Number of features")
    n_samples: int = Field(description="Number of training samples")
    n_trees: int = Field(description="Number of trees")
    training_score: Optional[float] = Field(default=None, description="Training R² score")
    oob_score: Optional[float] = Field(default=None, description="Out-of-bag score")
    average_tree_depth: float = Field(description="Average tree depth")
    average_tree_leaves: float = Field(description="Average number of leaves per tree")
    feature_importances: Optional[List[float]] = Field(default=None, description="Feature importance scores")


class ModelListResponse(BaseModel):
    """Response model for listing models."""
    
    models: List[ModelInfo] = Field(description="List of available models")
    total_count: int = Field(description="Total number of models")


class ModelResponse(BaseModel):
    """Generic response model for model operations."""
    
    success: bool = Field(description="Operation success status")
    message: str = Field(description="Response message")
    model_id: Optional[str] = Field(default=None, description="Model identifier")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Additional response data")


class HealthResponse(BaseModel):
    """Health check response model."""
    
    status: str = Field(description="Service status")
    version: str = Field(description="API version")
    timestamp: str = Field(description="Current timestamp")
    uptime: float = Field(description="Service uptime in seconds")


class ErrorResponse(BaseModel):
    """Error response model."""
    
    error: str = Field(description="Error type")
    message: str = Field(description="Error message")
    details: Optional[Dict[str, Any]] = Field(default=None, description="Additional error details")
    timestamp: str = Field(description="Error timestamp")


class ValidationRequest(BaseModel):
    """Request model for model validation."""
    
    model_id: str = Field(description="Model identifier")
    validation_data: List[List[float]] = Field(description="Validation feature data")
    validation_targets: List[float] = Field(description="Validation target values")
    metrics: Optional[List[str]] = Field(
        default=["r2_score", "rmse", "mae"], 
        description="Metrics to calculate"
    )
    
    @validator('validation_data')
    def validate_data(cls, v):
        if not v:
            raise ValueError("Validation data cannot be empty")
        return v
    
    @validator('validation_targets')
    def validate_targets(cls, v, values):
        if not v:
            raise ValueError("Validation targets cannot be empty")
        
        if 'validation_data' in values and len(v) != len(values['validation_data']):
            raise ValueError("Number of targets must match number of validation samples")
        
        return v


class ValidationResponse(BaseModel):
    """Response model for model validation."""
    
    model_id: str = Field(description="Model identifier")
    metrics: Dict[str, float] = Field(description="Validation metrics")
    n_samples: int = Field(description="Number of validation samples")
    validation_time: float = Field(description="Validation time in seconds")


class FeatureImportanceResponse(BaseModel):
    """Response model for feature importance."""
    
    model_id: str = Field(description="Model identifier")
    feature_importances: List[float] = Field(description="Feature importance scores")
    feature_names: Optional[List[str]] = Field(default=None, description="Feature names if available")
    n_features: int = Field(description="Number of features")


class ModelStatsResponse(BaseModel):
    """Response model for model statistics."""
    
    model_id: str = Field(description="Model identifier")
    n_trees: int = Field(description="Number of trees")
    average_depth: float = Field(description="Average tree depth")
    average_leaves: float = Field(description="Average leaves per tree")
    total_nodes: int = Field(description="Total number of nodes across all trees")
    memory_usage: Optional[float] = Field(default=None, description="Estimated memory usage in MB")
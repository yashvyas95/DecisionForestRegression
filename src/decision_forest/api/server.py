"""
FastAPI server for Decision Forest Regression API.

This module creates and configures the FastAPI application
with all endpoints and middleware.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.openapi.utils import get_openapi

from .endpoints import router
from .models import ErrorResponse

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="Decision Forest Regression API",
    description="""
    A modern REST API for Decision Forest Regression with ensemble learning.
    
    ## Features
    
    * **Train Models**: Train decision forest models with customizable parameters
    * **Make Predictions**: Get predictions with optional uncertainty estimates  
    * **Model Management**: List, view, and delete trained models
    * **Model Validation**: Validate models on test data with comprehensive metrics
    * **Feature Importance**: Get feature importance scores and explanations
    * **Model Statistics**: Detailed statistics about model structure and performance
    
    ## Quick Start
    
    1. **Train a model**:
       ```bash
       curl -X POST "http://localhost:8000/api/v1/train" \\
            -H "Content-Type: application/json" \\
            -d '{"data": [[1,2],[3,4]], "targets": [1.0, 2.0]}'
       ```
    
    2. **Make predictions**:
       ```bash
       curl -X POST "http://localhost:8000/api/v1/predict" \\
            -H "Content-Type: application/json" \\
            -d '{"model_id": "your-model-id", "data": [[1,2]]}'
       ```
    
    3. **List models**:
       ```bash
       curl "http://localhost:8000/api/v1/models"
       ```
    """,
    version="2.0.0",
    contact={
        "name": "Yash Vyas",
        "email": "yash.vyas@example.com",
        "url": "https://github.com/yashvyas/decision-forest-regression",
    },
    license_info={
        "name": "MIT",
        "url": "https://opensource.org/licenses/MIT",
    },
    openapi_tags=[
        {
            "name": "decision-forest",
            "description": "Decision Forest Regression operations",
        },
        {
            "name": "health",
            "description": "Health check and system status",
        },
    ]
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE"],
    allow_headers=["*"],
)

# Store application start time for uptime calculation
app.state.start_time = time.time()

# Include API routes
app.include_router(router)


@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    """Add processing time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response


@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log all incoming requests."""
    start_time = time.time()
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - Client: {request.client.host}")
    
    response = await call_next(request)
    
    # Log response
    process_time = time.time() - start_time
    logger.info(
        f"{request.method} {request.url.path} - "
        f"Status: {response.status_code} - "
        f"Time: {process_time:.3f}s"
    )
    
    return response


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Custom HTTP exception handler."""
    error_response = ErrorResponse(
        error="HTTPException",
        message=exc.detail,
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Handle ValueError exceptions."""
    error_response = ErrorResponse(
        error="ValueError",
        message=str(exc),
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=400,
        content=error_response.dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Handle general exceptions."""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    
    error_response = ErrorResponse(
        error="InternalServerError",
        message="An internal server error occurred",
        details={"exception_type": type(exc).__name__},
        timestamp=datetime.utcnow().isoformat()
    )
    
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
    )


@app.get("/")
async def root():
    """Root endpoint with API information."""
    uptime = time.time() - app.state.start_time
    
    return {
        "name": "Decision Forest Regression API",
        "version": "2.0.0",
        "description": "Modern REST API for decision forest regression",
        "docs": "/docs",
        "redoc": "/redoc",
        "health": "/api/v1/health",
        "uptime": uptime,
        "status": "running"
    }


@app.get("/metrics")
async def metrics():
    """Simple metrics endpoint."""
    from .endpoints import model_store
    
    uptime = time.time() - app.state.start_time
    
    return {
        "uptime": uptime,
        "models_count": len(model_store),
        "timestamp": datetime.utcnow().isoformat()
    }


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema
    
    openapi_schema = get_openapi(
        title=app.title,
        version=app.version,
        description=app.description,
        routes=app.routes,
    )
    
    # Add custom schema information
    openapi_schema["info"]["x-logo"] = {
        "url": "https://raw.githubusercontent.com/yashvyas/decision-forest-regression/main/docs/logo.png"
    }
    
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def run(host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
    """
    Run the FastAPI server.
    
    Args:
        host: Host address to bind to
        port: Port number to listen on
        reload: Enable auto-reload for development
    """
    import uvicorn
    
    logger.info(f"Starting Decision Forest Regression API server on {host}:{port}")
    
    uvicorn.run(
        "decision_forest.api.server:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info"
    )


if __name__ == "__main__":
    run()
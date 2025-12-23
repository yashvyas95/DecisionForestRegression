"""
Command-line interface for Decision Forest Regression.

This module provides a CLI for training models, making predictions,
and managing the Decision Forest Regression system.
"""

import click
import numpy as np
import json
import yaml
import logging
from pathlib import Path
from typing import Optional, List

from .core import DecisionForest
from .utils import load_csv_data, load_sample_data, evaluate_regression
from .api.server import run as run_server


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose output')
@click.option('--config', '-c', type=click.Path(exists=True), help='Configuration file path')
def cli(verbose: bool, config: Optional[str]):
    """Decision Forest Regression CLI."""
    # Setup logging
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load config if provided
    if config:
        click.echo(f"Loading configuration from {config}")
        # Configuration loading logic here


@cli.command()
@click.option('--data', '-d', required=True, help='Path to training data CSV file')
@click.option('--target', '-t', required=True, help='Target column name')
@click.option('--output', '-o', default='model.joblib', help='Output model file path')
@click.option('--n-trees', default=100, help='Number of trees in the forest')
@click.option('--max-depth', default=None, type=int, help='Maximum depth of trees')
@click.option('--test-size', default=0.2, help='Test set size (0.0-1.0)')
@click.option('--random-state', default=42, help='Random state for reproducibility')
def train(data: str, target: str, output: str, n_trees: int, max_depth: Optional[int], 
          test_size: float, random_state: int):
    """Train a decision forest model."""
    click.echo(f"Training decision forest with {n_trees} trees...")
    
    try:
        # Load data
        X_train, X_test, y_train, y_test = load_csv_data(
            data, target, test_size=test_size, random_state=random_state
        )
        
        click.echo(f"Loaded data: {X_train.shape[0]} training samples, {X_test.shape[0]} test samples")
        
        # Create and train model
        model = DecisionForest(
            n_trees=n_trees,
            max_depth=max_depth,
            random_state=random_state,
            verbose=1
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate model
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        click.echo(f"Training R² Score: {train_score:.4f}")
        click.echo(f"Test R² Score: {test_score:.4f}")
        
        if model.oob_score_:
            click.echo(f"Out-of-bag R² Score: {model.oob_score_:.4f}")
        
        # Save model
        model.save(output)
        click.echo(f"Model saved to {output}")
        
    except Exception as e:
        click.echo(f"Training failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--model', '-m', required=True, help='Path to trained model file')
@click.option('--data', '-d', required=True, help='Path to prediction data CSV file')
@click.option('--output', '-o', default='predictions.csv', help='Output predictions file')
@click.option('--uncertainty', is_flag=True, help='Include prediction uncertainty')
def predict(model: str, data: str, output: str, uncertainty: bool):
    """Make predictions using a trained model."""
    click.echo("Making predictions...")
    
    try:
        # Load model
        forest = DecisionForest.load(model)
        click.echo(f"Loaded model with {forest.n_trees} trees")
        
        # Load data
        import pandas as pd
        df = pd.read_csv(data)
        X = df.values.astype(float)
        
        # Make predictions
        predictions = forest.predict(X)
        
        # Create output dataframe
        result_df = pd.DataFrame({
            'prediction': predictions
        })
        
        if uncertainty:
            pred_mean, pred_std = forest.predict_proba(X)
            result_df['uncertainty'] = pred_std
        
        # Save predictions
        result_df.to_csv(output, index=False)
        click.echo(f"Predictions saved to {output}")
        
    except Exception as e:
        click.echo(f"Prediction failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--model', '-m', required=True, help='Path to trained model file')
@click.option('--data', '-d', required=True, help='Path to validation data CSV file')
@click.option('--target', '-t', required=True, help='Target column name')
def evaluate(model: str, data: str, target: str):
    """Evaluate a trained model."""
    click.echo("Evaluating model...")
    
    try:
        # Load model
        forest = DecisionForest.load(model)
        
        # Load validation data
        import pandas as pd
        df = pd.read_csv(data)
        
        if target not in df.columns:
            click.echo(f"Target column '{target}' not found in data", err=True)
            raise click.Abort()
        
        X = df.drop(columns=[target]).values.astype(float)
        y = df[target].values.astype(float)
        
        # Make predictions
        predictions = forest.predict(X)
        
        # Evaluate
        metrics = evaluate_regression(y, predictions, verbose=False)
        
        click.echo("Evaluation Results:")
        for metric, value in metrics.items():
            if isinstance(value, float):
                click.echo(f"  {metric}: {value:.4f}")
            else:
                click.echo(f"  {metric}: {value}")
        
    except Exception as e:
        click.echo(f"Evaluation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--host', default='0.0.0.0', help='Host address')
@click.option('--port', default=8000, help='Port number')
@click.option('--reload', is_flag=True, help='Enable auto-reload')
def serve(host: str, port: int, reload: bool):
    """Start the FastAPI server."""
    click.echo(f"Starting Decision Forest Regression API server on {host}:{port}")
    run_server(host=host, port=port, reload=reload)


@cli.command()
@click.option('--n-samples', default=1000, help='Number of samples')
@click.option('--n-features', default=10, help='Number of features')
@click.option('--output', '-o', default='sample_data.csv', help='Output CSV file')
def generate_data(n_samples: int, n_features: int, output: str):
    """Generate sample dataset for testing."""
    click.echo(f"Generating sample dataset with {n_samples} samples and {n_features} features...")
    
    try:
        X_train, X_test, y_train, y_test = load_sample_data(
            n_samples=n_samples, n_features=n_features, random_state=42
        )
        
        # Combine train and test data
        X = np.vstack([X_train, X_test])
        y = np.hstack([y_train, y_test])
        
        # Create DataFrame
        import pandas as pd
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(n_features)])
        df['target'] = y
        
        # Save to CSV
        df.to_csv(output, index=False)
        click.echo(f"Sample dataset saved to {output}")
        
    except Exception as e:
        click.echo(f"Data generation failed: {e}", err=True)
        raise click.Abort()


@cli.command()
@click.option('--model', '-m', required=True, help='Path to trained model file')
def info(model: str):
    """Display information about a trained model."""
    try:
        forest = DecisionForest.load(model)
        
        click.echo("Model Information:")
        click.echo(f"  Number of trees: {forest.n_trees}")
        click.echo(f"  Number of features: {forest.n_features_}")
        click.echo(f"  Number of training samples: {forest.n_samples_}")
        click.echo(f"  Average tree depth: {forest.get_average_depth():.1f}")
        click.echo(f"  Average leaves per tree: {forest.get_average_n_leaves():.1f}")
        
        if forest.oob_score_:
            click.echo(f"  Out-of-bag score: {forest.oob_score_:.4f}")
        
        if forest.feature_importances_ is not None:
            click.echo("\n  Top 5 Feature Importances:")
            importances = forest.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            for i in range(min(5, len(importances))):
                idx = indices[i]
                click.echo(f"    Feature {idx}: {importances[idx]:.4f}")
        
    except Exception as e:
        click.echo(f"Failed to load model info: {e}", err=True)
        raise click.Abort()


def main():
    """Main entry point for the CLI."""
    cli()


if __name__ == '__main__':
    main()
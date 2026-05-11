"""
Model Manager
=============

Handles model training, persistence, and versioning.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type
import pickle
import pandas as pd
import numpy as np
from loguru import logger

from .forecasters import TimeSeriesForecaster, XGBoostForecaster


class ModelManager:
    """
    Manages ML model lifecycle.
    
    Features:
    - Model training and evaluation
    - Model persistence (save/load)
    - Version control
    - Walk-forward optimization
    - Performance tracking
    """
    
    def __init__(self, models_dir: str = "./models"):
        """
        Initialize model manager.
        
        Args:
            models_dir: Directory for saving models
        """
        self.models_dir = Path(models_dir)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        self.active_models: Dict[str, TimeSeriesForecaster] = {}
        self.model_history: Dict[str, List[Dict]] = {}
        
    def train_model(
        self,
        model: TimeSeriesForecaster,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None,
        model_name: str = None
    ) -> Dict[str, float]:
        """
        Train a model and register it.
        
        Args:
            model: Forecaster instance to train
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            model_name: Unique name for the model
            
        Returns:
            Training metrics
        """
        model_name = model_name or f"{model.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        logger.info(f"Training model: {model_name}")
        
        # Train
        metrics = model.train(X_train, y_train, X_val, y_val)
        
        # Evaluate on validation set
        if X_val is not None and y_val is not None:
            val_metrics = model.evaluate(X_val, y_val)
            metrics.update({f"val_{k}": v for k, v in val_metrics.items()})
        
        # Register model
        self.active_models[model_name] = model
        
        # Track history
        if model_name not in self.model_history:
            self.model_history[model_name] = []
            
        self.model_history[model_name].append({
            'timestamp': datetime.now().isoformat(),
            'metrics': metrics,
            'train_size': len(X_train),
            'val_size': len(X_val) if X_val is not None else 0
        })
        
        logger.info(f"Model {model_name} trained with metrics: {metrics}")
        return metrics
    
    def walk_forward_train(
        self,
        model_class: Type[TimeSeriesForecaster],
        model_params: Dict,
        X: pd.DataFrame,
        y: pd.Series,
        n_splits: int = 5,
        test_size: float = 0.2,
        model_name: str = None
    ) -> Tuple[TimeSeriesForecaster, List[Dict]]:
        """
        Perform walk-forward optimization.
        
        This is the proper way to validate time series models,
        avoiding look-ahead bias.
        
        Args:
            model_class: Forecaster class to use
            model_params: Parameters for model initialization
            X: All features
            y: All targets
            n_splits: Number of walk-forward splits
            test_size: Fraction for each test fold
            model_name: Name for the final model
            
        Returns:
            Trained model and fold metrics
        """
        model_name = model_name or f"wf_{model_class.__name__}"
        
        logger.info(f"Starting walk-forward training with {n_splits} splits")
        
        total_len = len(X)
        test_len = int(total_len * test_size / n_splits)
        
        all_metrics = []
        all_predictions = []
        
        for i in range(n_splits):
            # Calculate split points
            test_start = total_len - (n_splits - i) * test_len
            test_end = test_start + test_len
            
            # Split data
            X_train = X.iloc[:test_start]
            y_train = y.iloc[:test_start]
            X_test = X.iloc[test_start:test_end]
            y_test = y.iloc[test_start:test_end]
            
            # Train model for this fold
            model = model_class(**model_params)
            model.train(X_train, y_train)
            
            # Evaluate
            metrics = model.evaluate(X_test, y_test)
            metrics['fold'] = i + 1
            metrics['train_size'] = len(X_train)
            metrics['test_size'] = len(X_test)
            all_metrics.append(metrics)
            
            # Store predictions
            preds = model.predict(X_test)
            all_predictions.extend(zip(X_test.index, preds))
            
            logger.info(f"Fold {i+1}/{n_splits}: {metrics}")
        
        # Train final model on all data
        final_model = model_class(**model_params)
        final_model.train(X, y)
        
        # Calculate average metrics
        avg_metrics = {}
        for key in all_metrics[0].keys():
            if key not in ['fold', 'train_size', 'test_size']:
                avg_metrics[f"avg_{key}"] = np.mean([m[key] for m in all_metrics])
                avg_metrics[f"std_{key}"] = np.std([m[key] for m in all_metrics])
        
        logger.info(f"Walk-forward complete. Average metrics: {avg_metrics}")
        
        # Register final model
        self.active_models[model_name] = final_model
        self.model_history[model_name] = all_metrics
        
        return final_model, all_metrics
    
    def save_model(
        self,
        model_name: str,
        include_history: bool = True
    ) -> str:
        """
        Save a model to disk.
        
        Args:
            model_name: Name of model to save
            include_history: Whether to save training history
            
        Returns:
            Path to saved model
        """
        if model_name not in self.active_models:
            raise ValueError(f"Model {model_name} not found")
        
        model = self.active_models[model_name]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Create model directory
        model_dir = self.models_dir / model_name
        model_dir.mkdir(exist_ok=True)
        
        # Save model
        model_path = model_dir / f"model_{timestamp}.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            'model_name': model_name,
            'model_type': model.__class__.__name__,
            'saved_at': timestamp,
            'is_trained': model.is_trained,
            'feature_names': model.feature_names
        }
        
        if include_history and model_name in self.model_history:
            metadata['history'] = self.model_history[model_name]
        
        metadata_path = model_dir / f"metadata_{timestamp}.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        # Update latest symlink (conceptually)
        latest_path = model_dir / "latest.pkl"
        if latest_path.exists():
            latest_path.unlink()
        
        # Copy to latest
        import shutil
        shutil.copy(model_path, latest_path)
        
        logger.info(f"Model saved to {model_path}")
        return str(model_path)
    
    def load_model(
        self,
        model_name: str,
        version: str = "latest"
    ) -> TimeSeriesForecaster:
        """
        Load a model from disk.
        
        Args:
            model_name: Name of model to load
            version: Specific version or "latest"
            
        Returns:
            Loaded model
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            raise ValueError(f"Model directory {model_dir} not found")
        
        if version == "latest":
            model_path = model_dir / "latest.pkl"
        else:
            model_path = model_dir / f"model_{version}.pkl"
        
        if not model_path.exists():
            raise ValueError(f"Model file {model_path} not found")
        
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        self.active_models[model_name] = model
        logger.info(f"Model loaded from {model_path}")
        
        return model
    
    def get_model(self, model_name: str) -> TimeSeriesForecaster:
        """Get an active model by name."""
        if model_name not in self.active_models:
            # Try to load from disk
            try:
                return self.load_model(model_name)
            except:
                raise ValueError(f"Model {model_name} not found")
        return self.active_models[model_name]
    
    def list_models(self) -> List[str]:
        """List all available models."""
        # Active models
        models = list(self.active_models.keys())
        
        # Saved models
        for model_dir in self.models_dir.iterdir():
            if model_dir.is_dir() and model_dir.name not in models:
                models.append(model_dir.name)
        
        return models
    
    def get_model_metrics(self, model_name: str) -> List[Dict]:
        """Get training history for a model."""
        return self.model_history.get(model_name, [])
    
    def compare_models(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        model_names: List[str] = None
    ) -> pd.DataFrame:
        """
        Compare performance of multiple models.
        
        Args:
            X_test: Test features
            y_test: Test target
            model_names: Models to compare (default: all active)
            
        Returns:
            Comparison DataFrame
        """
        if model_names is None:
            model_names = list(self.active_models.keys())
        
        results = []
        
        for name in model_names:
            if name not in self.active_models:
                continue
                
            model = self.active_models[name]
            metrics = model.evaluate(X_test, y_test)
            metrics['model_name'] = name
            results.append(metrics)
        
        return pd.DataFrame(results).set_index('model_name')
    
    def cleanup_old_versions(
        self,
        model_name: str,
        keep_n: int = 3
    ):
        """
        Remove old model versions, keeping only the N most recent.
        
        Args:
            model_name: Model to clean up
            keep_n: Number of versions to keep
        """
        model_dir = self.models_dir / model_name
        
        if not model_dir.exists():
            return
        
        # Find all model files
        model_files = sorted(
            model_dir.glob("model_*.pkl"),
            key=lambda x: x.stat().st_mtime,
            reverse=True
        )
        
        # Remove old versions
        for model_file in model_files[keep_n:]:
            model_file.unlink()
            
            # Remove corresponding metadata
            metadata_file = model_file.with_name(
                model_file.name.replace("model_", "metadata_").replace(".pkl", ".json")
            )
            if metadata_file.exists():
                metadata_file.unlink()
        
        logger.info(f"Cleaned up {model_name}, kept {keep_n} versions")

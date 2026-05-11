"""
Time Series Forecasters
=======================

Machine learning models for price prediction and signal generation.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from loguru import logger

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


class TimeSeriesForecaster(ABC):
    """
    Abstract base class for time series forecasters.
    
    All forecasters must implement train(), predict(), and evaluate().
    """
    
    def __init__(self, name: str = "BaseForecaster"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        self.is_trained = False
        self.feature_names: List[str] = []
        
    @abstractmethod
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Train the model."""
        pass
    
    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        pass
    
    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict probabilities (for classification)."""
        pass
    
    def evaluate(
        self,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Dict[str, float]:
        """
        Evaluate model performance.
        
        Returns:
            Dictionary of metrics
        """
        predictions = self.predict(X_test)
        
        metrics = {}
        
        if len(np.unique(y_test)) <= 5:  # Classification
            metrics['accuracy'] = accuracy_score(y_test, predictions)
            metrics['precision'] = precision_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['recall'] = recall_score(y_test, predictions, average='weighted', zero_division=0)
            metrics['f1'] = f1_score(y_test, predictions, average='weighted', zero_division=0)
        else:  # Regression
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            metrics['mse'] = mean_squared_error(y_test, predictions)
            metrics['mae'] = mean_absolute_error(y_test, predictions)
            metrics['r2'] = r2_score(y_test, predictions)
            
        return metrics
    
    def _scale_features(self, X: pd.DataFrame, fit: bool = False) -> np.ndarray:
        """Scale features using StandardScaler."""
        if self.scaler is None:
            return X.values
            
        if fit:
            return self.scaler.fit_transform(X)
        return self.scaler.transform(X)


class XGBoostForecaster(TimeSeriesForecaster):
    """
    XGBoost-based forecaster for classification or regression.
    
    XGBoost is excellent for:
    - Feature importance analysis
    - Handling non-linear relationships
    - Fast training and inference
    """
    
    def __init__(
        self,
        task: str = 'classification',
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs
    ):
        """
        Initialize XGBoost forecaster.
        
        Args:
            task: 'classification' or 'regression'
            n_estimators: Number of boosting rounds
            max_depth: Maximum tree depth
            learning_rate: Boosting learning rate
            **kwargs: Additional XGBoost parameters
        """
        super().__init__(name="XGBoostForecaster")
        
        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available. Install with: pip install xgboost")
        
        self.task = task
        self.params = {
            'n_estimators': n_estimators,
            'max_depth': max_depth,
            'learning_rate': learning_rate,
            'random_state': 42,
            'n_jobs': -1,
            **kwargs
        }
        
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Train XGBoost model."""
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_scaled = self._scale_features(X_train, fit=True)
        
        # Initialize model
        if self.task == 'classification':
            n_classes = len(np.unique(y_train))
            if n_classes == 2:
                self.model = xgb.XGBClassifier(
                    objective='binary:logistic',
                    eval_metric='logloss',
                    **self.params
                )
            else:
                self.model = xgb.XGBClassifier(
                    objective='multi:softprob',
                    num_class=n_classes,
                    eval_metric='mlogloss',
                    **self.params
                )
        else:
            self.model = xgb.XGBRegressor(
                objective='reg:squarederror',
                **self.params
            )
        
        # Prepare validation data
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val_scaled = self._scale_features(X_val)
            eval_set = [(X_val_scaled, y_val)]
        
        # Train
        self.model.fit(
            X_scaled, y_train,
            eval_set=eval_set,
            verbose=False
        )
        
        self.is_trained = True
        
        # Return training metrics
        train_pred = self.model.predict(X_scaled)
        metrics = {'train_accuracy': accuracy_score(y_train, train_pred)}
        
        if X_val is not None:
            val_pred = self.model.predict(X_val_scaled)
            metrics['val_accuracy'] = accuracy_score(y_val, val_pred)
        
        logger.info(f"XGBoost trained: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        X_scaled = self._scale_features(X)
        return self.model.predict(X_scaled)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
            
        X_scaled = self._scale_features(X)
        
        if self.task == 'classification':
            return self.model.predict_proba(X_scaled)
        else:
            # For regression, return predictions as "probabilities"
            return self.model.predict(X_scaled).reshape(-1, 1)
    
    def get_feature_importance(self) -> pd.Series:
        """Get feature importance scores."""
        if not self.is_trained:
            raise ValueError("Model not trained.")
            
        importance = self.model.feature_importances_
        return pd.Series(importance, index=self.feature_names).sort_values(ascending=False)


class LSTMForecaster(TimeSeriesForecaster):
    """
    LSTM-based forecaster for sequence prediction.
    
    LSTM is excellent for:
    - Capturing long-term dependencies
    - Sequential pattern recognition
    - Time series with complex patterns
    """
    
    def __init__(
        self,
        sequence_length: int = 20,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        task: str = 'classification',
        epochs: int = 50,
        batch_size: int = 32,
        learning_rate: float = 0.001
    ):
        """
        Initialize LSTM forecaster.
        
        Args:
            sequence_length: Length of input sequences
            hidden_size: LSTM hidden layer size
            num_layers: Number of LSTM layers
            dropout: Dropout rate
            task: 'classification' or 'regression'
            epochs: Training epochs
            batch_size: Training batch size
            learning_rate: Learning rate
        """
        super().__init__(name="LSTMForecaster")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch not available. Install with: pip install torch")
        
        self.sequence_length = sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.task = task
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.n_classes = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _build_model(self, input_size: int, output_size: int):
        """Build LSTM model."""
        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout if num_layers > 1 else 0,
                    batch_first=True
                )
                
                self.fc = nn.Sequential(
                    nn.Linear(hidden_size, hidden_size // 2),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_size // 2, output_size)
                )
                
            def forward(self, x):
                # x shape: (batch, seq_len, features)
                lstm_out, _ = self.lstm(x)
                # Use last output
                out = self.fc(lstm_out[:, -1, :])
                return out
        
        self.model = LSTMModel(
            input_size, self.hidden_size, self.num_layers, output_size, self.dropout
        ).to(self.device)
        
    def _create_sequences(self, X: np.ndarray, y: np.ndarray = None) -> Tuple:
        """Create sequences for LSTM input."""
        sequences = []
        targets = []
        
        for i in range(len(X) - self.sequence_length):
            sequences.append(X[i:i + self.sequence_length])
            if y is not None:
                targets.append(y[i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        if y is not None:
            targets = np.array(targets)
            return sequences, targets
        return sequences
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Train LSTM model."""
        self.feature_names = list(X_train.columns)
        
        # Scale features
        X_scaled = self._scale_features(X_train, fit=True)
        
        # Create sequences
        X_seq, y_seq = self._create_sequences(X_scaled, y_train.values)
        
        # Determine output size
        if self.task == 'classification':
            self.n_classes = len(np.unique(y_seq))
            output_size = self.n_classes
        else:
            output_size = 1
        
        # Build model
        self._build_model(X_train.shape[1], output_size)
        
        # Loss and optimizer
        if self.task == 'classification':
            criterion = nn.CrossEntropyLoss()
        else:
            criterion = nn.MSELoss()
            
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X_seq).to(self.device)
        
        if self.task == 'classification':
            y_tensor = torch.LongTensor(y_seq).to(self.device)
        else:
            y_tensor = torch.FloatTensor(y_seq).to(self.device)
        
        # Training loop
        self.model.train()
        best_loss = float('inf')
        
        for epoch in range(self.epochs):
            total_loss = 0
            n_batches = 0
            
            # Mini-batch training
            for i in range(0, len(X_tensor), self.batch_size):
                batch_X = X_tensor[i:i + self.batch_size]
                batch_y = y_tensor[i:i + self.batch_size]
                
                optimizer.zero_grad()
                outputs = self.model(batch_X)
                
                if self.task == 'regression':
                    outputs = outputs.squeeze()
                
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                n_batches += 1
            
            avg_loss = total_loss / n_batches
            
            if (epoch + 1) % 10 == 0:
                logger.debug(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
        
        self.is_trained = True
        
        # Calculate training accuracy
        self.model.eval()
        with torch.no_grad():
            train_pred = self.model(X_tensor)
            if self.task == 'classification':
                train_pred = train_pred.argmax(dim=1).cpu().numpy()
            else:
                train_pred = train_pred.cpu().numpy()
        
        metrics = {'train_loss': best_loss}
        if self.task == 'classification':
            metrics['train_accuracy'] = accuracy_score(y_seq, train_pred)
        
        logger.info(f"LSTM trained: {metrics}")
        return metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self._scale_features(X)
        X_seq = self._create_sequences(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.task == 'classification':
                predictions = outputs.argmax(dim=1).cpu().numpy()
            else:
                predictions = outputs.squeeze().cpu().numpy()
        
        # Pad beginning to match original length
        padding = np.full(self.sequence_length, np.nan)
        return np.concatenate([padding, predictions])
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class probabilities."""
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        X_scaled = self._scale_features(X)
        X_seq = self._create_sequences(X_scaled)
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X_seq).to(self.device)
            outputs = self.model(X_tensor)
            
            if self.task == 'classification':
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
            else:
                probs = outputs.cpu().numpy()
        
        # Pad beginning
        padding = np.full((self.sequence_length, probs.shape[1] if len(probs.shape) > 1 else 1), np.nan)
        return np.vstack([padding, probs])


class EnsembleForecaster(TimeSeriesForecaster):
    """
    Ensemble of multiple forecasters.
    
    Combines predictions from multiple models for more robust forecasts.
    """
    
    def __init__(
        self,
        forecasters: List[TimeSeriesForecaster] = None,
        weights: List[float] = None,
        voting: str = 'soft'
    ):
        """
        Initialize ensemble forecaster.
        
        Args:
            forecasters: List of forecaster instances
            weights: Weights for each forecaster (default: equal weights)
            voting: 'soft' (probability averaging) or 'hard' (majority voting)
        """
        super().__init__(name="EnsembleForecaster")
        
        self.forecasters = forecasters or []
        self.weights = weights
        self.voting = voting
        
    def add_forecaster(self, forecaster: TimeSeriesForecaster, weight: float = 1.0):
        """Add a forecaster to the ensemble."""
        self.forecasters.append(forecaster)
        if self.weights is None:
            self.weights = [1.0] * len(self.forecasters)
        else:
            self.weights.append(weight)
    
    def train(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: Optional[pd.DataFrame] = None,
        y_val: Optional[pd.Series] = None
    ) -> Dict[str, float]:
        """Train all forecasters."""
        all_metrics = {}
        
        for i, forecaster in enumerate(self.forecasters):
            logger.info(f"Training {forecaster.name}...")
            metrics = forecaster.train(X_train, y_train, X_val, y_val)
            all_metrics[forecaster.name] = metrics
        
        self.is_trained = True
        return all_metrics
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make ensemble predictions."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")
        
        if self.voting == 'hard':
            # Majority voting
            predictions = []
            for forecaster in self.forecasters:
                pred = forecaster.predict(X)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            # Weighted majority
            return np.apply_along_axis(
                lambda x: np.bincount(x.astype(int), weights=self.weights).argmax(),
                axis=0,
                arr=predictions
            )
        else:
            # Soft voting (probability averaging)
            probs = self.predict_proba(X)
            return probs.argmax(axis=1)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict averaged probabilities."""
        if not self.is_trained:
            raise ValueError("Ensemble not trained.")
        
        # Normalize weights
        weights = np.array(self.weights) / sum(self.weights)
        
        all_probs = []
        for i, forecaster in enumerate(self.forecasters):
            probs = forecaster.predict_proba(X)
            all_probs.append(probs * weights[i])
        
        return np.sum(all_probs, axis=0)

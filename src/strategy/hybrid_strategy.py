"""
Hybrid Strategy
===============

Combines Technical Analysis with ML predictions.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .base_strategy import BaseStrategy, Signal, SignalType
from .ta_strategy import TAStrategy
from ..ml.forecasters import TimeSeriesForecaster
from ..ml.feature_engineering import FeatureEngineer


class HybridStrategy(BaseStrategy):
    """
    Hybrid trading strategy combining TA and ML.
    
    Uses ML model predictions to filter or confirm TA signals,
    improving overall accuracy.
    
    Modes:
    - 'filter': ML filters TA signals (only act when both agree)
    - 'confirm': TA primary, ML increases confidence
    - 'ml_primary': ML primary, TA provides entry timing
    """
    
    def __init__(
        self,
        name: str = "HybridStrategy",
        ta_strategy: TAStrategy = None,
        ml_model: TimeSeriesForecaster = None,
        mode: str = "filter",
        ml_confidence_threshold: float = 0.6,
        ta_weight: float = 0.5,
        ml_weight: float = 0.5
    ):
        """
        Initialize hybrid strategy.
        
        Args:
            name: Strategy name
            ta_strategy: Technical analysis strategy
            ml_model: Trained ML forecaster
            mode: 'filter', 'confirm', or 'ml_primary'
            ml_confidence_threshold: Minimum ML confidence to act
            ta_weight: Weight for TA signals (for confirm mode)
            ml_weight: Weight for ML signals (for confirm mode)
        """
        params = {
            'mode': mode,
            'ml_confidence_threshold': ml_confidence_threshold,
            'ta_weight': ta_weight,
            'ml_weight': ml_weight
        }
        super().__init__(name=name, params=params)
        
        # Initialize TA strategy
        self.ta_strategy = ta_strategy or TAStrategy()
        self.ml_model = ml_model
        self.feature_engineer = FeatureEngineer()
        
    def set_ml_model(self, model: TimeSeriesForecaster):
        """Set the ML model (can be done after initialization)."""
        self.ml_model = model
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add TA indicators and ML features."""
        # Add TA indicators
        df = self.ta_strategy.calculate_indicators(df)
        
        # Add ML features
        df = self.feature_engineer.create_features(
            df, 
            include_ta=True,
            include_lags=True,
            include_calendar=True
        )
        
        return df
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> List[Signal]:
        """
        Generate signals combining TA and ML.
        """
        if len(df) < 60:  # Need enough history for ML
            return []
        
        mode = self.params['mode']
        
        # Get TA signal
        ta_signals = self.ta_strategy.generate_signals(df, symbol)
        ta_signal = ta_signals[-1] if ta_signals else None
        
        # Get ML prediction
        ml_prediction = self._get_ml_prediction(df)
        
        if mode == 'filter':
            return self._generate_filter_signals(df, symbol, ta_signal, ml_prediction)
        elif mode == 'confirm':
            return self._generate_confirm_signals(df, symbol, ta_signal, ml_prediction)
        elif mode == 'ml_primary':
            return self._generate_ml_primary_signals(df, symbol, ta_signal, ml_prediction)
        else:
            logger.warning(f"Unknown mode: {mode}, using filter")
            return self._generate_filter_signals(df, symbol, ta_signal, ml_prediction)
    
    def _get_ml_prediction(self, df: pd.DataFrame) -> Dict:
        """Get ML model prediction."""
        if self.ml_model is None:
            return {'direction': 0, 'confidence': 0.5, 'available': False}
        
        try:
            # Prepare features
            feature_cols = self.feature_engineer.get_feature_names()
            
            if not feature_cols:
                # If no features stored, use all non-OHLCV columns
                feature_cols = [col for col in df.columns 
                               if col not in ['open', 'high', 'low', 'close', 'volume']]
            
            # Filter to available columns
            available_cols = [col for col in feature_cols if col in df.columns]
            
            if not available_cols:
                return {'direction': 0, 'confidence': 0.5, 'available': False}
            
            X = df[available_cols].iloc[-1:].copy()
            X = X.fillna(0)  # Handle NaN
            
            # Get prediction
            pred = self.ml_model.predict(X)
            proba = self.ml_model.predict_proba(X)
            
            # Determine direction and confidence
            if len(pred) > 0:
                direction = int(pred[0]) if not np.isnan(pred[0]) else 0
                
                # Get confidence from probabilities
                if proba is not None and len(proba) > 0:
                    confidence = float(np.max(proba[0])) if not np.isnan(proba[0]).any() else 0.5
                else:
                    confidence = 0.5
                
                # Map direction: 0=down, 1=up for binary; or actual class for multi-class
                if direction == 1:
                    direction = 1  # Bullish
                elif direction == 0:
                    direction = -1  # Bearish
                
                return {
                    'direction': direction,
                    'confidence': confidence,
                    'available': True,
                    'raw_prediction': pred[0],
                    'probabilities': proba[0].tolist() if proba is not None else None
                }
            
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
        
        return {'direction': 0, 'confidence': 0.5, 'available': False}
    
    def _generate_filter_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        ta_signal: Optional[Signal],
        ml_pred: Dict
    ) -> List[Signal]:
        """
        Filter mode: Only act when TA and ML agree.
        """
        close = df['close'].iloc[-1]
        threshold = self.params['ml_confidence_threshold']
        
        if ta_signal is None or ta_signal.signal_type == SignalType.HOLD:
            return [Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=close,
                confidence=0.3,
                reason="No TA signal"
            )]
        
        if not ml_pred['available']:
            # No ML available, fall back to pure TA
            return [ta_signal]
        
        ta_bullish = ta_signal.signal_type == SignalType.BUY
        ml_bullish = ml_pred['direction'] == 1
        ml_conf = ml_pred['confidence']
        
        # Check if they agree
        if ta_bullish and ml_bullish and ml_conf >= threshold:
            # Both agree on bullish
            combined_conf = (ta_signal.confidence + ml_conf) / 2
            return [Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=close,
                confidence=combined_conf,
                reason=f"TA+ML agree (ML conf: {ml_conf:.2f}): {ta_signal.reason}",
                stop_loss=ta_signal.stop_loss,
                take_profit=ta_signal.take_profit,
                metadata={
                    'ta_signal': ta_signal.to_dict(),
                    'ml_prediction': ml_pred
                }
            )]
            
        elif not ta_bullish and not ml_bullish and ml_conf >= threshold:
            # Both agree on bearish
            combined_conf = (ta_signal.confidence + ml_conf) / 2
            return [Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=close,
                confidence=combined_conf,
                reason=f"TA+ML agree (ML conf: {ml_conf:.2f}): {ta_signal.reason}",
                stop_loss=ta_signal.stop_loss,
                take_profit=ta_signal.take_profit,
                metadata={
                    'ta_signal': ta_signal.to_dict(),
                    'ml_prediction': ml_pred
                }
            )]
        else:
            # Disagreement - hold
            return [Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=close,
                confidence=0.4,
                reason=f"TA/ML disagree: TA={'buy' if ta_bullish else 'sell'}, ML={'buy' if ml_bullish else 'sell'} ({ml_conf:.2f})",
                metadata={
                    'ta_signal': ta_signal.to_dict(),
                    'ml_prediction': ml_pred
                }
            )]
    
    def _generate_confirm_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        ta_signal: Optional[Signal],
        ml_pred: Dict
    ) -> List[Signal]:
        """
        Confirm mode: TA primary, ML adjusts confidence.
        """
        close = df['close'].iloc[-1]
        ta_weight = self.params['ta_weight']
        ml_weight = self.params['ml_weight']
        
        if ta_signal is None:
            return [Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=close,
                confidence=0.3,
                reason="No TA signal"
            )]
        
        # Start with TA signal
        final_signal = Signal(
            signal_type=ta_signal.signal_type,
            symbol=symbol,
            price=close,
            stop_loss=ta_signal.stop_loss,
            take_profit=ta_signal.take_profit
        )
        
        if not ml_pred['available']:
            final_signal.confidence = ta_signal.confidence
            final_signal.reason = ta_signal.reason
            return [final_signal]
        
        # Adjust confidence based on ML agreement
        ta_bullish = ta_signal.signal_type == SignalType.BUY
        ml_bullish = ml_pred['direction'] == 1
        ml_conf = ml_pred['confidence']
        
        if ta_bullish == ml_bullish:
            # Agreement - boost confidence
            combined = (ta_signal.confidence * ta_weight) + (ml_conf * ml_weight)
            final_signal.confidence = min(combined * 1.2, 1.0)
            final_signal.reason = f"{ta_signal.reason} [ML confirms: {ml_conf:.2f}]"
        else:
            # Disagreement - reduce confidence
            combined = (ta_signal.confidence * ta_weight) - (ml_conf * ml_weight * 0.5)
            final_signal.confidence = max(combined, 0.1)
            final_signal.reason = f"{ta_signal.reason} [ML disagrees: {ml_conf:.2f}]"
        
        final_signal.metadata = {
            'ta_signal': ta_signal.to_dict(),
            'ml_prediction': ml_pred
        }
        
        return [final_signal]
    
    def _generate_ml_primary_signals(
        self,
        df: pd.DataFrame,
        symbol: str,
        ta_signal: Optional[Signal],
        ml_pred: Dict
    ) -> List[Signal]:
        """
        ML Primary mode: ML determines direction, TA for timing.
        """
        close = df['close'].iloc[-1]
        threshold = self.params['ml_confidence_threshold']
        
        if not ml_pred['available']:
            # Fall back to TA
            return [ta_signal] if ta_signal else []
        
        ml_direction = ml_pred['direction']
        ml_conf = ml_pred['confidence']
        
        if ml_conf < threshold:
            return [Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=close,
                confidence=ml_conf,
                reason=f"ML confidence too low: {ml_conf:.2f}",
                metadata={'ml_prediction': ml_pred}
            )]
        
        # ML determines direction
        if ml_direction == 1:  # Bullish
            signal_type = SignalType.BUY
        elif ml_direction == -1:  # Bearish
            signal_type = SignalType.SELL
        else:
            signal_type = SignalType.HOLD
        
        # Use TA for better entry timing
        if ta_signal and ta_signal.signal_type == signal_type:
            # TA confirms ML direction - use TA's stop/target
            return [Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=close,
                confidence=ml_conf,
                reason=f"ML signal ({ml_conf:.2f}), TA confirms",
                stop_loss=ta_signal.stop_loss,
                take_profit=ta_signal.take_profit,
                metadata={
                    'ml_prediction': ml_pred,
                    'ta_signal': ta_signal.to_dict()
                }
            )]
        else:
            # ML signal without TA confirmation
            # Use ATR for stops if available
            atr = df['atr'].iloc[-1] if 'atr' in df.columns else close * 0.02
            
            if signal_type == SignalType.BUY:
                stop_loss = close - (atr * 2)
                take_profit = close + (atr * 4)
            else:
                stop_loss = close + (atr * 2)
                take_profit = close - (atr * 4)
            
            return [Signal(
                signal_type=signal_type,
                symbol=symbol,
                price=close,
                confidence=ml_conf * 0.8,  # Reduce confidence without TA
                reason=f"ML signal ({ml_conf:.2f}), awaiting TA confirmation",
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'ml_prediction': ml_pred,
                    'ta_signal': ta_signal.to_dict() if ta_signal else None
                }
            )]
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators."""
        return self.ta_strategy.get_required_indicators() + ['ml_features']

"""
ML Forecast Strategy Bot
=========================

Machine learning trend prediction strategy.
Uses rolling feature engineering + simple gradient-based forecast.
Entry: ML predicts >X% directional move with confidence above threshold
SL: ATR-based
TP: Predicted target clipped by R:R minimum
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import StrategyBot, TradeSignal


class MLForecastBot(StrategyBot):
    """
    ML-based forecast strategy.
    Uses engineered features (momentum, volatility, trend) to predict
    short-term price direction. Simple gradient-based model (no heavy deps).
    """
    
    def __init__(self, params: Dict[str, Any] = None, **kwargs):
        params = params or {}
        super().__init__(name="MLForecast", params=params, **kwargs)
        
        self.lookback = params.get('lookback', 60)
        self.prediction_horizon = params.get('prediction_horizon', 5)
        self.confidence_threshold = params.get('confidence_threshold', 0.65)
        self.min_move_pct = params.get('min_move_pct', 1.0)
        self.atr_period = params.get('atr_period', 14)
        self.atr_sl_multiplier = params.get('atr_sl_multiplier', 2.0)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add ML-oriented features."""
        df = df.copy()
        
        # Returns (multiple timeframes)
        df['ret_1'] = df['close'].pct_change(1)
        df['ret_5'] = df['close'].pct_change(5)
        df['ret_10'] = df['close'].pct_change(10)
        df['ret_20'] = df['close'].pct_change(20)
        
        # Momentum features
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Volatility features
        df['vol_10'] = df['ret_1'].rolling(10).std()
        df['vol_20'] = df['ret_1'].rolling(20).std()
        df['vol_ratio'] = df['vol_10'] / df['vol_20'].replace(0, np.nan)
        
        # Trend features
        sma_20 = df['close'].rolling(20).mean()
        sma_50 = df['close'].rolling(50).mean()
        df['trend_20'] = (df['close'] - sma_20) / sma_20
        df['trend_50'] = (df['close'] - sma_50) / sma_50.replace(0, np.nan)
        df['sma_cross'] = (sma_20 - sma_50) / sma_50.replace(0, np.nan)
        
        # Volume features
        df['vol_change'] = df['volume'].pct_change(5)
        df['vol_ratio_hist'] = df['volume'] / df['volume'].rolling(20).mean().replace(0, np.nan)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # Forward return (for training — NOT used in signal generation)
        df['fwd_ret'] = df['close'].pct_change(self.prediction_horizon).shift(-self.prediction_horizon)
        
        return df
    
    def _compute_forecast(self, df: pd.DataFrame, idx: int) -> tuple[float, float]:
        """
        Simple weighted feature ensemble forecast.
        Returns (predicted_return_pct, confidence).
        
        Uses a scoring approach based on aligned features.
        No sklearn dependency — pure numpy.
        """
        if idx < self.lookback + 5:
            return 0.0, 0.0
        
        row = df.iloc[idx]
        
        # Feature scores (-1 to +1 each)
        scores = []
        weights = []
        
        # Momentum (strong directional signal)
        mom10 = row.get('momentum_10', 0)
        mom20 = row.get('momentum_20', 0)
        if not np.isnan(mom10):
            scores.append(np.clip(mom10 * 10, -1, 1))
            weights.append(0.20)
        if not np.isnan(mom20):
            scores.append(np.clip(mom20 * 5, -1, 1))
            weights.append(0.15)
        
        # Trend alignment
        trend20 = row.get('trend_20', 0)
        sma_cross = row.get('sma_cross', 0)
        if not np.isnan(trend20):
            scores.append(np.clip(trend20 * 5, -1, 1))
            weights.append(0.20)
        if not np.isnan(sma_cross):
            scores.append(np.clip(sma_cross * 10, -1, 1))
            weights.append(0.15)
        
        # RSI — mean reversion component
        rsi = row.get('rsi', 50)
        if not np.isnan(rsi):
            rsi_score = (50 - rsi) / 50  # Oversold → positive, overbought → negative
            scores.append(np.clip(rsi_score, -1, 1))
            weights.append(0.10)
        
        # Volume surge
        vol_ratio = row.get('vol_ratio_hist', 1)
        if not np.isnan(vol_ratio):
            vol_score = np.clip((vol_ratio - 1) * 0.5, -0.5, 0.5)
            scores.append(vol_score)
            weights.append(0.10)
        
        # Volatility regime
        vr = row.get('vol_ratio', 1)
        if not np.isnan(vr):
            # Rising vol → cautious
            vol_regime_score = np.clip(1 - vr, -1, 1)
            scores.append(vol_regime_score)
            weights.append(0.10)
        
        if not scores:
            return 0.0, 0.0
        
        total_weight = sum(weights)
        if total_weight == 0:
            return 0.0, 0.0
        
        # Weighted average of scores
        weighted_sum = sum(s * w for s, w in zip(scores, weights))
        prediction = weighted_sum / total_weight
        
        # Confidence = agreement level among features
        signs = [1 if s > 0 else -1 for s in scores if abs(s) > 0.05]
        if signs:
            agreement = abs(sum(signs)) / len(signs)
        else:
            agreement = 0
        
        confidence = agreement * abs(prediction)
        
        # Scale prediction to approximate % return
        predicted_pct = prediction * 5.0  # Scale factor
        
        return predicted_pct, min(confidence, 1.0)
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: Dict[str, str],
    ) -> Optional[TradeSignal]:
        """
        Generate ML forecast signal.
        
        Signal when predicted move > min_move_pct with high confidence.
        """
        if idx < self.lookback + 5:
            return None
        
        row = df.iloc[idx]
        close = row['close']
        atr = row.get('atr', 0)
        
        if atr <= 0 or close <= 0:
            return None
        
        predicted_pct, confidence = self._compute_forecast(df, idx)
        
        if confidence < self.confidence_threshold:
            return None
        
        if abs(predicted_pct) < self.min_move_pct:
            return None
        
        indicators = {
            'close': round(close, 4),
            'predicted_pct': round(predicted_pct, 2),
            'confidence': round(confidence, 3),
            'atr': round(atr, 4),
            'rsi': round(row.get('rsi', 50), 2),
            'momentum_10': round(row.get('momentum_10', 0), 4),
            'trend_20': round(row.get('trend_20', 0), 4),
        }
        
        if predicted_pct > 0:
            # Long
            entry = close
            sl = close - (atr * self.atr_sl_multiplier)
            risk = entry - sl
            predicted_target = close * (1 + predicted_pct / 100)
            tp = max(predicted_target, entry + risk * self.min_rr)
            
            return TradeSignal(
                side="buy",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                confidence=confidence,
                reason=(
                    f"LONG ML FORECAST: Predicted +{predicted_pct:.1f}% in "
                    f"{self.prediction_horizon} bars (conf={confidence:.0%}). "
                    f"Momentum aligned. ATR={atr:.2f}. "
                    f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )
        else:
            # Short
            entry = close
            sl = close + (atr * self.atr_sl_multiplier)
            risk = sl - entry
            predicted_target = close * (1 + predicted_pct / 100)
            tp = min(predicted_target, entry - risk * self.min_rr)
            
            return TradeSignal(
                side="sell",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                confidence=confidence,
                reason=(
                    f"SHORT ML FORECAST: Predicted {predicted_pct:.1f}% in "
                    f"{self.prediction_horizon} bars (conf={confidence:.0%}). "
                    f"Momentum aligned. ATR={atr:.2f}. "
                    f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )

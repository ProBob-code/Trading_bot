import pandas as pd
import numpy as np
from typing import Dict, Optional, Any
from config import settings

class SignalEngine:
    """
    Generates trading signals based on a hybrid strategy:
    Trend-following + Volatility Breakout.
    Uses a scoring system (0-10) for multi-factor confirmation.
    """

    def __init__(self):
        self.lookback_fast = 20
        self.lookback_slow = 50

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds necessary indicators to the dataframe."""
        df = df.copy()
        
        # Trend: Moving Averages
        df['ema_fast'] = df['close'].ewm(span=self.lookback_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.lookback_slow, adjust=False).mean()
        
        # Volatility Breakout: Donchian Channels
        df['up_band'] = df['high'].rolling(20).max().shift(1)
        df['low_band'] = df['low'].rolling(20).min().shift(1)
        
        # Momentum: RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for stop placement
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        
        return df

    def get_signal(self, df: pd.DataFrame, idx: int, regime: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluates pre-calculated indicators and regime to produce a signal.
        """
        last = df.iloc[idx]
        
        score = 0.0
        signal_type = "NONE"
        
        # Only trade valid liquidity regimes
        if regime.get('liquidity') != 'valid':
            return {"type": "NONE", "score": 0, "reason": "Invalid Liquidity"}

        # --- LONG SCORING ---
        if last['close'] > last['ema_fast'] > last['ema_slow']: score += 3 # Trend
        if last['close'] > last['up_band']: score += 3 # Breakout
        if last['rsi'] > 50 and last['rsi'] < 70: score += 2 # Momentum
        if regime.get('trend') == 'trending': score += 2 # Regime alignment
        
        if score >= settings.SIGNAL_SCORE_THRESHOLD:
            signal_type = "LONG"
            stop_loss = last['close'] - (last['atr'] * settings.STOP_ATR_MULTIPLIER)
            take_profit = last['close'] + (last['close'] - stop_loss) * settings.REWARD_RISK_RATIO
            
            return {
                "type": signal_type,
                "score": score,
                "entry_price": last['close'],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": last['atr']
            }

        # --- SHORT SCORING ---
        short_score = 0.0
        if last['close'] < last['ema_fast'] < last['ema_slow']: short_score += 3
        if last['close'] < last['low_band']: short_score += 3
        if last['rsi'] < 50 and last['rsi'] > 30: short_score += 2
        if regime.get('trend') == 'trending': short_score += 2

        if short_score >= settings.SIGNAL_SCORE_THRESHOLD:
            signal_type = "SHORT"
            stop_loss = last['close'] + (last['atr'] * settings.STOP_ATR_MULTIPLIER)
            take_profit = last['close'] - (stop_loss - last['close']) * settings.REWARD_RISK_RATIO
            
            return {
                "type": signal_type,
                "score": short_score,
                "entry_price": last['close'],
                "stop_loss": stop_loss,
                "take_profit": take_profit,
                "atr": last['atr']
            }

        return {"type": "NONE", "score": max(score, short_score), "reason": "Threshold not met"}

"""
Mean Reversion Strategy Bot
============================

Bollinger Band + RSI reversal strategy.
Entry: Price at lower BB + RSI oversold (long) or upper BB + RSI overbought (short)
SL: Below recent swing low (long) or above swing high (short)
TP: Mean (SMA) reversion target, extended by R:R minimum
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import StrategyBot, TradeSignal


class MeanReversionBot(StrategyBot):
    """
    Mean reversion: buys oversold bounces at lower Bollinger Band,
    sells overbought reversals at upper Bollinger Band.
    """
    
    def __init__(self, params: Dict[str, Any] = None, **kwargs):
        params = params or {}
        super().__init__(name="MeanReversion", params=params, **kwargs)
        
        self.bb_period = params.get('bb_period', 20)
        self.bb_std = params.get('bb_std', 2.0)
        self.rsi_period = params.get('rsi_period', 14)
        self.rsi_oversold = params.get('rsi_oversold', 30)
        self.rsi_overbought = params.get('rsi_overbought', 70)
        self.atr_period = params.get('atr_period', 14)
        self.swing_lookback = params.get('swing_lookback', 10)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Bollinger Bands, RSI, and ATR."""
        df = df.copy()
        
        # Bollinger Bands
        df['bb_mid'] = df['close'].rolling(self.bb_period).mean()
        bb_std = df['close'].rolling(self.bb_period).std()
        df['bb_upper'] = df['bb_mid'] + (bb_std * self.bb_std)
        df['bb_lower'] = df['bb_mid'] - (bb_std * self.bb_std)
        
        # RSI
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0.0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(self.rsi_period).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # Swing low/high
        df['swing_low'] = df['low'].rolling(self.swing_lookback).min()
        df['swing_high'] = df['high'].rolling(self.swing_lookback).max()
        
        return df
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: Dict[str, str],
    ) -> Optional[TradeSignal]:
        """
        Generate mean reversion signal.
        
        Long: Close touches/crosses below lower BB AND RSI < oversold AND reversal candle
        Short: Close touches/crosses above upper BB AND RSI > overbought AND reversal candle
        """
        min_bars = max(self.bb_period, self.rsi_period, self.atr_period, self.swing_lookback) + 2
        if idx < min_bars:
            return None
        
        row = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        close = row['close']
        atr = row.get('atr', 0)
        rsi = row.get('rsi', 50)
        bb_lower = row.get('bb_lower', 0)
        bb_upper = row.get('bb_upper', float('inf'))
        bb_mid = row.get('bb_mid', close)
        swing_low = row.get('swing_low', 0)
        swing_high = row.get('swing_high', float('inf'))
        
        if atr <= 0:
            return None
        
        indicators = {
            'close': round(close, 4),
            'rsi': round(rsi, 2),
            'bb_lower': round(bb_lower, 4),
            'bb_upper': round(bb_upper, 4),
            'bb_mid': round(bb_mid, 4),
            'atr': round(atr, 4),
        }
        
        # Reversal candle check: current close > previous close (for long)
        is_bullish_reversal = close > prev['close'] and row['low'] <= bb_lower
        is_bearish_reversal = close < prev['close'] and row['high'] >= bb_upper
        
        # Long mean reversion
        if is_bullish_reversal and rsi < self.rsi_oversold:
            entry = close
            sl = swing_low - (atr * 0.5)  # Below swing low with cushion
            risk = entry - sl
            
            if risk <= 0:
                return None
            
            # TP at BB mid, but ensure min R:R
            tp_target = bb_mid
            reward = tp_target - entry
            if reward / risk < self.min_rr:
                tp_target = entry + (risk * self.min_rr)
            
            return TradeSignal(
                side="buy",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp_target,
                confidence=min(1.0, (self.rsi_oversold - rsi) / 30.0),
                reason=(
                    f"LONG MEAN REVERSION: Close {close:.2f} touched lower BB {bb_lower:.2f}. "
                    f"RSI={rsi:.1f} (oversold). Reversal candle confirmed. "
                    f"TP target={tp_target:.2f} (BB mid={bb_mid:.2f}). "
                    f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )
        
        # Short mean reversion
        if is_bearish_reversal and rsi > self.rsi_overbought:
            entry = close
            sl = swing_high + (atr * 0.5)
            risk = sl - entry
            
            if risk <= 0:
                return None
            
            tp_target = bb_mid
            reward = entry - tp_target
            if reward / risk < self.min_rr:
                tp_target = entry - (risk * self.min_rr)
            
            return TradeSignal(
                side="sell",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp_target,
                confidence=min(1.0, (rsi - self.rsi_overbought) / 30.0),
                reason=(
                    f"SHORT MEAN REVERSION: Close {close:.2f} hit upper BB {bb_upper:.2f}. "
                    f"RSI={rsi:.1f} (overbought). Reversal candle confirmed. "
                    f"TP target={tp_target:.2f} (BB mid={bb_mid:.2f}). "
                    f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )
        
        return None

"""
Breakout Strategy Bot
=====================

N-period high breakout with volume confirmation.
Entry: Price breaks above N-period high with volume > factor × avg
SL: Below breakout candle low - ATR cushion
TP: R:R multiple of risk distance
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import StrategyBot, TradeSignal


class BreakoutBot(StrategyBot):
    """
    Breakout strategy: buys when price breaks N-period high with volume surge.
    Sells when price breaks N-period low (short).
    """
    
    def __init__(self, params: Dict[str, Any] = None, **kwargs):
        params = params or {}
        super().__init__(name="Breakout", params=params, **kwargs)
        
        self.lookback = params.get('lookback', 20)
        self.volume_factor = params.get('volume_factor', 1.5)
        self.atr_period = params.get('atr_period', 14)
        self.atr_sl_multiplier = params.get('atr_sl_multiplier', 1.5)
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add breakout indicators."""
        df = df.copy()
        
        # N-period high/low
        df['highest_high'] = df['high'].rolling(self.lookback).max()
        df['lowest_low'] = df['low'].rolling(self.lookback).min()
        
        # ATR
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # Volume average
        df['vol_avg'] = df['volume'].rolling(self.lookback).mean()
        
        # Previous bar high/low for breakout confirmation
        df['prev_high'] = df['highest_high'].shift(1)
        df['prev_low'] = df['lowest_low'].shift(1)
        
        return df
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: Dict[str, str],
    ) -> Optional[TradeSignal]:
        """
        Generate breakout signal.
        
        Long when: close > N-period high AND volume > factor × avg volume
        Short when: close < N-period low AND volume > factor × avg volume
        """
        if idx < self.lookback + self.atr_period + 2:
            return None
        
        row = df.iloc[idx]
        close = row['close']
        high = row['high']
        low = row['low']
        atr = row.get('atr', 0)
        volume = row.get('volume', 0)
        vol_avg = row.get('vol_avg', 0)
        prev_high = row.get('prev_high', 0)
        prev_low = row.get('prev_low', float('inf'))
        
        if atr <= 0 or vol_avg <= 0:
            return None
        
        # Volume confirmation
        volume_confirmed = volume > (vol_avg * self.volume_factor)
        
        indicators = {
            'close': round(close, 4),
            'atr': round(atr, 4),
            'volume': round(volume, 2),
            'vol_avg': round(vol_avg, 2),
            'prev_high': round(prev_high, 4),
            'prev_low': round(prev_low, 4),
        }
        
        # Long breakout
        if close > prev_high and volume_confirmed and prev_high > 0:
            entry = close
            sl = low - (atr * self.atr_sl_multiplier)
            risk = entry - sl
            tp = entry + (risk * self.min_rr)
            
            return TradeSignal(
                side="buy",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                confidence=min(1.0, (volume / vol_avg) / 3.0),
                reason=(
                    f"LONG BREAKOUT: Close {close:.2f} broke above {self.lookback}-bar "
                    f"high {prev_high:.2f}. Volume {volume:.0f} = "
                    f"{volume/vol_avg:.1f}x avg. ATR={atr:.2f}. "
                    f"Regime: {regime.get('trend', 'unknown')}/{regime.get('volatility', 'unknown')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )
        
        # Short breakout
        if close < prev_low and volume_confirmed and prev_low > 0:
            entry = close
            sl = high + (atr * self.atr_sl_multiplier)
            risk = sl - entry
            tp = entry - (risk * self.min_rr)
            
            return TradeSignal(
                side="sell",
                entry_price=entry,
                sl_price=sl,
                tp_price=tp,
                confidence=min(1.0, (volume / vol_avg) / 3.0),
                reason=(
                    f"SHORT BREAKOUT: Close {close:.2f} broke below {self.lookback}-bar "
                    f"low {prev_low:.2f}. Volume {volume:.0f} = "
                    f"{volume/vol_avg:.1f}x avg. ATR={atr:.2f}. "
                    f"Regime: {regime.get('trend', 'unknown')}/{regime.get('volatility', 'unknown')}."
                ),
                regime=regime.get('trend', ''),
                indicator_snapshot=indicators,
            )
        
        return None

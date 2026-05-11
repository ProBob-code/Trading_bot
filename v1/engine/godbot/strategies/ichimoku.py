"""
Ichimoku Strategy Bot
=====================

Ichimoku Cloud strategy with TK cross and Chikou confirmation.
Entry: Tenkan crosses Kijun above cloud + Chikou above price
SL: Below Kijun-sen or cloud bottom
TP: Next resistance / R:R target
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any

from .base import StrategyBot, TradeSignal


class IchimokuBot(StrategyBot):
    """
    Ichimoku Cloud strategy.
    Long: TK cross above cloud, Chikou confirms.
    Short: TK cross below cloud, Chikou confirms.
    """
    
    def __init__(self, params: Dict[str, Any] = None, **kwargs):
        params = params or {}
        super().__init__(name="Ichimoku", params=params, **kwargs)
        
        self.conversion_period = params.get('conversion', 9)
        self.base_period = params.get('base', 26)
        self.span_b_period = params.get('span_b', 52)
        self.atr_period = params.get('atr_period', 14)
    
    def _donchian(self, series_high: pd.Series, series_low: pd.Series, period: int):
        """Donchian channel midline (used for Tenkan/Kijun)."""
        return (series_high.rolling(period).max() + series_low.rolling(period).min()) / 2
    
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add Ichimoku components."""
        df = df.copy()
        
        # Tenkan-sen (Conversion Line)
        df['tenkan'] = self._donchian(df['high'], df['low'], self.conversion_period)
        
        # Kijun-sen (Base Line)
        df['kijun'] = self._donchian(df['high'], df['low'], self.base_period)
        
        # Senkou Span A (Leading Span A) — shifted forward
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(self.base_period)
        
        # Senkou Span B (Leading Span B) — shifted forward
        span_b_raw = self._donchian(df['high'], df['low'], self.span_b_period)
        df['span_b'] = span_b_raw.shift(self.base_period)
        
        # Chikou Span (Lagging Span) — current close shifted back
        df['chikou'] = df['close'].shift(-self.base_period)
        
        # Cloud boundaries
        df['cloud_top'] = df[['span_a', 'span_b']].max(axis=1)
        df['cloud_bottom'] = df[['span_a', 'span_b']].min(axis=1)
        
        # ATR for SL
        tr = pd.concat([
            df['high'] - df['low'],
            (df['high'] - df['close'].shift(1)).abs(),
            (df['low'] - df['close'].shift(1)).abs(),
        ], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # TK cross signals
        df['tenkan_above'] = df['tenkan'] > df['kijun']
        df['tenkan_above_prev'] = df['tenkan_above'].shift(1)
        
        return df
    
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: Dict[str, str],
    ) -> Optional[TradeSignal]:
        """
        Generate Ichimoku signal.
        
        Long: Tenkan crosses above Kijun + price above cloud + Chikou above price
        Short: Tenkan crosses below Kijun + price below cloud + Chikou below price
        """
        min_bars = self.span_b_period + self.base_period + 5
        if idx < min_bars or idx >= len(df) - self.base_period:
            return None
        
        row = df.iloc[idx]
        close = row['close']
        tenkan = row.get('tenkan', 0)
        kijun = row.get('kijun', 0)
        cloud_top = row.get('cloud_top', 0)
        cloud_bottom = row.get('cloud_bottom', 0)
        atr = row.get('atr', 0)
        tenkan_above = row.get('tenkan_above', False)
        tenkan_above_prev = row.get('tenkan_above_prev', False)
        
        # Chikou confirmation: current close vs price 26 bars ago
        chikou_ref_idx = idx - self.base_period
        chikou_above = False
        chikou_below = False
        if chikou_ref_idx >= 0:
            chikou_ref_price = df.iloc[chikou_ref_idx]['close']
            chikou_above = close > chikou_ref_price
            chikou_below = close < chikou_ref_price
        
        if atr <= 0 or tenkan == 0 or kijun == 0:
            return None
        
        indicators = {
            'close': round(close, 4),
            'tenkan': round(tenkan, 4),
            'kijun': round(kijun, 4),
            'cloud_top': round(cloud_top, 4),
            'cloud_bottom': round(cloud_bottom, 4),
            'atr': round(atr, 4),
        }
        
        # Bullish TK cross above cloud
        if tenkan_above and not tenkan_above_prev:
            above_cloud = close > cloud_top
            if above_cloud and chikou_above:
                entry = close
                sl = max(kijun, cloud_bottom) - (atr * 0.5)
                risk = entry - sl
                
                if risk <= 0:
                    return None
                
                tp = entry + (risk * self.min_rr)
                
                return TradeSignal(
                    side="buy",
                    entry_price=entry,
                    sl_price=sl,
                    tp_price=tp,
                    confidence=0.7 if regime.get('trend') == 'trending' else 0.5,
                    reason=(
                        f"LONG ICHIMOKU: TK bullish cross (T={tenkan:.2f} > K={kijun:.2f}). "
                        f"Price {close:.2f} above cloud ({cloud_top:.2f}). "
                        f"Chikou confirms. SL at {sl:.2f}. "
                        f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                    ),
                    regime=regime.get('trend', ''),
                    indicator_snapshot=indicators,
                )
        
        # Bearish TK cross below cloud
        if not tenkan_above and tenkan_above_prev:
            below_cloud = close < cloud_bottom
            if below_cloud and chikou_below:
                entry = close
                sl = min(kijun, cloud_top) + (atr * 0.5)
                risk = sl - entry
                
                if risk <= 0:
                    return None
                
                tp = entry - (risk * self.min_rr)
                
                return TradeSignal(
                    side="sell",
                    entry_price=entry,
                    sl_price=sl,
                    tp_price=tp,
                    confidence=0.7 if regime.get('trend') == 'trending' else 0.5,
                    reason=(
                        f"SHORT ICHIMOKU: TK bearish cross (T={tenkan:.2f} < K={kijun:.2f}). "
                        f"Price {close:.2f} below cloud ({cloud_bottom:.2f}). "
                        f"Chikou confirms. SL at {sl:.2f}. "
                        f"Regime: {regime.get('trend', '?')}/{regime.get('volatility', '?')}."
                    ),
                    regime=regime.get('trend', ''),
                    indicator_snapshot=indicators,
                )
        
        return None

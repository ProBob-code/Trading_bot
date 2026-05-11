"""
V2 Regime Detector
====================

Detects market regime from recent price data:
- TRENDING: Strong directional movement
- RANGING: Sideways / mean-reverting
- VOLATILE: High volatility breakout

Used to filter incompatible strategies before execution.
"""

import math
from typing import List, Dict, Optional
from loguru import logger


class MarketRegime:
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"
    UNKNOWN = "UNKNOWN"


class RegimeDetector:
    """
    Detects market regime from price history.
    
    Method:
    - ADX proxy (slope strength) for trend detection
    - ATR ratio for volatility regime
    - Bollinger Band width for ranging detection
    """
    
    def __init__(
        self,
        lookback: int = 20,
        trend_threshold: float = 0.6,
        volatility_threshold: float = 2.0,
    ):
        """
        Args:
            lookback: Number of candles for regime computation
            trend_threshold: R² threshold for trending (0-1)
            volatility_threshold: ATR/avg_ATR ratio for high vol
        """
        self.lookback = lookback
        self.trend_threshold = trend_threshold
        self.volatility_threshold = volatility_threshold
    
    def detect(self, closes: List[float], highs: List[float] = None, lows: List[float] = None) -> Dict:
        """
        Detect current market regime.
        
        Args:
            closes: Recent closing prices (most recent last)
            highs: Recent high prices (optional, for ATR)
            lows: Recent low prices (optional, for ATR)
            
        Returns:
            { regime, trend_strength, volatility_ratio, details }
        """
        if not closes or len(closes) < self.lookback:
            return {
                'regime': MarketRegime.UNKNOWN,
                'trend_strength': 0,
                'volatility_ratio': 0,
                'details': 'Insufficient price data',
            }
        
        recent = closes[-self.lookback:]
        
        # 1. Trend strength via linear regression R²
        r_squared = self._linear_r_squared(recent)
        
        # 2. Volatility ratio
        vol_ratio = self._volatility_ratio(recent, highs, lows)
        
        # 3. Determine regime
        if vol_ratio >= self.volatility_threshold:
            regime = MarketRegime.VOLATILE
        elif r_squared >= self.trend_threshold:
            regime = MarketRegime.TRENDING
        else:
            regime = MarketRegime.RANGING
        
        # Trend direction
        slope = recent[-1] - recent[0]
        trend_dir = "BULLISH" if slope > 0 else "BEARISH" if slope < 0 else "FLAT"
        
        return {
            'regime': regime,
            'trend_strength': round(r_squared, 4),
            'trend_direction': trend_dir,
            'volatility_ratio': round(vol_ratio, 4),
            'lookback': self.lookback,
            'current_price': recent[-1],
        }
    
    def _linear_r_squared(self, prices: List[float]) -> float:
        """
        Compute R² of linear regression on prices.
        
        High R² → strong trend. Low R² → ranging/noisy.
        """
        n = len(prices)
        if n < 2:
            return 0.0
        
        x_mean = (n - 1) / 2.0
        y_mean = sum(prices) / n
        
        ss_xy = sum((i - x_mean) * (p - y_mean) for i, p in enumerate(prices))
        ss_xx = sum((i - x_mean) ** 2 for i in range(n))
        ss_yy = sum((p - y_mean) ** 2 for p in prices)
        
        if ss_xx == 0 or ss_yy == 0:
            return 0.0
        
        r = ss_xy / math.sqrt(ss_xx * ss_yy)
        return r ** 2
    
    def _volatility_ratio(
        self,
        closes: List[float],
        highs: Optional[List[float]] = None,
        lows: Optional[List[float]] = None
    ) -> float:
        """
        Compute current volatility relative to average.
        
        If highs/lows provided, uses true range. Otherwise uses close-to-close.
        """
        n = len(closes)
        if n < 3:
            return 1.0
        
        if highs and lows and len(highs) >= n and len(lows) >= n:
            # True Range
            recent_highs = highs[-n:]
            recent_lows = lows[-n:]
            ranges = []
            for i in range(1, n):
                tr = max(
                    recent_highs[i] - recent_lows[i],
                    abs(recent_highs[i] - closes[i - 1]),
                    abs(recent_lows[i] - closes[i - 1]),
                )
                ranges.append(tr)
        else:
            # Close-to-close returns
            ranges = [abs(closes[i] - closes[i - 1]) for i in range(1, n)]
        
        if not ranges:
            return 1.0
        
        avg_range = sum(ranges) / len(ranges)
        recent_range = ranges[-1] if ranges else 0
        
        if avg_range == 0:
            return 1.0
        
        return recent_range / avg_range

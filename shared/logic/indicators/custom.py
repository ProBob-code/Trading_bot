"""
Custom Technical Indicators
===========================

Custom and composite indicators not found in standard TA libraries.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from loguru import logger


class CustomIndicators:
    """
    Custom and composite technical indicators.
    
    Includes:
    - Supertrend
    - Squeeze Momentum
    - Hull Moving Average
    - ZigZag
    - Market Structure
    """
    
    @staticmethod
    def add_supertrend(
        df: pd.DataFrame,
        period: int = 10,
        multiplier: float = 3.0
    ) -> pd.DataFrame:
        """
        Add Supertrend indicator.
        
        A trend-following indicator based on ATR.
        
        Args:
            df: OHLCV DataFrame
            period: ATR period
            multiplier: ATR multiplier
            
        Adds columns: supertrend, supertrend_direction
        """
        df = df.copy()
        
        # Calculate ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        # Calculate basic upper and lower bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = 1
        
        for i in range(1, len(df)):
            # Previous values
            prev_st = supertrend.iloc[i-1]
            prev_close = df['close'].iloc[i-1]
            curr_close = df['close'].iloc[i]
            
            # Calculate current bands
            curr_upper = upper_band.iloc[i]
            curr_lower = lower_band.iloc[i]
            
            # Adjust bands based on previous values
            if prev_st == upper_band.iloc[i-1]:
                curr_upper = min(curr_upper, prev_st) if curr_close > prev_st else curr_upper
            if prev_st == lower_band.iloc[i-1]:
                curr_lower = max(curr_lower, prev_st) if curr_close < prev_st else curr_lower
            
            # Determine trend direction
            if curr_close > prev_st:
                supertrend.iloc[i] = curr_lower
                direction.iloc[i] = 1  # Bullish
            else:
                supertrend.iloc[i] = curr_upper
                direction.iloc[i] = -1  # Bearish
                
        df['supertrend'] = supertrend
        df['supertrend_dir'] = direction
        
        return df
    
    @staticmethod
    def add_hull_ma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add Hull Moving Average (HMA).
        
        Faster and smoother than traditional moving averages.
        
        Args:
            df: OHLCV DataFrame
            period: HMA period
            
        Adds column: hma_{period}
        """
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        wma_half = df['close'].ewm(span=half_period, adjust=False).mean()
        wma_full = df['close'].ewm(span=period, adjust=False).mean()
        
        raw_hma = 2 * wma_half - wma_full
        df[f'hma_{period}'] = raw_hma.ewm(span=sqrt_period, adjust=False).mean()
        
        return df
    
    @staticmethod
    def add_squeeze_momentum(
        df: pd.DataFrame,
        bb_period: int = 20,
        bb_mult: float = 2.0,
        kc_period: int = 20,
        kc_mult: float = 1.5
    ) -> pd.DataFrame:
        """
        Add Squeeze Momentum Indicator.
        
        Identifies periods of consolidation (squeeze) and breakouts.
        
        Args:
            df: OHLCV DataFrame
            bb_period: Bollinger Bands period
            bb_mult: Bollinger Bands std multiplier
            kc_period: Keltner Channel period
            kc_mult: Keltner Channel ATR multiplier
            
        Adds columns: squeeze, squeeze_momentum
        """
        df = df.copy()
        
        # Bollinger Bands
        bb_mid = df['close'].rolling(window=bb_period).mean()
        bb_std = df['close'].rolling(window=bb_period).std()
        bb_upper = bb_mid + (bb_mult * bb_std)
        bb_lower = bb_mid - (bb_mult * bb_std)
        
        # Keltner Channel
        kc_mid = df['close'].ewm(span=kc_period, adjust=False).mean()
        
        # ATR for Keltner
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr = tr.rolling(window=kc_period).mean()
        
        kc_upper = kc_mid + (kc_mult * atr)
        kc_lower = kc_mid - (kc_mult * atr)
        
        # Squeeze: BB inside KC
        df['squeeze'] = (bb_lower > kc_lower) & (bb_upper < kc_upper)
        
        # Momentum (Linear Regression)
        highest = df['high'].rolling(window=kc_period).max()
        lowest = df['low'].rolling(window=kc_period).min()
        avg = (highest + lowest) / 2 + kc_mid
        avg = avg / 2
        
        df['squeeze_momentum'] = df['close'] - avg
        
        return df
    
    @staticmethod
    def add_zigzag(
        df: pd.DataFrame,
        threshold_pct: float = 5.0
    ) -> pd.DataFrame:
        """
        Add ZigZag indicator for identifying swing highs and lows.
        
        Args:
            df: OHLCV DataFrame
            threshold_pct: Minimum percentage move for a swing
            
        Adds columns: zigzag, swing_high, swing_low
        """
        df = df.copy()
        
        zigzag = pd.Series(index=df.index, dtype=float)
        swing_high = pd.Series(index=df.index, dtype=bool)
        swing_low = pd.Series(index=df.index, dtype=bool)
        
        swing_high[:] = False
        swing_low[:] = False
        
        if len(df) < 2:
            df['zigzag'] = zigzag
            return df
        
        # Find initial direction
        last_pivot = df['close'].iloc[0]
        last_pivot_type = 0  # 0=none, 1=high, -1=low
        threshold = threshold_pct / 100
        
        for i in range(1, len(df)):
            curr_price = df['close'].iloc[i]
            
            if last_pivot_type == 0:
                # Initialize
                if curr_price > last_pivot * (1 + threshold):
                    last_pivot = curr_price
                    last_pivot_type = 1
                elif curr_price < last_pivot * (1 - threshold):
                    last_pivot = curr_price
                    last_pivot_type = -1
            elif last_pivot_type == 1:  # Looking for swing high
                if curr_price > last_pivot:
                    last_pivot = curr_price
                elif curr_price < last_pivot * (1 - threshold):
                    # Swing high confirmed
                    swing_high.iloc[i-1] = True
                    zigzag.iloc[i-1] = last_pivot
                    last_pivot = curr_price
                    last_pivot_type = -1
            else:  # Looking for swing low
                if curr_price < last_pivot:
                    last_pivot = curr_price
                elif curr_price > last_pivot * (1 + threshold):
                    # Swing low confirmed
                    swing_low.iloc[i-1] = True
                    zigzag.iloc[i-1] = last_pivot
                    last_pivot = curr_price
                    last_pivot_type = 1
        
        df['zigzag'] = zigzag
        df['swing_high'] = swing_high
        df['swing_low'] = swing_low
        
        return df
    
    @staticmethod
    def add_market_structure(
        df: pd.DataFrame,
        lookback: int = 10
    ) -> pd.DataFrame:
        """
        Add Market Structure analysis (Higher Highs, Lower Lows).
        
        Args:
            df: OHLCV DataFrame
            lookback: Number of periods to analyze
            
        Adds columns: hh, ll, hl, lh, structure_trend
        """
        df = df.copy()
        
        # Find local maxima and minima
        df['local_max'] = (
            (df['high'] > df['high'].shift(1)) & 
            (df['high'] > df['high'].shift(-1))
        )
        df['local_min'] = (
            (df['low'] < df['low'].shift(1)) & 
            (df['low'] < df['low'].shift(-1))
        )
        
        # Initialize structure columns
        df['hh'] = False  # Higher High
        df['ll'] = False  # Lower Low
        df['hl'] = False  # Higher Low
        df['lh'] = False  # Lower High
        
        # Find structure
        prev_high = None
        prev_low = None
        
        for i in range(lookback, len(df)):
            if df['local_max'].iloc[i]:
                curr_high = df['high'].iloc[i]
                if prev_high is not None:
                    if curr_high > prev_high:
                        df.loc[df.index[i], 'hh'] = True
                    else:
                        df.loc[df.index[i], 'lh'] = True
                prev_high = curr_high
                
            if df['local_min'].iloc[i]:
                curr_low = df['low'].iloc[i]
                if prev_low is not None:
                    if curr_low > prev_low:
                        df.loc[df.index[i], 'hl'] = True
                    else:
                        df.loc[df.index[i], 'll'] = True
                prev_low = curr_low
        
        # Determine overall trend
        # Uptrend: HH + HL, Downtrend: LL + LH
        df['structure_trend'] = 0
        df.loc[df['hh'] | df['hl'], 'structure_trend'] = 1
        df.loc[df['ll'] | df['lh'], 'structure_trend'] = -1
        df['structure_trend'] = df['structure_trend'].rolling(window=lookback).mean()
        
        return df
    
    @staticmethod
    def add_trend_strength(
        df: pd.DataFrame,
        ema_period: int = 21
    ) -> pd.DataFrame:
        """
        Add composite trend strength indicator.
        
        Combines multiple trend indicators into a single score.
        
        Adds column: trend_strength (-100 to +100)
        """
        df = df.copy()
        
        # Calculate components
        ema = df['close'].ewm(span=ema_period, adjust=False).mean()
        
        # Price vs EMA
        price_ema = ((df['close'] - ema) / ema * 100).clip(-20, 20) * 2.5
        
        # EMA slope
        ema_slope = ((ema - ema.shift(5)) / ema.shift(5) * 100).clip(-10, 10) * 5
        
        # Momentum (rate of change)
        roc = ((df['close'] - df['close'].shift(10)) / df['close'].shift(10) * 100).clip(-20, 20) * 2.5
        
        # Combine
        df['trend_strength'] = (price_ema + ema_slope + roc).clip(-100, 100)
        
        return df
    
    @staticmethod
    def add_support_resistance(
        df: pd.DataFrame,
        lookback: int = 50,
        tolerance_pct: float = 0.5
    ) -> pd.DataFrame:
        """
        Add Support and Resistance levels.
        
        Args:
            df: OHLCV DataFrame
            lookback: Period to analyze
            tolerance_pct: Tolerance for level clustering
            
        Adds columns: support_1, support_2, resistance_1, resistance_2
        """
        df = df.copy()
        
        def find_levels(prices, is_support=True):
            """Find price levels where price has historically bounced."""
            levels = []
            tolerance = tolerance_pct / 100
            
            for i in range(5, len(prices) - 5):
                if is_support:
                    if prices[i] < prices[i-1] and prices[i] < prices[i+1]:
                        if prices[i] < prices[i-2] and prices[i] < prices[i+2]:
                            levels.append(prices[i])
                else:
                    if prices[i] > prices[i-1] and prices[i] > prices[i+1]:
                        if prices[i] > prices[i-2] and prices[i] > prices[i+2]:
                            levels.append(prices[i])
            
            # Cluster nearby levels
            if not levels:
                return []
            
            levels = sorted(levels)
            clustered = [levels[0]]
            for level in levels[1:]:
                if abs(level - clustered[-1]) / clustered[-1] > tolerance:
                    clustered.append(level)
                else:
                    clustered[-1] = (clustered[-1] + level) / 2
            
            return sorted(clustered)[:3]  # Return top 3 levels
        
        # Initialize columns
        df['support_1'] = np.nan
        df['support_2'] = np.nan
        df['resistance_1'] = np.nan
        df['resistance_2'] = np.nan
        
        # Calculate rolling S/R
        for i in range(lookback, len(df)):
            window = df.iloc[i-lookback:i]
            
            supports = find_levels(window['low'].values, is_support=True)
            resistances = find_levels(window['high'].values, is_support=False)
            
            curr_price = df['close'].iloc[i]
            
            # Get nearest support levels below current price
            valid_supports = [s for s in supports if s < curr_price]
            if len(valid_supports) >= 1:
                df.loc[df.index[i], 'support_1'] = valid_supports[-1]
            if len(valid_supports) >= 2:
                df.loc[df.index[i], 'support_2'] = valid_supports[-2]
            
            # Get nearest resistance levels above current price
            valid_resistances = [r for r in resistances if r > curr_price]
            if len(valid_resistances) >= 1:
                df.loc[df.index[i], 'resistance_1'] = valid_resistances[0]
            if len(valid_resistances) >= 2:
                df.loc[df.index[i], 'resistance_2'] = valid_resistances[1]
        
        return df
    
    @staticmethod
    def add_all_custom(df: pd.DataFrame) -> pd.DataFrame:
        """Add all custom indicators."""
        df = CustomIndicators.add_supertrend(df)
        df = CustomIndicators.add_hull_ma(df, 20)
        df = CustomIndicators.add_squeeze_momentum(df)
        df = CustomIndicators.add_trend_strength(df)
        return df

"""
Technical Analysis Indicators
=============================

Comprehensive technical indicator library for trading strategies.
All functions operate on pandas DataFrames with OHLCV data.
"""

from typing import Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger

try:
    from ta.trend import (
        SMAIndicator, EMAIndicator, MACD, ADXIndicator,
        IchimokuIndicator, AroonIndicator
    )
    from ta.momentum import RSIIndicator, StochasticOscillator, WilliamsRIndicator
    from ta.volatility import BollingerBands, AverageTrueRange, KeltnerChannel
    from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False
    logger.warning("ta library not available. Some indicators may not work.")


class TechnicalIndicators:
    """
    Collection of technical analysis indicators.
    
    All methods are static and operate on OHLCV DataFrames.
    Returns the input DataFrame with new indicator columns added.
    
    Usage:
        df = TechnicalIndicators.add_all_indicators(df)
        # or add specific indicators
        df = TechnicalIndicators.add_moving_averages(df, [9, 21, 50, 200])
    """
    
    # =========================================================================
    # TREND INDICATORS
    # =========================================================================
    
    @staticmethod
    def add_sma(df: pd.DataFrame, periods: list = [20, 50, 200]) -> pd.DataFrame:
        """
        Add Simple Moving Averages.
        
        Args:
            df: OHLCV DataFrame
            periods: List of periods for SMAs
            
        Returns:
            DataFrame with SMA columns added
        """
        for period in periods:
            df[f'sma_{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    @staticmethod
    def add_ema(df: pd.DataFrame, periods: list = [9, 21, 50]) -> pd.DataFrame:
        """
        Add Exponential Moving Averages.
        
        Args:
            df: OHLCV DataFrame
            periods: List of periods for EMAs
            
        Returns:
            DataFrame with EMA columns added
        """
        for period in periods:
            df[f'ema_{period}'] = df['close'].ewm(span=period, adjust=False).mean()
        return df
    
    @staticmethod
    def add_moving_averages(df: pd.DataFrame, periods: list = [9, 21, 44, 50, 200]) -> pd.DataFrame:
        """Add both SMA and EMA for given periods."""
        df = TechnicalIndicators.add_sma(df, periods)
        df = TechnicalIndicators.add_ema(df, periods)
        return df
    
    @staticmethod
    def add_macd(
        df: pd.DataFrame,
        fast: int = 12,
        slow: int = 26,
        signal: int = 9
    ) -> pd.DataFrame:
        """
        Add MACD (Moving Average Convergence Divergence).
        
        Adds columns: macd, macd_signal, macd_histogram
        """
        if TA_AVAILABLE:
            macd = MACD(df['close'], window_slow=slow, window_fast=fast, window_sign=signal)
            df['macd'] = macd.macd()
            df['macd_signal'] = macd.macd_signal()
            df['macd_histogram'] = macd.macd_diff()
        else:
            # Manual calculation
            ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
            ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
            df['macd'] = ema_fast - ema_slow
            df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
            df['macd_histogram'] = df['macd'] - df['macd_signal']
        return df
    
    @staticmethod
    def add_ichimoku(
        df: pd.DataFrame,
        conversion: int = 9,
        base: int = 26,
        span_b: int = 52
    ) -> pd.DataFrame:
        """
        Add Ichimoku Cloud indicators.
        
        Adds columns: ichimoku_conv, ichimoku_base, ichimoku_a, ichimoku_b
        """
        if TA_AVAILABLE:
            ich = IchimokuIndicator(
                high=df['high'],
                low=df['low'],
                window1=conversion,
                window2=base,
                window3=span_b
            )
            df['ichimoku_conv'] = ich.ichimoku_conversion_line()
            df['ichimoku_base'] = ich.ichimoku_base_line()
            df['ichimoku_a'] = ich.ichimoku_a()
            df['ichimoku_b'] = ich.ichimoku_b()
        else:
            # Manual calculation
            high_conv = df['high'].rolling(window=conversion).max()
            low_conv = df['low'].rolling(window=conversion).min()
            df['ichimoku_conv'] = (high_conv + low_conv) / 2
            
            high_base = df['high'].rolling(window=base).max()
            low_base = df['low'].rolling(window=base).min()
            df['ichimoku_base'] = (high_base + low_base) / 2
            
            df['ichimoku_a'] = (df['ichimoku_conv'] + df['ichimoku_base']) / 2
            
            high_b = df['high'].rolling(window=span_b).max()
            low_b = df['low'].rolling(window=span_b).min()
            df['ichimoku_b'] = (high_b + low_b) / 2
            
        return df
    
    @staticmethod
    def add_adx(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average Directional Index (trend strength).
        
        Adds columns: adx, adx_pos, adx_neg
        """
        if TA_AVAILABLE:
            adx = ADXIndicator(df['high'], df['low'], df['close'], window=period)
            df['adx'] = adx.adx()
            df['adx_pos'] = adx.adx_pos()
            df['adx_neg'] = adx.adx_neg()
        else:
            # Simplified calculation
            df['adx'] = np.nan  # Would need full implementation
        return df
    
    # =========================================================================
    # MOMENTUM INDICATORS
    # =========================================================================
    
    @staticmethod
    def add_rsi(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Relative Strength Index.
        
        Adds column: rsi
        """
        if TA_AVAILABLE:
            rsi = RSIIndicator(df['close'], window=period)
            df['rsi'] = rsi.rsi()
        else:
            # Manual calculation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        return df
    
    @staticmethod
    def add_stochastic(
        df: pd.DataFrame,
        k_period: int = 14,
        d_period: int = 3
    ) -> pd.DataFrame:
        """
        Add Stochastic Oscillator.
        
        Adds columns: stoch_k, stoch_d
        """
        if TA_AVAILABLE:
            stoch = StochasticOscillator(
                df['high'], df['low'], df['close'],
                window=k_period, smooth_window=d_period
            )
            df['stoch_k'] = stoch.stoch()
            df['stoch_d'] = stoch.stoch_signal()
        else:
            highest = df['high'].rolling(window=k_period).max()
            lowest = df['low'].rolling(window=k_period).min()
            df['stoch_k'] = 100 * (df['close'] - lowest) / (highest - lowest)
            df['stoch_d'] = df['stoch_k'].rolling(window=d_period).mean()
        return df
    
    @staticmethod
    def add_williams_r(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Williams %R.
        
        Adds column: williams_r
        """
        if TA_AVAILABLE:
            wr = WilliamsRIndicator(df['high'], df['low'], df['close'], lbp=period)
            df['williams_r'] = wr.williams_r()
        else:
            highest = df['high'].rolling(window=period).max()
            lowest = df['low'].rolling(window=period).min()
            df['williams_r'] = -100 * (highest - df['close']) / (highest - lowest)
        return df
    
    # =========================================================================
    # VOLATILITY INDICATORS
    # =========================================================================
    
    @staticmethod
    def add_bollinger_bands(
        df: pd.DataFrame,
        period: int = 20,
        std: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Bollinger Bands.
        
        Adds columns: bb_upper, bb_middle, bb_lower, bb_width, bb_pct
        """
        if TA_AVAILABLE:
            bb = BollingerBands(df['close'], window=period, window_dev=std)
            df['bb_upper'] = bb.bollinger_hband()
            df['bb_middle'] = bb.bollinger_mavg()
            df['bb_lower'] = bb.bollinger_lband()
            df['bb_width'] = bb.bollinger_wband()
            df['bb_pct'] = bb.bollinger_pband()
        else:
            df['bb_middle'] = df['close'].rolling(window=period).mean()
            rolling_std = df['close'].rolling(window=period).std()
            df['bb_upper'] = df['bb_middle'] + (std * rolling_std)
            df['bb_lower'] = df['bb_middle'] - (std * rolling_std)
            df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
            df['bb_pct'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        return df
    
    @staticmethod
    def add_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """
        Add Average True Range.
        
        Adds columns: atr, atr_pct
        """
        if TA_AVAILABLE:
            atr = AverageTrueRange(df['high'], df['low'], df['close'], window=period)
            df['atr'] = atr.average_true_range()
        else:
            high_low = df['high'] - df['low']
            high_close = abs(df['high'] - df['close'].shift(1))
            low_close = abs(df['low'] - df['close'].shift(1))
            tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
            df['atr'] = tr.rolling(window=period).mean()
            
        df['atr_pct'] = df['atr'] / df['close'] * 100
        return df
    
    @staticmethod
    def add_keltner_channel(
        df: pd.DataFrame,
        period: int = 20,
        atr_mult: float = 2.0
    ) -> pd.DataFrame:
        """
        Add Keltner Channel.
        
        Adds columns: kc_upper, kc_middle, kc_lower
        """
        if TA_AVAILABLE:
            kc = KeltnerChannel(
                df['high'], df['low'], df['close'],
                window=period, window_atr=period
            )
            df['kc_upper'] = kc.keltner_channel_hband()
            df['kc_middle'] = kc.keltner_channel_mband()
            df['kc_lower'] = kc.keltner_channel_lband()
        else:
            df['kc_middle'] = df['close'].ewm(span=period, adjust=False).mean()
            # Need ATR first
            if 'atr' not in df.columns:
                df = TechnicalIndicators.add_atr(df, period)
            df['kc_upper'] = df['kc_middle'] + (atr_mult * df['atr'])
            df['kc_lower'] = df['kc_middle'] - (atr_mult * df['atr'])
        return df
    
    # =========================================================================
    # VOLUME INDICATORS
    # =========================================================================
    
    @staticmethod
    def add_obv(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add On-Balance Volume.
        
        Adds column: obv
        """
        if TA_AVAILABLE:
            obv = OnBalanceVolumeIndicator(df['close'], df['volume'])
            df['obv'] = obv.on_balance_volume()
        else:
            obv = [0]
            for i in range(1, len(df)):
                if df['close'].iloc[i] > df['close'].iloc[i-1]:
                    obv.append(obv[-1] + df['volume'].iloc[i])
                elif df['close'].iloc[i] < df['close'].iloc[i-1]:
                    obv.append(obv[-1] - df['volume'].iloc[i])
                else:
                    obv.append(obv[-1])
            df['obv'] = obv
        return df
    
    @staticmethod
    def add_vwap(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Volume Weighted Average Price.
        
        Adds column: vwap
        """
        if TA_AVAILABLE:
            try:
                vwap = VolumeWeightedAveragePrice(
                    df['high'], df['low'], df['close'], df['volume']
                )
                df['vwap'] = vwap.volume_weighted_average_price()
            except:
                # Calculate manually if ta fails
                typical_price = (df['high'] + df['low'] + df['close']) / 3
                df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        else:
            typical_price = (df['high'] + df['low'] + df['close']) / 3
            df['vwap'] = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        return df
    
    @staticmethod
    def add_volume_sma(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Add Volume Simple Moving Average."""
        df[f'volume_sma_{period}'] = df['volume'].rolling(window=period).mean()
        return df
    
    # =========================================================================
    # PRICE PATTERNS
    # =========================================================================
    
    @staticmethod
    def add_price_channels(
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """
        Add Donchian Channels (Price Channels).
        
        Adds columns: dc_upper, dc_lower, dc_middle
        """
        df['dc_upper'] = df['high'].rolling(window=period).max()
        df['dc_lower'] = df['low'].rolling(window=period).min()
        df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2
        return df
    
    @staticmethod
    def add_pivot_points(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add Classic Pivot Points.
        
        Adds columns: pivot, r1, r2, r3, s1, s2, s3
        """
        # Use previous period's high, low, close
        prev_high = df['high'].shift(1)
        prev_low = df['low'].shift(1)
        prev_close = df['close'].shift(1)
        
        df['pivot'] = (prev_high + prev_low + prev_close) / 3
        df['r1'] = 2 * df['pivot'] - prev_low
        df['s1'] = 2 * df['pivot'] - prev_high
        df['r2'] = df['pivot'] + (prev_high - prev_low)
        df['s2'] = df['pivot'] - (prev_high - prev_low)
        df['r3'] = prev_high + 2 * (df['pivot'] - prev_low)
        df['s3'] = prev_low - 2 * (prev_high - df['pivot'])
        return df
    
    # =========================================================================
    # RETURNS AND VOLATILITY
    # =========================================================================
    
    @staticmethod
    def add_returns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Add various return calculations.
        
        Adds columns: returns, log_returns, returns_5, returns_20
        """
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        df['returns_5'] = df['close'].pct_change(periods=5)
        df['returns_20'] = df['close'].pct_change(periods=20)
        return df
    
    @staticmethod
    def add_volatility(df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """
        Add historical volatility.
        
        Adds columns: volatility, volatility_pct
        """
        if 'returns' not in df.columns:
            df = TechnicalIndicators.add_returns(df)
        
        df['volatility'] = df['returns'].rolling(window=period).std()
        df['volatility_pct'] = df['volatility'] * np.sqrt(252) * 100  # Annualized
        return df
    
    # =========================================================================
    # HIGH-LEVEL FUNCTIONS
    # =========================================================================
    
    @staticmethod
    def add_all_indicators(
        df: pd.DataFrame,
        include_volume: bool = True
    ) -> pd.DataFrame:
        """
        Add all standard technical indicators.
        
        This is a convenience function that adds:
        - Moving averages (SMA, EMA)
        - MACD
        - RSI
        - Bollinger Bands
        - ATR
        - Ichimoku Cloud
        - Stochastic
        - Volume indicators (if include_volume=True)
        
        Args:
            df: OHLCV DataFrame
            include_volume: Whether to add volume-based indicators
            
        Returns:
            DataFrame with all indicators added
        """
        df = df.copy()
        
        # Trend
        df = TechnicalIndicators.add_moving_averages(df, [9, 21, 44, 50, 200])
        df = TechnicalIndicators.add_macd(df)
        df = TechnicalIndicators.add_ichimoku(df)
        df = TechnicalIndicators.add_adx(df)
        
        # Momentum
        df = TechnicalIndicators.add_rsi(df)
        df = TechnicalIndicators.add_stochastic(df)
        
        # Volatility
        df = TechnicalIndicators.add_bollinger_bands(df)
        df = TechnicalIndicators.add_atr(df)
        
        # Returns
        df = TechnicalIndicators.add_returns(df)
        df = TechnicalIndicators.add_volatility(df)
        
        # Volume
        if include_volume:
            df = TechnicalIndicators.add_obv(df)
            df = TechnicalIndicators.add_vwap(df)
            df = TechnicalIndicators.add_volume_sma(df, 20)
        
        logger.info(f"Added {len(df.columns) - 5} technical indicators")
        return df
    
    @staticmethod
    def get_indicator_list() -> list:
        """Return list of all available indicator names."""
        return [
            'sma', 'ema', 'macd', 'ichimoku', 'adx',
            'rsi', 'stochastic', 'williams_r',
            'bollinger_bands', 'atr', 'keltner_channel',
            'obv', 'vwap', 'volume_sma',
            'price_channels', 'pivot_points',
            'returns', 'volatility'
        ]

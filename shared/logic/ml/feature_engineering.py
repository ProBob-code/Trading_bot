"""
Feature Engineering for ML Models
=================================

Creates features from OHLCV data for time series forecasting.
"""

from typing import List, Optional, Tuple
import pandas as pd
import numpy as np
from loguru import logger


class FeatureEngineer:
    """
    Feature engineering for machine learning models.
    
    Creates features from OHLCV data including:
    - Price-based features (returns, momentum)
    - Technical indicator features
    - Lag features
    - Calendar features
    - Statistical features
    """
    
    def __init__(self, lookback_periods: List[int] = None):
        """
        Initialize feature engineer.
        
        Args:
            lookback_periods: Periods for lag features (default: [1, 2, 3, 5, 10, 20])
        """
        self.lookback_periods = lookback_periods or [1, 2, 3, 5, 10, 20]
        self.feature_names: List[str] = []
        
    def create_features(
        self,
        df: pd.DataFrame,
        include_ta: bool = True,
        include_lags: bool = True,
        include_calendar: bool = True,
        target_column: str = 'close'
    ) -> pd.DataFrame:
        """
        Create all features from OHLCV data.
        
        Args:
            df: OHLCV DataFrame with datetime index
            include_ta: Include technical indicator features
            include_lags: Include lag features
            include_calendar: Include calendar features
            target_column: Column to use for target-related features
            
        Returns:
            DataFrame with all features
        """
        df = df.copy()
        
        # Price features
        df = self._add_price_features(df)
        
        # Returns features
        df = self._add_return_features(df)
        
        # Technical features (if indicators already exist)
        if include_ta:
            df = self._add_ta_features(df)
        
        # Lag features
        if include_lags:
            df = self._add_lag_features(df, target_column)
        
        # Calendar features
        if include_calendar:
            df = self._add_calendar_features(df)
        
        # Statistical features
        df = self._add_statistical_features(df)
        
        # Store feature names
        self.feature_names = [col for col in df.columns 
                            if col not in ['open', 'high', 'low', 'close', 'volume']]
        
        logger.info(f"Created {len(self.feature_names)} features")
        return df
    
    def _add_price_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features."""
        # OHLC relationships
        df['hl_pct'] = (df['high'] - df['low']) / df['close'] * 100
        df['oc_pct'] = (df['close'] - df['open']) / df['open'] * 100
        df['body_pct'] = abs(df['close'] - df['open']) / (df['high'] - df['low'] + 0.0001) * 100
        
        # Price position
        df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low'] + 0.0001)
        
        # Gap
        df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1) * 100
        
        # Distance from highs/lows
        for period in [5, 10, 20]:
            df[f'dist_high_{period}'] = (df['close'] - df['high'].rolling(period).max()) / df['close'] * 100
            df[f'dist_low_{period}'] = (df['close'] - df['low'].rolling(period).min()) / df['close'] * 100
        
        return df
    
    def _add_return_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add return-based features."""
        # Simple returns
        for period in self.lookback_periods:
            df[f'return_{period}'] = df['close'].pct_change(period) * 100
        
        # Log returns
        df['log_return_1'] = np.log(df['close'] / df['close'].shift(1)) * 100
        
        # Cumulative returns
        df['cum_return_5'] = df['close'].pct_change(5).rolling(5).sum() * 100
        df['cum_return_20'] = df['close'].pct_change(20).rolling(20).sum() * 100
        
        # Momentum
        df['momentum_5'] = df['close'] / df['close'].shift(5) - 1
        df['momentum_10'] = df['close'] / df['close'].shift(10) - 1
        df['momentum_20'] = df['close'] / df['close'].shift(20) - 1
        
        # Acceleration
        df['acceleration'] = df['momentum_5'] - df['momentum_5'].shift(5)
        
        return df
    
    def _add_ta_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add technical indicator-based features."""
        # RSI features (if RSI exists)
        if 'rsi' in df.columns:
            df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
            df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
            df['rsi_direction'] = np.sign(df['rsi'] - df['rsi'].shift(1))
        
        # MACD features
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            df['macd_cross'] = np.sign(df['macd'] - df['macd_signal'])
            df['macd_cross_change'] = df['macd_cross'].diff()
        
        # Bollinger Band features
        if 'bb_pct' in df.columns:
            df['bb_squeeze'] = (df['bb_pct'] > 0.8).astype(int) - (df['bb_pct'] < 0.2).astype(int)
        
        # Moving average features
        for ma_col in ['sma_20', 'sma_50', 'ema_21']:
            if ma_col in df.columns:
                df[f'{ma_col}_dist'] = (df['close'] - df[ma_col]) / df[ma_col] * 100
                df[f'{ma_col}_slope'] = df[ma_col].pct_change(5) * 100
        
        # MA crossovers
        if 'sma_20' in df.columns and 'sma_50' in df.columns:
            df['ma_cross_20_50'] = (df['sma_20'] > df['sma_50']).astype(int)
        
        # Ichimoku features
        if 'ichimoku_a' in df.columns and 'ichimoku_b' in df.columns:
            df['above_cloud'] = ((df['close'] > df['ichimoku_a']) & 
                                (df['close'] > df['ichimoku_b'])).astype(int)
            df['below_cloud'] = ((df['close'] < df['ichimoku_a']) & 
                                (df['close'] < df['ichimoku_b'])).astype(int)
        
        # ATR features
        if 'atr' in df.columns:
            df['atr_normalized'] = df['atr'] / df['close'] * 100
            df['atr_change'] = df['atr'].pct_change(5)
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
            df['volume_trend'] = df['volume'].rolling(5).mean() / df['volume'].rolling(20).mean()
        
        return df
    
    def _add_lag_features(self, df: pd.DataFrame, target_column: str = 'close') -> pd.DataFrame:
        """Add lagged features for time series prediction."""
        # Lagged returns
        for lag in self.lookback_periods:
            df[f'lag_return_{lag}'] = df[target_column].pct_change().shift(lag)
        
        # Lagged close prices (normalized)
        df['lag_close_1'] = df['close'].shift(1) / df['close'].shift(2) - 1
        df['lag_close_5'] = df['close'].shift(5) / df['close'].shift(6) - 1
        
        # Lagged volume
        df['lag_volume_1'] = df['volume'].shift(1) / df['volume'].rolling(20).mean()
        
        # Lagged volatility
        df['lag_volatility'] = df['close'].pct_change().rolling(10).std().shift(1)
        
        return df
    
    def _add_calendar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calendar-based features."""
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("Index is not DatetimeIndex, skipping calendar features")
            return df
        
        # Day of week (0=Monday, 4=Friday)
        df['day_of_week'] = df.index.dayofweek
        df['is_monday'] = (df.index.dayofweek == 0).astype(int)
        df['is_friday'] = (df.index.dayofweek == 4).astype(int)
        
        # Month features
        df['month'] = df.index.month
        df['is_month_start'] = df.index.is_month_start.astype(int)
        df['is_month_end'] = df.index.is_month_end.astype(int)
        
        # Quarter
        df['quarter'] = df.index.quarter
        
        # Hour (for intraday)
        if hasattr(df.index, 'hour'):
            df['hour'] = df.index.hour
            df['is_market_open'] = ((df.index.hour >= 9) & (df.index.hour < 10)).astype(int)
            df['is_market_close'] = ((df.index.hour >= 15) & (df.index.hour < 16)).astype(int)
        
        return df
    
    def _add_statistical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add statistical features."""
        returns = df['close'].pct_change()
        
        # Rolling statistics
        for window in [5, 10, 20]:
            df[f'volatility_{window}'] = returns.rolling(window).std() * np.sqrt(252)
            df[f'skewness_{window}'] = returns.rolling(window).skew()
            df[f'kurtosis_{window}'] = returns.rolling(window).kurt()
        
        # Z-score
        df['price_zscore'] = (df['close'] - df['close'].rolling(20).mean()) / df['close'].rolling(20).std()
        
        # Relative position
        df['percentile_20'] = df['close'].rolling(20).apply(
            lambda x: (x.iloc[-1] - x.min()) / (x.max() - x.min() + 0.0001)
        )
        
        return df
    
    def create_target(
        self,
        df: pd.DataFrame,
        horizon: int = 1,
        target_type: str = 'direction',
        threshold: float = 0.0
    ) -> pd.DataFrame:
        """
        Create target variable for supervised learning.
        
        Args:
            df: DataFrame with features
            horizon: Prediction horizon (periods ahead)
            target_type: 'direction' (up/down), 'return', or 'multi_class'
            threshold: Threshold for direction (to create neutral zone)
            
        Returns:
            DataFrame with target column added
        """
        df = df.copy()
        
        # Future return
        future_return = df['close'].shift(-horizon) / df['close'] - 1
        
        if target_type == 'return':
            df['target'] = future_return
        elif target_type == 'direction':
            if threshold > 0:
                # Multi-class: -1 (down), 0 (neutral), 1 (up)
                df['target'] = np.where(
                    future_return > threshold, 1,
                    np.where(future_return < -threshold, -1, 0)
                )
            else:
                # Binary: 0 (down), 1 (up)
                df['target'] = (future_return > 0).astype(int)
        elif target_type == 'multi_class':
            # 5 classes: strong down, down, neutral, up, strong up
            df['target'] = pd.cut(
                future_return,
                bins=[-np.inf, -0.02, -0.005, 0.005, 0.02, np.inf],
                labels=[0, 1, 2, 3, 4]
            ).astype(int)
        
        return df
    
    def prepare_data(
        self,
        df: pd.DataFrame,
        target_column: str = 'target',
        train_size: float = 0.8,
        drop_na: bool = True
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Prepare data for training.
        
        Args:
            df: DataFrame with features and target
            target_column: Name of target column
            train_size: Fraction for training data
            drop_na: Whether to drop NaN rows
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        df = df.copy()
        
        if drop_na:
            df = df.dropna()
        
        # Split features and target
        feature_cols = [col for col in df.columns 
                       if col not in ['open', 'high', 'low', 'close', 'volume', target_column]]
        
        X = df[feature_cols]
        y = df[target_column]
        
        # Time-based split (no shuffling for time series)
        split_idx = int(len(df) * train_size)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        logger.info(f"Train size: {len(X_train)}, Test size: {len(X_test)}")
        logger.info(f"Features: {len(feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_names(self) -> List[str]:
        """Return list of feature names."""
        return self.feature_names

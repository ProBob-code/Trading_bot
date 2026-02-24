import pandas as pd
import numpy as np

class DataValidator:
    """
    Validates OHLCV data for integrity and consistency.
    """
    
    @staticmethod
    def validate_ohlcv(df: pd.DataFrame) -> bool:
        """
        Checks for:
        1. Required columns
        2. Missing values
        3. High >= Low, High >= Open/Close, Low <= Open/Close
        4. No negative prices or volumes
        """
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        if not all(col in df.columns for col in required_cols):
            return False
            
        if df.isnull().any().any():
            return False
            
        # Price sanity
        if not (df['high'] >= df['low']).all():
            return False
            
        if not (df['high'] >= df['open']).all() or not (df['high'] >= df['close']).all():
            return False
            
        if not (df['low'] <= df['open']).all() or not (df['low'] <= df['close']).all():
            return False
            
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            return False
            
        return True

    @staticmethod
    def check_time_gaps(df: pd.DataFrame, expected_freq: str) -> bool:
        """Checks for missing time intervals in the index."""
        full_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq=expected_freq)
        return len(df) == len(full_range)

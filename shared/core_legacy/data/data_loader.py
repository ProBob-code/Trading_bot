import pandas as pd
import numpy as np
from typing import Optional, List
from pathlib import Path

class DataLoader:
    """
    Handles loading and processing of OHLCV data.
    Ensures data integrity and prevents lookahead bias.
    """
    
    def __init__(self, data_path: Optional[str] = None):
        self.data_path = Path(data_path) if data_path else None

    def load_from_csv(self, file_path: str) -> pd.DataFrame:
        """Loads OHLCV data from a CSV file."""
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
        df.set_index('timestamp', inplace=True)
        df.sort_index(inplace=True)
        return df

    def resample_data(self, df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
        """
        Resamples OHLCV data to a different timeframe.
        Example: '1H', '4H', '1D'
        """
        resampled = df.resample(timeframe).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum'
        }).dropna()
        return resampled

    @staticmethod
    def get_prior_candle_data(df: pd.DataFrame) -> pd.DataFrame:
        """
        Shifts the dataframe to ensure all references are to previous candles.
        Prevents lookahead bias in signal generation.
        """
        return df.shift(1)

    def generate_mock_data(self, symbol: str, periods: int = 1000, freq: str = '1H') -> pd.DataFrame:
        """Generates synthetic OHLCV data for testing purpose."""
        np.random.seed(42)
        timestamps = pd.date_range(end=pd.Timestamp.now(), periods=periods, freq=freq)
        
        close_prices = 100 + np.cumsum(np.random.randn(periods))
        opens = close_prices + np.random.randn(periods) * 0.5
        highs = np.maximum(opens, close_prices) + np.random.rand(periods)
        lows = np.minimum(opens, close_prices) - np.random.rand(periods)
        volumes = np.random.randint(100, 1000, size=periods)
        
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': close_prices,
            'volume': volumes
        }, index=timestamps)
        
        df.index.name = 'timestamp'
        return df

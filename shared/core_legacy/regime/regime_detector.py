import pandas as pd
import numpy as np
from typing import Dict, Any
from config import settings

class RegimeDetector:
    """
    Classifies market state into Trend, Volatility, and Liquidity regimes.
    Outputs:
    - trend: 'trending' | 'ranging'
    - volatility: 'low' | 'normal' | 'high'
    - liquidity: 'valid' | 'invalid'
    """

    @staticmethod
    def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average Directional Index (ADX)."""
        plus_dm = df['high'].diff()
        minus_dm = df['low'].diff()
        
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0
        minus_dm = abs(minus_dm)
        
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(period).mean()
        plus_di = 100 * (plus_dm.rolling(period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(period).mean() / atr)
        dx = 100 * (abs(plus_di - minus_di) / (plus_di + minus_di))
        adx = dx.rolling(period).mean()
        return adx

    @staticmethod
    def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculates Average True Range (ATR)."""
        tr = pd.concat([
            df['high'] - df['low'],
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        ], axis=1).max(axis=1)
        return tr.rolling(period).mean()

    def detect(self, df: pd.DataFrame, idx: int) -> Dict[str, Any]:
        """
        Detects the market regime using pre-calculated values in the dataframe at index idx.
        """
        if idx < 30: 
            return {"trend": "ranging", "volatility": "normal", "liquidity": "valid"}

        # Use pre-calculated columns
        current_adx = df['adx'].iloc[idx]
        trend = "trending" if current_adx > settings.ADX_TREND_THRESHOLD else "ranging"

        # Volatility
        atr_pct = df['atr_percentile'].iloc[idx]
        if atr_pct > 0.8:
            volatility = "high"
        elif atr_pct < 0.2:
            volatility = "low"
        else:
            volatility = "normal"

        # Liquidity
        vol_pct = df['volume_percentile'].iloc[idx]
        liquidity = "valid" if vol_pct > settings.VOLUME_LIQUIDITY_PERCENTILE else "invalid"

        return {
            "trend": trend,
            "volatility": volatility,
            "liquidity": liquidity,
            "adx": round(current_adx, 2),
            "atr_pct": round(atr_pct, 2)
        }

    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Adds regime indicators to the full dataframe once."""
        df = df.copy()
        df['adx'] = self.calculate_adx(df, period=settings.ATR_PERIOD)
        atr = self.calculate_atr(df, period=settings.ATR_PERIOD)
        atr_relative = atr / df['close']
        df['atr_percentile'] = atr_relative.rolling(100).rank(pct=True)
        df['volume_percentile'] = df['volume'].rolling(100).rank(pct=True)
        return df

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger

# Import V2 dynamic strategies
from src.strategies.v2_strategies import REGISTRY, Signal

class StrategyEngine:
    """
    Multi-strategy trading engine.
    
    Supports multiple technical analysis strategies and ML forecasting.
    Now dynamically loaded from v2_strategies.REGISTRY.
    """
    
    def __init__(self, min_confluence: int = 2):
        """
        Initialize strategy engine.
        
        Args:
            min_confluence: Minimum signals needed to trigger trade (default 2)
        """
        self.min_confluence = min_confluence
        # Dynamically load strategy IDs from registry
        self.strategies = [s['id'] for s in REGISTRY]
        logger.info(f"✅ StrategyEngine initialized with {len(self.strategies)} strategies: {self.strategies}")
    
    def analyze(
        self, 
        df: pd.DataFrame, 
        strategy: str = 'ichimoku',
        settings: Optional[Dict] = None
    ) -> Signal:
        """
        Analyze price data and generate signal.
        
        Args:
            df: DataFrame with OHLCV data
            strategy: Strategy to use
            settings: Custom settings
            
        Returns:
            Signal object
        """
        if df.empty or len(df) < 50:
            return Signal('none', 'HOLD', 0.0, 0.0, ['Insufficient data'], datetime.now())
        
        current_price = float(df['close'].iloc[-1])
        
        # Dynamic lookup from Registry
        strategy_def = next((s for s in REGISTRY if s['id'] == strategy), None)
        
        if strategy_def:
            try:
                # Call strategy logic dynamically
                return strategy_def['logic'](df, current_price, self.min_confluence)
            except Exception as e:
                logger.error(f"❌ Error executing strategy {strategy}: {e}")
                return Signal(strategy, 'HOLD', 0.0, current_price, [f"Error: {str(e)}"], datetime.now())
        else:
            logger.warning(f"⚠️ Unknown strategy requested: {strategy}")
            return Signal(strategy, 'HOLD', 0.0, current_price, ['Unknown strategy'], datetime.now())
    
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        Useful for visualization on the frontend.
        """
        if df.empty:
            return df
            
        df = df.copy()
        high = df['high']
        low = df['low']
        close = df['close']
        
        # --- Ichimoku ---
        nine_high = high.rolling(9).max()
        nine_low = low.rolling(9).min()
        df['tenkan'] = (nine_high + nine_low) / 2
        
        period26_high = high.rolling(26).max()
        period26_low = low.rolling(26).min()
        df['kijun'] = (period26_high + period26_low) / 2
        
        df['span_a'] = ((df['tenkan'] + df['kijun']) / 2).shift(26)
        period52_high = high.rolling(52).max()
        period52_low = low.rolling(52).min()
        df['span_b'] = ((period52_high + period52_low) / 2).shift(26)
        
        # --- Bollinger Bands ---
        df['sma20'] = close.rolling(20).mean()
        std = close.rolling(20).std()
        df['upper_band'] = df['sma20'] + (2 * std)
        df['lower_band'] = df['sma20'] - (2 * std)
        
        # --- MACD ---
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        df['macd'] = ema12 - ema26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']
        
        # --- RSI ---
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df

# Factory function
def get_strategy_engine(min_confluence: int = 3) -> StrategyEngine:
    """Create a strategy engine instance."""
    return StrategyEngine(min_confluence=min_confluence)

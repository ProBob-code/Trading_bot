"""
Multi-Strategy Trading Engine
=============================

Multiple trading strategies with customization support:
- Ichimoku Cloud
- Bollinger Bands
- MACD + RSI
- ML Time Series Forecast
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
from loguru import logger


@dataclass
class Signal:
    """Trading signal from a strategy."""
    strategy: str
    signal: str  # 'BUY', 'SELL', 'HOLD'
    strength: int  # 1-5
    price: float
    reasons: List[str]
    timestamp: datetime


class StrategyEngine:
    """
    Multi-strategy trading engine.
    
    Supports multiple technical analysis strategies and ML forecasting.
    """
    
    def __init__(self, min_confluence: int = 2):
        """
        Initialize strategy engine.
        
        Args:
            min_confluence: Minimum signals needed to trigger trade (default 2)
        """
        self.min_confluence = min_confluence
        self.strategies = ['ichimoku', 'bollinger', 'macd_rsi', 'ml_forecast', 'combined']
    
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
        if df.empty or len(df) < 52:
            return Signal('none', 'HOLD', 0, 0, ['Insufficient data'], datetime.now())
        
        current_price = float(df['close'].iloc[-1])
        
        if strategy == 'ichimoku':
            return self._ichimoku_signal(df, current_price)
        elif strategy == 'bollinger':
            return self._bollinger_signal(df, current_price)
        elif strategy == 'macd_rsi':
            return self._macd_rsi_signal(df, current_price)
        elif strategy == 'ml_forecast':
            return self._ml_forecast_signal(df, current_price)
        elif strategy == 'combined':
            return self._combined_signal(df, current_price)
        else:
            return Signal(strategy, 'HOLD', 0, current_price, ['Unknown strategy'], datetime.now())
    
    def _ichimoku_signal(self, df: pd.DataFrame, price: float) -> Signal:
        """Ichimoku Cloud strategy."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Tenkan-sen (9-period)
        nine_high = high.rolling(9).max()
        nine_low = low.rolling(9).min()
        tenkan = (nine_high + nine_low) / 2
        
        # Kijun-sen (26-period)
        period26_high = high.rolling(26).max()
        period26_low = low.rolling(26).min()
        kijun = (period26_high + period26_low) / 2
        
        # Senkou Span A & B
        span_a = ((tenkan + kijun) / 2).shift(26)
        period52_high = high.rolling(52).max()
        period52_low = low.rolling(52).min()
        span_b = ((period52_high + period52_low) / 2).shift(26)
        
        # Get latest values
        tenkan_now = tenkan.iloc[-1]
        kijun_now = kijun.iloc[-1]
        span_a_now = span_a.iloc[-1] if not pd.isna(span_a.iloc[-1]) else price
        span_b_now = span_b.iloc[-1] if not pd.isna(span_b.iloc[-1]) else price
        
        cloud_top = max(span_a_now, span_b_now)
        cloud_bottom = min(span_a_now, span_b_now)
        
        bullish = 0
        bearish = 0
        reasons = []
        
        # 1. Kumo Breakout
        if price > cloud_top:
            bullish += 1
            reasons.append("Price above cloud")
        elif price < cloud_bottom:
            bearish += 1
            reasons.append("Price below cloud")
        
        # 2. TK Cross
        if tenkan_now > kijun_now:
            bullish += 1
            reasons.append("TK bullish cross")
        elif tenkan_now < kijun_now:
            bearish += 1
            reasons.append("TK bearish cross")
        
        # 3. Cloud color
        if span_a_now > span_b_now:
            bullish += 1
            reasons.append("Green cloud (bullish)")
        elif span_a_now < span_b_now:
            bearish += 1
            reasons.append("Red cloud (bearish)")
        
        # Generate signal
        if bullish >= self.min_confluence:
            return Signal('ichimoku', 'BUY', bullish, price, reasons, datetime.now())
        elif bearish >= self.min_confluence:
            return Signal('ichimoku', 'SELL', bearish, price, reasons, datetime.now())
        else:
            return Signal('ichimoku', 'HOLD', max(bullish, bearish), price, reasons, datetime.now())
    
    def _bollinger_signal(self, df: pd.DataFrame, price: float) -> Signal:
        """Bollinger Bands strategy."""
        close = df['close']
        
        # Calculate Bollinger Bands (20 period, 2 std)
        period = 20
        sma = close.rolling(period).mean()
        std = close.rolling(period).std()
        
        upper = sma + (2 * std)
        lower = sma - (2 * std)
        
        upper_now = upper.iloc[-1]
        lower_now = lower.iloc[-1]
        sma_now = sma.iloc[-1]
        
        # Band width (squeeze detection)
        band_width = (upper_now - lower_now) / sma_now * 100
        avg_band_width = ((upper - lower) / sma * 100).tail(50).mean()
        
        bullish = 0
        bearish = 0
        reasons = []
        
        # 1. Price position
        if price <= lower_now:
            bullish += 2  # Strong signal
            reasons.append("Price at lower band (oversold)")
        elif price >= upper_now:
            bearish += 2
            reasons.append("Price at upper band (overbought)")
        
        # 2. Squeeze (low bandwidth = potential breakout)
        if band_width < avg_band_width * 0.8:
            reasons.append("Bollinger squeeze detected")
            # Direction based on recent movement
            if close.iloc[-1] > close.iloc[-5]:
                bullish += 1
            else:
                bearish += 1
        
        # 3. Band walk
        if price > sma_now:
            bullish += 1
            reasons.append("Above middle band")
        elif price < sma_now:
            bearish += 1
            reasons.append("Below middle band")
        
        if bullish >= self.min_confluence:
            return Signal('bollinger', 'BUY', bullish, price, reasons, datetime.now())
        elif bearish >= self.min_confluence:
            return Signal('bollinger', 'SELL', bearish, price, reasons, datetime.now())
        else:
            return Signal('bollinger', 'HOLD', max(bullish, bearish), price, reasons, datetime.now())
    
    def _macd_rsi_signal(self, df: pd.DataFrame, price: float) -> Signal:
        """MACD + RSI combined strategy."""
        close = df['close']
        
        # MACD
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        histogram = macd_line - signal_line
        
        # RSI (14 period)
        delta = close.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        
        macd_now = macd_line.iloc[-1]
        signal_now = signal_line.iloc[-1]
        hist_now = histogram.iloc[-1]
        hist_prev = histogram.iloc[-2]
        rsi_now = rsi.iloc[-1]
        
        bullish = 0
        bearish = 0
        reasons = []
        
        # 1. MACD cross
        if macd_now > signal_now and macd_line.iloc[-2] <= signal_line.iloc[-2]:
            bullish += 2
            reasons.append("MACD bullish cross")
        elif macd_now < signal_now and macd_line.iloc[-2] >= signal_line.iloc[-2]:
            bearish += 2
            reasons.append("MACD bearish cross")
        
        # 2. MACD histogram momentum
        if hist_now > hist_prev and hist_now > 0:
            bullish += 1
            reasons.append("MACD momentum increasing")
        elif hist_now < hist_prev and hist_now < 0:
            bearish += 1
            reasons.append("MACD momentum decreasing")
        
        # 3. RSI zones
        if rsi_now < 30:
            bullish += 1
            reasons.append(f"RSI oversold ({rsi_now:.0f})")
        elif rsi_now > 70:
            bearish += 1
            reasons.append(f"RSI overbought ({rsi_now:.0f})")
        
        if bullish >= self.min_confluence:
            return Signal('macd_rsi', 'BUY', bullish, price, reasons, datetime.now())
        elif bearish >= self.min_confluence:
            return Signal('macd_rsi', 'SELL', bearish, price, reasons, datetime.now())
        else:
            return Signal('macd_rsi', 'HOLD', max(bullish, bearish), price, reasons, datetime.now())
    
    def _ml_forecast_signal(self, df: pd.DataFrame, price: float) -> Signal:
        """ML-based price prediction (simple trend forecast)."""
        close = df['close']
        
        # Simple linear regression for trend
        n = min(20, len(close))
        x = np.arange(n)
        y = close.tail(n).values
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Volatility
        volatility = close.tail(20).std() / close.tail(20).mean() * 100
        
        # SMA crossover
        sma_short = close.rolling(10).mean().iloc[-1]
        sma_long = close.rolling(30).mean().iloc[-1]
        
        bullish = 0
        bearish = 0
        reasons = []
        
        # 1. Trend direction
        if slope > 0:
            bullish += 1
            reasons.append(f"Upward trend (slope: {slope:.2f})")
        elif slope < 0:
            bearish += 1
            reasons.append(f"Downward trend (slope: {slope:.2f})")
        
        # 2. Trend strength
        if abs(slope) > y.mean() * 0.001:  # Significant slope
            if slope > 0:
                bullish += 1
                reasons.append("Strong upward momentum")
            else:
                bearish += 1
                reasons.append("Strong downward momentum")
        
        # 3. SMA alignment
        if sma_short > sma_long:
            bullish += 1
            reasons.append("Short SMA above long SMA")
        elif sma_short < sma_long:
            bearish += 1
            reasons.append("Short SMA below long SMA")
        
        if bullish >= self.min_confluence:
            return Signal('ml_forecast', 'BUY', bullish, price, reasons, datetime.now())
        elif bearish >= self.min_confluence:
            return Signal('ml_forecast', 'SELL', bearish, price, reasons, datetime.now())
        else:
            return Signal('ml_forecast', 'HOLD', max(bullish, bearish), price, reasons, datetime.now())
    
    def get_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all technical indicators for a DataFrame.
        
        Returns a copy of the DataFrame with added indicator columns.
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

    def _combined_signal(self, df: pd.DataFrame, price: float) -> Signal:
        """Combine all strategies for consensus."""
        signals = [
            self._ichimoku_signal(df, price),
            self._bollinger_signal(df, price),
            self._macd_rsi_signal(df, price),
            self._ml_forecast_signal(df, price)
        ]
        
        buy_votes = sum(1 for s in signals if s.signal == 'BUY')
        sell_votes = sum(1 for s in signals if s.signal == 'SELL')
        
        all_reasons = []
        for s in signals:
            all_reasons.extend([f"[{s.strategy}] {r}" for r in s.reasons])
        
        if buy_votes >= 2:
            return Signal('combined', 'BUY', buy_votes, price, all_reasons, datetime.now())
        elif sell_votes >= 2:
            return Signal('combined', 'SELL', sell_votes, price, all_reasons, datetime.now())
        else:
            return Signal('combined', 'HOLD', max(buy_votes, sell_votes), price, all_reasons, datetime.now())


# Factory function
def get_strategy_engine(min_confluence: int = 3) -> StrategyEngine:
    """Create a strategy engine instance."""
    return StrategyEngine(min_confluence=min_confluence)

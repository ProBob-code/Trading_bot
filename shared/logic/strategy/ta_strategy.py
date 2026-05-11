"""
Technical Analysis Strategy
===========================

Pure technical analysis based trading strategy.
"""

from typing import Dict, List, Optional
import pandas as pd
import numpy as np
from loguru import logger

from .base_strategy import BaseStrategy, Signal, SignalType
from ..indicators.technical import TechnicalIndicators
from ..indicators.custom import CustomIndicators


class TAStrategy(BaseStrategy):
    """
    Technical Analysis based trading strategy.
    
    Uses multiple TA indicators to generate trading signals:
    - Trend: MA crossover, Ichimoku Cloud, ADX
    - Momentum: RSI, Stochastic
    - Volatility: Bollinger Bands, ATR
    
    Signal is generated when multiple indicators align (confluence).
    """
    
    def __init__(
        self,
        name: str = "TAStrategy",
        ma_fast: int = 9,
        ma_slow: int = 21,
        ma_signal: int = 44,
        rsi_period: int = 14,
        rsi_overbought: int = 70,
        rsi_oversold: int = 30,
        bb_period: int = 20,
        bb_std: float = 2.0,
        atr_period: int = 14,
        atr_sl_multiplier: float = 2.0,
        min_confluence: int = 3,
        use_ichimoku: bool = True,
        use_supertrend: bool = True
    ):
        """
        Initialize TA Strategy.
        
        Args:
            name: Strategy name
            ma_fast: Fast MA period
            ma_slow: Slow MA period
            ma_signal: Signal MA period (44-period as in your notebook)
            rsi_period: RSI period
            rsi_overbought: RSI overbought level
            rsi_oversold: RSI oversold level
            bb_period: Bollinger Bands period
            bb_std: Bollinger Bands std deviation
            atr_period: ATR period for volatility
            atr_sl_multiplier: ATR multiplier for stop loss
            min_confluence: Minimum indicators that must agree
            use_ichimoku: Use Ichimoku Cloud
            use_supertrend: Use Supertrend indicator
        """
        params = {
            'ma_fast': ma_fast,
            'ma_slow': ma_slow,
            'ma_signal': ma_signal,
            'rsi_period': rsi_period,
            'rsi_overbought': rsi_overbought,
            'rsi_oversold': rsi_oversold,
            'bb_period': bb_period,
            'bb_std': bb_std,
            'atr_period': atr_period,
            'atr_sl_multiplier': atr_sl_multiplier,
            'min_confluence': min_confluence,
            'use_ichimoku': use_ichimoku,
            'use_supertrend': use_supertrend
        }
        super().__init__(name=name, params=params)
        
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add all required technical indicators."""
        df = df.copy()
        
        p = self.params
        
        # Moving Averages
        df = TechnicalIndicators.add_sma(df, [p['ma_slow'], p['ma_signal']])
        df = TechnicalIndicators.add_ema(df, [p['ma_fast'], p['ma_slow']])
        
        # MACD
        df = TechnicalIndicators.add_macd(df)
        
        # RSI
        df = TechnicalIndicators.add_rsi(df, p['rsi_period'])
        
        # Bollinger Bands
        df = TechnicalIndicators.add_bollinger_bands(df, p['bb_period'], p['bb_std'])
        
        # ATR
        df = TechnicalIndicators.add_atr(df, p['atr_period'])
        
        # Ichimoku
        if p['use_ichimoku']:
            df = TechnicalIndicators.add_ichimoku(df)
        
        # Supertrend
        if p['use_supertrend']:
            df = CustomIndicators.add_supertrend(df)
        
        # Stochastic
        df = TechnicalIndicators.add_stochastic(df)
        
        # Volume
        df = TechnicalIndicators.add_volume_sma(df, 20)
        
        self.is_initialized = True
        return df
    
    def generate_signals(
        self,
        df: pd.DataFrame,
        symbol: str = "UNKNOWN"
    ) -> List[Signal]:
        """
        Generate trading signals based on technical analysis.
        
        Uses confluence of multiple indicators.
        """
        if len(df) < 50:
            return []
        
        # Ensure indicators are calculated
        if not self._has_required_indicators(df):
            df = self.calculate_indicators(df)
        
        signals = []
        p = self.params
        
        # Get latest values
        row = df.iloc[-1]
        prev_row = df.iloc[-2]
        close = row['close']
        
        # ================================================================
        # BULLISH CONDITIONS
        # ================================================================
        bullish_signals = []
        bearish_signals = []
        
        # 1. Price above signal MA (MA44)
        if f"sma_{p['ma_signal']}" in df.columns:
            ma_signal = row[f"sma_{p['ma_signal']}"]
            if close > ma_signal:
                bullish_signals.append("Close > MA44")
            else:
                bearish_signals.append("Close <= MA44")
        
        # 2. EMA crossover (fast > slow)
        if f"ema_{p['ma_fast']}" in df.columns and f"ema_{p['ma_slow']}" in df.columns:
            ema_fast = row[f"ema_{p['ma_fast']}"]
            ema_slow = row[f"ema_{p['ma_slow']}"]
            if ema_fast > ema_slow:
                bullish_signals.append("EMA9 > EMA21")
            else:
                bearish_signals.append("EMA9 <= EMA21")
        
        # 3. Ichimoku Cloud
        if p['use_ichimoku'] and 'ichimoku_a' in df.columns:
            ich_a = row['ichimoku_a']
            ich_b = row['ichimoku_b']
            ich_conv = row['ichimoku_conv']
            ich_base = row['ichimoku_base']
            
            # Price above cloud
            if close > ich_a and close > ich_b:
                bullish_signals.append("Price above Ichimoku Cloud")
            elif close < ich_a and close < ich_b:
                bearish_signals.append("Price below Ichimoku Cloud")
            
            # Conversion line above base line
            if ich_conv > ich_base:
                bullish_signals.append("Ichimoku TK Cross bullish")
            else:
                bearish_signals.append("Ichimoku TK Cross bearish")
        
        # 4. RSI
        if 'rsi' in df.columns:
            rsi = row['rsi']
            if rsi < p['rsi_oversold']:
                bullish_signals.append(f"RSI oversold ({rsi:.1f})")
            elif rsi > p['rsi_overbought']:
                bearish_signals.append(f"RSI overbought ({rsi:.1f})")
            elif rsi > 50:
                bullish_signals.append("RSI > 50")
            else:
                bearish_signals.append("RSI <= 50")
        
        # 5. MACD
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd = row['macd']
            macd_sig = row['macd_signal']
            macd_hist = row.get('macd_histogram', macd - macd_sig)
            prev_macd_hist = prev_row.get('macd_histogram', prev_row['macd'] - prev_row['macd_signal'])
            
            if macd > macd_sig:
                bullish_signals.append("MACD > Signal")
            else:
                bearish_signals.append("MACD <= Signal")
            
            # MACD crossover
            if macd_hist > 0 and prev_macd_hist <= 0:
                bullish_signals.append("MACD bullish crossover")
            elif macd_hist < 0 and prev_macd_hist >= 0:
                bearish_signals.append("MACD bearish crossover")
        
        # 6. Bollinger Bands
        if 'bb_upper' in df.columns:
            bb_upper = row['bb_upper']
            bb_lower = row['bb_lower']
            bb_pct = row.get('bb_pct', (close - bb_lower) / (bb_upper - bb_lower))
            
            if close <= bb_lower * 1.01:  # Near lower band
                bullish_signals.append("Price at lower BB")
            elif close >= bb_upper * 0.99:  # Near upper band
                bearish_signals.append("Price at upper BB")
        
        # 7. Supertrend
        if p['use_supertrend'] and 'supertrend_dir' in df.columns:
            st_dir = row['supertrend_dir']
            if st_dir == 1:
                bullish_signals.append("Supertrend bullish")
            else:
                bearish_signals.append("Supertrend bearish")
        
        # 8. Volume confirmation
        if 'volume_sma_20' in df.columns:
            vol_ratio = row['volume'] / row['volume_sma_20']
            if vol_ratio > 1.5:
                # High volume - confirms direction
                pass  # Add to existing signals
        
        # ================================================================
        # GENERATE FINAL SIGNAL
        # ================================================================
        n_bullish = len(bullish_signals)
        n_bearish = len(bearish_signals)
        min_conf = p['min_confluence']
        
        # Calculate confidence
        total_indicators = n_bullish + n_bearish
        
        if n_bullish >= min_conf and n_bullish > n_bearish:
            confidence = n_bullish / max(total_indicators, 1)
            
            # Calculate stop loss and take profit
            atr = row.get('atr', close * 0.02)
            stop_loss = close - (atr * p['atr_sl_multiplier'])
            take_profit = close + (atr * p['atr_sl_multiplier'] * 2)  # 2:1 RR
            
            signal = Signal(
                signal_type=SignalType.BUY,
                symbol=symbol,
                price=close,
                confidence=min(confidence, 1.0),
                reason="; ".join(bullish_signals),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'n_bullish': n_bullish,
                    'n_bearish': n_bearish,
                    'indicators': bullish_signals
                }
            )
            signals.append(signal)
            
        elif n_bearish >= min_conf and n_bearish > n_bullish:
            confidence = n_bearish / max(total_indicators, 1)
            
            # For sell/short
            atr = row.get('atr', close * 0.02)
            stop_loss = close + (atr * p['atr_sl_multiplier'])
            take_profit = close - (atr * p['atr_sl_multiplier'] * 2)
            
            signal = Signal(
                signal_type=SignalType.SELL,
                symbol=symbol,
                price=close,
                confidence=min(confidence, 1.0),
                reason="; ".join(bearish_signals),
                stop_loss=stop_loss,
                take_profit=take_profit,
                metadata={
                    'n_bullish': n_bullish,
                    'n_bearish': n_bearish,
                    'indicators': bearish_signals
                }
            )
            signals.append(signal)
            
        else:
            # Hold - no clear direction
            signal = Signal(
                signal_type=SignalType.HOLD,
                symbol=symbol,
                price=close,
                confidence=0.5,
                reason=f"No confluence: {n_bullish} bullish, {n_bearish} bearish",
                metadata={
                    'n_bullish': n_bullish,
                    'n_bearish': n_bearish,
                    'bullish_indicators': bullish_signals,
                    'bearish_indicators': bearish_signals
                }
            )
            signals.append(signal)
        
        return signals
    
    def _has_required_indicators(self, df: pd.DataFrame) -> bool:
        """Check if required indicators are present."""
        required = [
            f"sma_{self.params['ma_signal']}",
            'rsi',
            'bb_upper',
            'atr'
        ]
        return all(col in df.columns for col in required)
    
    def get_required_indicators(self) -> List[str]:
        """Return list of required indicators."""
        return [
            'sma', 'ema', 'macd', 'rsi', 
            'bollinger_bands', 'atr', 'ichimoku', 
            'supertrend', 'stochastic'
        ]

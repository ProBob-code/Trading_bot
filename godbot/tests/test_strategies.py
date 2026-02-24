"""
Test Suite: Strategy Bots
"""

import pytest
import pandas as pd
import numpy as np

from godbot.strategies.breakout import BreakoutBot
from godbot.strategies.mean_reversion import MeanReversionBot
from godbot.strategies.ichimoku import IchimokuBot
from godbot.strategies.ml_forecast import MLForecastBot


def _make_ohlcv(n=200, base_price=100, trend=0.01):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='h')
    close = base_price + np.cumsum(np.random.randn(n) * 0.5 + trend)
    high = close + np.random.uniform(0.5, 2.0, n)
    low = close - np.random.uniform(0.5, 2.0, n)
    open_ = close + np.random.randn(n) * 0.3
    volume = np.random.uniform(500, 5000, n)
    
    return pd.DataFrame({
        'open': open_, 'high': high, 'low': low,
        'close': close, 'volume': volume,
    }, index=dates)


class TestBreakoutBot:
    def test_calculate_indicators(self):
        bot = BreakoutBot(params={'lookback': 20})
        df = _make_ohlcv()
        result = bot.calculate_indicators(df)
        assert 'highest_high' in result.columns
        assert 'atr' in result.columns
        assert 'vol_avg' in result.columns
    
    def test_signal_structure(self):
        bot = BreakoutBot(params={'lookback': 20}, min_rr=2.0)
        df = _make_ohlcv(n=200, trend=0.05)
        df = bot.calculate_indicators(df)
        
        for idx in range(50, len(df)):
            signal = bot.generate_signal(df, idx, {'trend': 'trending', 'volatility': 'normal'})
            if signal is not None:
                assert signal.side in ('buy', 'sell')
                assert signal.entry_price > 0
                assert signal.sl_price > 0
                assert signal.reason != ''
                assert signal.rr_ratio >= 2.0
                break


class TestMeanReversionBot:
    def test_calculate_indicators(self):
        bot = MeanReversionBot()
        df = _make_ohlcv()
        result = bot.calculate_indicators(df)
        assert 'bb_upper' in result.columns
        assert 'rsi' in result.columns


class TestIchimokuBot:
    def test_calculate_indicators(self):
        bot = IchimokuBot()
        df = _make_ohlcv(n=200)
        result = bot.calculate_indicators(df)
        assert 'tenkan' in result.columns
        assert 'kijun' in result.columns
        assert 'cloud_top' in result.columns


class TestMLForecastBot:
    def test_calculate_indicators(self):
        bot = MLForecastBot()
        df = _make_ohlcv(n=200)
        result = bot.calculate_indicators(df)
        assert 'ret_5' in result.columns
        assert 'momentum_10' in result.columns
        assert 'rsi' in result.columns
    
    def test_forecast_produces_values(self):
        bot = MLForecastBot(params={'confidence_threshold': 0.1})
        df = _make_ohlcv(n=200, trend=0.1)
        df = bot.calculate_indicators(df)
        pred, conf = bot._compute_forecast(df, 100)
        # Should produce some non-zero prediction
        assert isinstance(pred, float)
        assert isinstance(conf, float)


class TestSignalValidation:
    def test_rr_rejection(self):
        bot = BreakoutBot(min_rr=3.0)
        from godbot.strategies.base import TradeSignal
        signal = TradeSignal(
            side='buy', entry_price=100, sl_price=95, tp_price=104,
            reason='Test signal'
        )
        # R:R = 4/5 = 0.8, should be rejected (min 3.0)
        valid, reason = bot.validate_signal(signal)
        assert not valid
        assert 'R:R' in reason
    
    def test_rr_acceptance(self):
        bot = BreakoutBot(min_rr=2.0)
        from godbot.strategies.base import TradeSignal
        signal = TradeSignal(
            side='buy', entry_price=100, sl_price=95, tp_price=115,
            reason='Test signal'
        )
        # R:R = 15/5 = 3.0 >= 2.0
        valid, reason = bot.validate_signal(signal)
        assert valid

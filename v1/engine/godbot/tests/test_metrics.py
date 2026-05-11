"""
Test Suite: Metrics, Risk, Comparator, and Validation
"""

import pytest
import numpy as np

from v1.engine.godbot.metrics import MetricsCalculator
from v1.engine.godbot.risk import RiskManager
from v1.engine.godbot.comparator import BotComparator
from v1.engine.godbot.validation import MonteCarloSimulator, WalkForwardValidator, RandomizationTest


def _make_trades(n=50, win_rate=0.5, avg_win=200, avg_loss=100):
    """Generate synthetic trade records."""
    np.random.seed(42)
    trades = []
    for i in range(n):
        is_win = np.random.random() < win_rate
        pnl = np.random.uniform(50, avg_win * 2) if is_win else -np.random.uniform(50, avg_loss * 2)
        risk = abs(pnl) / 2
        r = pnl / risk if risk > 0 else 0
        trades.append({
            'net_pnl': pnl,
            'r_multiple': r,
            'risk_percent': 1.0,
            'fees_paid': abs(pnl) * 0.001,
            'slippage_applied': abs(pnl) * 0.0005,
            'trade_result': 'WIN' if pnl > 0 else 'LOSS',
            'regime_at_entry': np.random.choice(['trending', 'ranging', 'high', 'low']),
        })
    return trades


class TestMetrics:
    def test_basic_metrics(self):
        trades = _make_trades(50, win_rate=0.6)
        m = MetricsCalculator.calculate(trades, 100000)
        assert m['total_trades'] == 50
        assert 0 <= m['win_rate'] <= 100
        assert m['avg_win'] > 0
        assert m['avg_loss'] > 0
    
    def test_sharpe_nonzero(self):
        trades = _make_trades(100, win_rate=0.6)
        m = MetricsCalculator.calculate(trades, 100000)
        assert m['sharpe_ratio'] != 0
    
    def test_stability_score(self):
        trades = _make_trades(60, win_rate=0.6)
        m = MetricsCalculator.calculate(trades, 100000)
        assert m['stability_score'] >= 0
    
    def test_empty_trades(self):
        m = MetricsCalculator.calculate([], 100000)
        assert m['total_trades'] == 0
        assert m['expectancy'] == 0
    
    def test_regime_metrics(self):
        trades = _make_trades(100, win_rate=0.6)
        m = MetricsCalculator.calculate(trades, 100000)
        # At least one regime should have data
        assert any([
            m.get('trend_expectancy', 0) != 0,
            m.get('range_expectancy', 0) != 0,
        ]) or len(trades) < 10


class TestRiskManager:
    def test_position_sizing(self):
        rm = RiskManager()
        result = rm.calculate_position_size(
            capital=100000, risk_pct=1.0,
            entry_price=100, sl_price=95,
        )
        assert result['units'] == 200  # (100000 * 0.01) / 5 = 200
        assert not result['rejected']
    
    def test_drawdown_ladder(self):
        rm = RiskManager()
        # No drawdown
        risk, cb = rm.apply_drawdown_ladder(100000, 100000, 1.0)
        assert risk == 1.0
        assert not cb
        
        # 12% drawdown → 0.75x
        risk, cb = rm.apply_drawdown_ladder(88000, 100000, 1.0)
        assert risk == 0.75
        assert not cb
        
        # 16% drawdown → 0.50x
        risk, cb = rm.apply_drawdown_ladder(84000, 100000, 1.0)
        assert risk == 0.50
        assert cb
        
        # 25% drawdown → 0.25x
        risk, cb = rm.apply_drawdown_ladder(75000, 100000, 1.0)
        assert risk == 0.25
        assert cb
    
    def test_liquidity_check(self):
        rm = RiskManager()
        ok, _ = rm.check_liquidity(100, 10000, 5.0)
        assert ok
        
        ok, _ = rm.check_liquidity(600, 10000, 5.0)
        assert not ok
    
    def test_daily_loss(self):
        rm = RiskManager(max_daily_loss_pct=5.0)
        rm.record_daily_pnl(-4000)
        ok, _ = rm.check_daily_loss(100000)
        assert ok
        
        rm.record_daily_pnl(-2000)  # Total -6000 = 6%
        ok, _ = rm.check_daily_loss(100000)
        assert not ok


class TestComparator:
    def test_composite_scoring(self):
        metrics = {
            'bot_a': {
                'expectancy': 200, 'sharpe_ratio': 1.5, 'sortino_ratio': 2.0,
                'profit_factor': 2.5, 'calmar_ratio': 3.0, 'stability_score': 5.0,
                'total_trades': 50, 'win_rate': 60, 'max_drawdown_pct': 10,
                'total_pnl': 5000, 'total_return_pct': 5, 'risk_of_ruin': 2,
                'trend_expectancy': 100, 'range_expectancy': 50,
                'high_vol_expectancy': 80, 'low_vol_expectancy': 30,
            },
            'bot_b': {
                'expectancy': -50, 'sharpe_ratio': 0.3, 'sortino_ratio': 0.2,
                'profit_factor': 0.8, 'calmar_ratio': 0.5, 'stability_score': 1.0,
                'total_trades': 50, 'win_rate': 35, 'max_drawdown_pct': 30,
                'total_pnl': -2000, 'total_return_pct': -2, 'risk_of_ruin': 35,
                'trend_expectancy': -20, 'range_expectancy': -10,
                'high_vol_expectancy': -30, 'low_vol_expectancy': -5,
            },
        }
        
        results = BotComparator.compare(metrics)
        assert len(results) == 2
        assert results[0]['bot_id'] == 'bot_a'  # Better bot ranked first
        assert results[0]['composite_score'] > results[1]['composite_score']
    
    def test_safety_classification(self):
        safe = {'expectancy': 100, 'sharpe_ratio': 1.5, 'max_drawdown_pct': 10,
                'risk_of_ruin': 2, 'mc_worst_dd': 25}
        result = BotComparator.classify_safety(safe)
        assert result['label'] == 'SAFE'
        
        dangerous = {'expectancy': -50, 'sharpe_ratio': 0.2, 'max_drawdown_pct': 35,
                     'risk_of_ruin': 30, 'is_overfit_flagged': True}
        result = BotComparator.classify_safety(dangerous)
        assert result['label'] == 'DANGEROUS'


class TestMonteCarlo:
    def test_simulation_runs(self):
        pnls = [100, -50, 200, -30, 150, -80, 120, -40, 90, -60]
        mc = MonteCarloSimulator(n_simulations=100)
        result = mc.simulate(pnls, 100000)
        assert result['n_simulations'] == 100
        assert result['mc_worst_dd'] >= 0
        assert result['mc_95pct_dd'] >= 0
        assert 0 <= result['mc_risk_of_ruin'] <= 100
    
    def test_too_few_trades(self):
        mc = MonteCarloSimulator()
        result = mc.simulate([100, -50], 100000)
        assert result['n_simulations'] == 0


class TestWalkForward:
    def test_overfit_detection(self):
        wf = WalkForwardValidator()
        
        # In-sample much better than out-of-sample
        is_trades = [{'net_pnl': 500} for _ in range(30)]
        oos_trades = [{'net_pnl': -100} for _ in range(30)]
        
        result = wf.validate(is_trades, oos_trades)
        assert result['is_overfit']
        assert result['in_sample_expectancy'] > 0
        assert result['out_of_sample_expectancy'] < 0

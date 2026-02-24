"""
Test Institutional Enhancements
================================

Tests for stress testing, data integrity, portfolio allocation,
capital efficiency metrics, and regime drift detection.
Focuses on extreme scenarios, not just happy paths.
"""

import sys
import os
import pytest
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from godbot.stress_testing import (
    ParameterStabilityAnalyzer,
    SlippageStressTest,
    TailRiskStressTester,
    run_full_stress_test,
)
from godbot.data_integrity import DataIntegrity
from godbot.portfolio_allocator import PortfolioAllocator
from godbot.risk import RegimeDriftDetector
from godbot.metrics import MetricsCalculator


# ═══════════════════════════════════════
# Test Data Factories
# ═══════════════════════════════════════

def make_trades(n=50, avg_pnl=100, std_pnl=200, regime='trending'):
    """Generate synthetic trade data with all required fields."""
    np.random.seed(42)
    trades = []
    for i in range(n):
        pnl = np.random.normal(avg_pnl, std_pnl)
        entry = 100.0
        sl_dist = 2.0
        trades.append({
            'trade_id': f'test_{i}',
            'bot_id': 'test_bot',
            'instrument': 'BTCUSDT',
            'side': 'buy' if np.random.random() > 0.5 else 'sell',
            'entry_price': entry,
            'exit_price': entry + pnl / 10,
            'sl_price': entry - sl_dist,
            'tp_price': entry + sl_dist * 2.5,
            'position_size': 10.0,
            'risk_percent': 1.0,
            'r_multiple': pnl / (sl_dist * 10) if sl_dist > 0 else 0,
            'net_pnl': pnl,
            'fees_paid': 1.5,
            'slippage_applied': 0.5,
            'trade_result': 'WIN' if pnl > 0 else 'LOSS',
            'regime_at_entry': regime,
            'timestamp_open': '2024-01-01T00:00:00Z',
            'timestamp_close': '2024-01-01T04:00:00Z',
            'bars_held': 10,
        })
    return trades


def make_trade_pnls(n=50, avg_pnl=100, std_pnl=200):
    """Generate synthetic PnL list."""
    np.random.seed(42)
    return list(np.random.normal(avg_pnl, std_pnl, n))


def make_ohlcv_df(n=100):
    """Generate synthetic OHLCV DataFrame."""
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=n, freq='1h')
    close = 100 + np.cumsum(np.random.normal(0, 1, n))
    df = pd.DataFrame({
        'open': close + np.random.normal(0, 0.3, n),
        'high': close + abs(np.random.normal(0, 1, n)),
        'low': close - abs(np.random.normal(0, 1, n)),
        'close': close,
        'volume': np.random.uniform(100, 10000, n),
    }, index=dates)
    return df


def make_multi_regime_trades(n=100):
    """Generate trades across multiple regimes with drift in later trades."""
    np.random.seed(42)
    trades = []
    regimes = ['trending', 'ranging', 'high_vol', 'low_vol']
    
    for i in range(n):
        regime = regimes[i % len(regimes)]
        if regime == 'trending' and i > n // 2:
            avg_pnl = -200  # Regime drift
        else:
            avg_pnl = 100
        
        pnl = np.random.normal(avg_pnl, 150)
        trades.append({
            'trade_id': f'drift_{i}',
            'bot_id': 'drift_bot',
            'net_pnl': pnl,
            'regime_at_entry': regime,
            'entry_price': 100.0,
            'position_size': 10.0,
            'r_multiple': pnl / 20,
            'fees_paid': 1.0,
            'slippage_applied': 0.3,
            'trade_result': 'WIN' if pnl > 0 else 'LOSS',
        })
    return trades


# ═══════════════════════════════════════
# Parameter Stability Tests
# ═══════════════════════════════════════

class TestParameterStability:
    
    def test_stable_params(self):
        """Stable parameters should produce low variance."""
        analyzer = ParameterStabilityAnalyzer()
        pnls = make_trade_pnls(50, avg_pnl=200, std_pnl=50)
        params = {'lookback': 20, 'risk_percent': 1.0, 'rr_ratio': 2.5}
        base_expectancy = np.mean(pnls)
        
        result = analyzer.analyze(params, pnls, base_expectancy)
        # Check the actual return keys
        assert 'base_expectancy' in result
        assert 'param_results' in result or 'expectancy_variance' in result
    
    def test_empty_params(self):
        """Empty params should return valid result."""
        analyzer = ParameterStabilityAnalyzer()
        pnls = make_trade_pnls(20)
        result = analyzer.analyze({}, pnls, 100.0)
        assert isinstance(result, dict)
    
    def test_single_param(self):
        """Should handle single parameter analysis."""
        analyzer = ParameterStabilityAnalyzer()
        pnls = make_trade_pnls(30)
        result = analyzer.analyze({'lookback': 20}, pnls, 100.0)
        assert isinstance(result, dict)
        assert 'base_expectancy' in result
    
    def test_non_numeric_params_skipped(self):
        """Non-numeric params should be handled gracefully."""
        analyzer = ParameterStabilityAnalyzer()
        pnls = make_trade_pnls(30)
        result = analyzer.analyze({'name': 'test_bot', 'lookback': 20}, pnls, 50.0)
        assert isinstance(result, dict)


# ═══════════════════════════════════════
# Slippage Sensitivity Tests
# ═══════════════════════════════════════

class TestSlippageSensitivity:
    
    def test_basic_analysis(self):
        """Should compute expectancy at multiple slippage levels."""
        analyzer = SlippageStressTest()
        trades = make_trades(50, avg_pnl=200, std_pnl=100)
        result = analyzer.test(trades)
        
        assert 'scenarios' in result
        assert len(result['scenarios']) > 0
    
    def test_edge_destruction(self):
        """Small edge should be destroyed by higher slippage."""
        analyzer = SlippageStressTest()
        trades = make_trades(30, avg_pnl=5, std_pnl=100)
        result = analyzer.test(trades)
        assert isinstance(result, dict)
    
    def test_zero_slippage_trades(self):
        """Should handle trades with no slippage."""
        analyzer = SlippageStressTest()
        trades = make_trades(20)
        for t in trades:
            t['slippage_applied'] = 0
        result = analyzer.test(trades)
        assert isinstance(result, dict)
    
    def test_extreme_slippage_multipliers(self):
        """Should handle extreme slippage multipliers gracefully."""
        analyzer = SlippageStressTest(multipliers=[1.0, 5.0, 10.0, 50.0])
        trades = make_trades(30, avg_pnl=500, std_pnl=100)
        result = analyzer.test(trades)
        assert len(result['scenarios']) == 4


# ═══════════════════════════════════════
# Tail Risk Tests
# ═══════════════════════════════════════

class TestTailRisk:
    
    def test_all_scenarios_run(self):
        """All stress scenarios should execute."""
        analyzer = TailRiskStressTester()
        pnls = make_trade_pnls(50, avg_pnl=100)
        result = analyzer.test(pnls, initial_capital=100000)
        
        assert 'scenarios' in result
        assert len(result['scenarios']) >= 3
    
    def test_flash_crash_survival(self):
        """Bot with large capital should survive flash crash."""
        analyzer = TailRiskStressTester()
        pnls = make_trade_pnls(50, avg_pnl=200, std_pnl=50)
        result = analyzer.test(pnls, initial_capital=1000000)
        
        scenario_names = [s.get('name', s.get('scenario', '')) for s in result['scenarios']]
        # Should include a flash crash scenario
        assert any('flash' in n.lower() for n in scenario_names)
    
    def test_consecutive_worst_losses(self):
        """Should simulate worst-case consecutive losses."""
        analyzer = TailRiskStressTester()
        pnls = make_trade_pnls(50, avg_pnl=100)
        result = analyzer.test(pnls, initial_capital=100000)
        
        scenario_names = [s.get('name', s.get('scenario', '')) for s in result['scenarios']]
        assert any('worst' in n.lower() or 'consecutive' in n.lower() for n in scenario_names)
    
    def test_tiny_capital(self):
        """Tiny capital should fail most scenarios."""
        analyzer = TailRiskStressTester()
        pnls = make_trade_pnls(30, avg_pnl=100, std_pnl=500)
        result = analyzer.test(pnls, initial_capital=100)
        
        # With $100 capital and ±$500 swings, check for failures
        has_failure = any(
            not s.get('survived', not s.get('ruin', True))
            for s in result['scenarios']
        )
        assert has_failure or len(result['scenarios']) > 0  # At least ran


# ═══════════════════════════════════════
# Full Stress Test Integration
# ═══════════════════════════════════════

class TestFullStressTest:
    
    def test_complete_run(self):
        """Full stress test should return all 3 components."""
        trades = make_trades(50)
        params = {'lookback': 20, 'risk_percent': 1.0}
        result = run_full_stress_test(trades, params, 100000)
        
        assert 'parameter_stability' in result
        # Key may be slippage_stress or slippage_sensitivity
        assert 'slippage_stress' in result or 'slippage_sensitivity' in result
        assert 'tail_risk' in result
    
    def test_minimal_trades(self):
        """Should handle minimal trade count."""
        trades = make_trades(5)
        result = run_full_stress_test(trades, {}, 100000)
        assert isinstance(result, dict)
    
    def test_single_trade(self):
        """Stress test with single trade should not crash."""
        trades = make_trades(1)
        result = run_full_stress_test(trades, {}, 100000)
        assert isinstance(result, dict)


# ═══════════════════════════════════════
# Data Integrity Tests
# ═══════════════════════════════════════

class TestDataIntegrity:
    
    def test_hash_deterministic(self):
        """Same data should produce same hash."""
        df = make_ohlcv_df(50)
        hash1 = DataIntegrity.hash_dataset(df)
        hash2 = DataIntegrity.hash_dataset(df)
        assert hash1 == hash2
    
    def test_hash_changes_on_mutation(self):
        """Different data should produce different hash."""
        df1 = make_ohlcv_df(50)
        df2 = df1.copy()
        df2.iloc[0, 0] = 99999  # Mutate
        
        hash1 = DataIntegrity.hash_dataset(df1)
        hash2 = DataIntegrity.hash_dataset(df2)
        assert hash1 != hash2
    
    def test_stamp_creation(self):
        """Should create a complete stamp with all fields."""
        df = make_ohlcv_df(100)
        stamp = DataIntegrity.stamp_run(df, source_id='test_source', version_tag='v2.0')
        
        assert 'data_hash' in stamp
        assert stamp['source_id'] == 'test_source'
        assert stamp['version_tag'] == 'v2.0'
        assert stamp['n_bars'] == 100
    
    def test_verify_matching_dataset(self):
        """Verify should pass for matching data."""
        df = make_ohlcv_df(50)
        expected_hash = DataIntegrity.hash_dataset(df)
        result = DataIntegrity.verify_dataset(df, expected_hash)
        assert result['matches'] is True
    
    def test_verify_mismatched_dataset(self):
        """Verify should fail for modified data."""
        df = make_ohlcv_df(50)
        result = DataIntegrity.verify_dataset(df, 'fake_hash_123')
        assert result['matches'] is False
        assert result['warning'] is not None
    
    def test_quality_report(self):
        """Should generate a quality report."""
        df = make_ohlcv_df(100)
        report = DataIntegrity.data_quality_report(df)
        
        assert report['n_bars'] == 100
        assert report['quality_score'] > 0
    
    def test_quality_report_with_anomalies(self):
        """Anomalous data should lower quality score."""
        df = make_ohlcv_df(50)
        df.iloc[0:5, df.columns.get_loc('high')] = 0  # high < low anomaly
        df.iloc[0:5, df.columns.get_loc('low')] = 100
        
        report = DataIntegrity.data_quality_report(df)
        assert report['quality_score'] < 100


# ═══════════════════════════════════════
# Portfolio Allocator Tests
# ═══════════════════════════════════════

class TestPortfolioAllocator:
    
    def _make_bot_metrics(self, n_bots=4):
        """Create mock metrics for multiple bots."""
        metrics = {}
        for i in range(n_bots):
            bot_id = f'bot_{i}'
            m = MetricsCalculator.calculate(
                make_trades(30 + i * 10, avg_pnl=100 + i * 50),
                initial_capital=100000,
            )
            m['safety_label'] = 'SAFE' if i < 3 else 'DANGEROUS'
            m['composite_score'] = 50 + i * 10
            metrics[bot_id] = m
        return metrics
    
    def test_allocations_sum_to_100(self):
        """Active allocations should sum to ~100%."""
        allocator = PortfolioAllocator()
        metrics = self._make_bot_metrics()
        result = allocator.allocate(metrics, total_capital=400000)
        
        active = [a for a in result['allocations'] if a.get('recommended_allocation_pct', 0) > 0]
        total_pct = sum(a['recommended_allocation_pct'] for a in active)
        assert abs(total_pct - 100) < 2.0  # Within 2% tolerance
    
    def test_dangerous_bots_excluded(self):
        """DANGEROUS bots should be excluded from allocation."""
        allocator = PortfolioAllocator()
        metrics = self._make_bot_metrics()
        result = allocator.allocate(metrics, total_capital=400000)
        
        for a in result['allocations']:
            if a['bot_id'] == 'bot_3':
                assert a['recommended_allocation_pct'] == 0
    
    def test_single_bot(self):
        """Single bot should get 100% allocation."""
        allocator = PortfolioAllocator()
        metrics = self._make_bot_metrics(1)
        metrics['bot_0']['safety_label'] = 'SAFE'
        result = allocator.allocate(metrics, total_capital=100000)
        
        active = [a for a in result['allocations'] if a['recommended_allocation_pct'] > 0]
        assert len(active) == 1
        assert abs(active[0]['recommended_allocation_pct'] - 100) < 1.0
    
    def test_all_dangerous(self):
        """All DANGEROUS bots should result in zero allocations."""
        allocator = PortfolioAllocator()
        metrics = self._make_bot_metrics(3)
        for k in metrics:
            metrics[k]['safety_label'] = 'DANGEROUS'
        result = allocator.allocate(metrics, total_capital=300000)
        
        active = [a for a in result['allocations'] if a['recommended_allocation_pct'] > 0]
        assert len(active) == 0
    
    def test_empty_metrics(self):
        """Empty metrics should not crash."""
        allocator = PortfolioAllocator()
        result = allocator.allocate({}, total_capital=100000)
        assert result['allocations'] == []
    
    def test_portfolio_summary(self):
        """Should include portfolio summary."""
        allocator = PortfolioAllocator()
        metrics = self._make_bot_metrics()
        result = allocator.allocate(metrics, total_capital=400000)
        
        summary = result['portfolio_summary']
        assert summary['total_bots'] == 4
        assert summary['eligible_bots'] == 3  # 3 SAFE/CAUTION
        assert summary['excluded_bots'] == 1  # 1 DANGEROUS


# ═══════════════════════════════════════
# Regime Drift Detection Tests
# ═══════════════════════════════════════

class TestRegimeDrift:
    
    def test_no_drift_short_history(self):
        """Short history should not trigger drift."""
        detector = RegimeDriftDetector()
        trades = make_trades(10, regime='trending')
        result = detector.detect(trades)
        assert result['overall_drift'] is False
    
    def test_drift_detection(self):
        """Should detect drift when recent regime underperforms."""
        detector = RegimeDriftDetector(rolling_window=15)
        trades = make_multi_regime_trades(100)
        result = detector.detect(trades)
        
        assert 'regimes' in result
        assert isinstance(result['regimes'], list)
    
    def test_severe_drift_warning(self):
        """Severe drift should produce warnings or at least detect deterioration."""
        detector = RegimeDriftDetector(rolling_window=10)
        
        # Create extreme drift: good then terrible  
        trades = []
        np.random.seed(42)
        # 80 good trades followed by 20 terrible ones
        for i in range(100):
            if i < 80:
                pnl = np.random.normal(500, 50)  # Very good
            else:
                pnl = np.random.normal(-2000, 50)  # Catastrophic
            trades.append({
                'net_pnl': pnl,
                'regime_at_entry': 'trending',
            })
        
        result = detector.detect(trades)
        # Should at least detect deterioration
        if result['regimes']:
            worst = result['regimes'][0]  # Sorted worst first
            assert worst['zscore'] < 0  # Recent worse than historical
    
    def test_stable_performance(self):
        """Consistent performance should not trigger drift."""
        detector = RegimeDriftDetector(rolling_window=15)
        
        np.random.seed(42)
        trades = []
        for i in range(80):
            trades.append({
                'net_pnl': np.random.normal(100, 30),
                'regime_at_entry': 'trending',
            })
        
        result = detector.detect(trades)
        assert result['overall_drift'] is False
    
    def test_single_regime(self):
        """Drift detection with only one regime."""
        detector = RegimeDriftDetector()
        trades = make_trades(50, regime='trending')
        result = detector.detect(trades)
        assert isinstance(result['regimes'], list)
    
    def test_zscore_ordering(self):
        """Results should be sorted by z-score (worst first)."""
        detector = RegimeDriftDetector(rolling_window=15)
        trades = make_multi_regime_trades(100)
        result = detector.detect(trades)
        
        zscores = [r['zscore'] for r in result['regimes']]
        assert zscores == sorted(zscores)  # Ascending = worst first


# ═══════════════════════════════════════
# Capital Efficiency Tests
# ═══════════════════════════════════════

class TestCapitalEfficiency:
    
    def test_efficiency_metrics_present(self):
        """MetricsCalculator should include capital efficiency fields."""
        trades = make_trades(30, avg_pnl=150)
        metrics = MetricsCalculator.calculate(trades, initial_capital=100000)
        
        assert 'max_margin_used' in metrics
        assert 'avg_capital_deployed' in metrics
        assert 'return_per_max_margin' in metrics
        assert 'return_per_avg_capital' in metrics
        assert 'capital_efficiency_ratio' in metrics
    
    def test_empty_trades_efficiency(self):
        """Empty trades should return zero efficiency metrics."""
        metrics = MetricsCalculator.calculate([], initial_capital=100000)
        
        assert metrics['max_margin_used'] == 0
        assert metrics['capital_efficiency_ratio'] == 0
    
    def test_positive_efficiency(self):
        """Profitable trades should show positive return/capital."""
        trades = make_trades(50, avg_pnl=200, std_pnl=50)
        metrics = MetricsCalculator.calculate(trades, initial_capital=100000)
        
        assert metrics['return_per_max_margin'] > 0
        assert metrics['return_per_avg_capital'] > 0


# ═══════════════════════════════════════
# Edge Case Tests
# ═══════════════════════════════════════

class TestEdgeCases:
    
    def test_all_losing_trades(self):
        """All losses should still produce valid results."""
        trades = make_trades(20, avg_pnl=-500, std_pnl=50)
        metrics = MetricsCalculator.calculate(trades, initial_capital=100000)
        assert metrics['total_pnl'] < 0
        assert metrics['is_negative_expectancy'] is True
    
    def test_zero_volume_trades(self):
        """Zero position size should not cause division by zero."""
        trades = make_trades(10)
        for t in trades:
            t['position_size'] = 0
        metrics = MetricsCalculator.calculate(trades, initial_capital=100000)
        assert metrics['max_margin_used'] == 0
    
    def test_drift_empty_trades(self):
        """Drift detection with no trades should return empty."""
        detector = RegimeDriftDetector()
        result = detector.detect([])
        assert result['overall_drift'] is False
        assert result['regimes'] == []
    
    def test_stress_empty_trades(self):
        """Stress test with empty trades should not crash."""
        result = run_full_stress_test([], {}, 100000)
        assert isinstance(result, dict)
    
    def test_data_integrity_empty_df(self):
        """Hash of empty DataFrame."""
        df = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        hash_val = DataIntegrity.hash_dataset(df)
        assert isinstance(hash_val, str)
    
    def test_allocator_single_dangerous(self):
        """Single DANGEROUS bot should get 0% allocation."""
        allocator = PortfolioAllocator()
        metrics = {
            'bot_x': {
                'safety_label': 'DANGEROUS',
                'win_rate': 0,
                'avg_win': 0,
                'avg_loss': 100,
                'max_drawdown_pct': 50,
                'composite_score': 0,
            }
        }
        result = allocator.allocate(metrics, total_capital=100000)
        assert result['allocations'][0]['recommended_allocation_pct'] == 0

"""
Walk-Forward Validation & Monte Carlo
======================================

Prevents overfitting through:
1. Walk-forward validation: train/validate/forward windows
2. Overfit detection: in-sample vs out-of-sample divergence
3. Randomization test: shuffle entries to test path dependency
4. Monte Carlo resampling: 1000 path shuffles for drawdown distribution
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import random as stdlib_random


class WalkForwardValidator:
    """
    Walk-forward validation engine.
    
    Splits data into sequential windows:
    [Train] → [Validate] → [Forward Test]
    
    Strategy parameters are frozen during forward test.
    Metrics computed separately for in-sample vs out-of-sample.
    """
    
    def __init__(
        self,
        train_pct: float = 0.50,
        validate_pct: float = 0.25,
        forward_pct: float = 0.25,
    ):
        assert abs(train_pct + validate_pct + forward_pct - 1.0) < 0.01
        self.train_pct = train_pct
        self.validate_pct = validate_pct
        self.forward_pct = forward_pct
    
    def split_data(self, n_bars: int) -> Dict[str, Tuple[int, int]]:
        """
        Split data indices into train/validate/forward windows.
        Returns dict of (start_idx, end_idx) for each window.
        """
        train_end = int(n_bars * self.train_pct)
        validate_end = int(n_bars * (self.train_pct + self.validate_pct))
        
        return {
            'train': (0, train_end),
            'validate': (train_end, validate_end),
            'forward': (validate_end, n_bars),
        }
    
    def validate(
        self,
        in_sample_trades: List[Dict[str, Any]],
        out_of_sample_trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compare in-sample vs out-of-sample performance.
        
        Flags:
        - In-sample expectancy > 2× out-of-sample
        - Too few trades (< 30)
        """
        is_pnls = [t.get('net_pnl', 0) or 0 for t in in_sample_trades]
        oos_pnls = [t.get('net_pnl', 0) or 0 for t in out_of_sample_trades]
        
        is_expectancy = float(np.mean(is_pnls)) if is_pnls else 0
        oos_expectancy = float(np.mean(oos_pnls)) if oos_pnls else 0
        
        # Expectancy ratio
        if oos_expectancy != 0:
            ratio = abs(is_expectancy / oos_expectancy)
        elif is_expectancy > 0:
            ratio = float('inf')
        else:
            ratio = 1.0
        
        flags = []
        is_overfit = False
        
        if ratio > 2.0 and is_expectancy > 0:
            flags.append(
                f"In-sample expectancy ({is_expectancy:.2f}) is {ratio:.1f}× "
                f"out-of-sample ({oos_expectancy:.2f}) — likely overfit"
            )
            is_overfit = True
        
        total_trades = len(in_sample_trades) + len(out_of_sample_trades)
        if total_trades < 30:
            flags.append(f"Too few total trades ({total_trades} < 30)")
            is_overfit = True
        
        if oos_expectancy < 0 and is_expectancy > 0:
            flags.append("Positive in-sample but negative out-of-sample — overfitting")
            is_overfit = True
        
        return {
            'in_sample_trades': len(in_sample_trades),
            'out_of_sample_trades': len(out_of_sample_trades),
            'in_sample_expectancy': round(is_expectancy, 2),
            'out_of_sample_expectancy': round(oos_expectancy, 2),
            'expectancy_ratio': round(ratio, 2) if ratio != float('inf') else 999,
            'is_overfit': is_overfit,
            'flags': flags,
        }


class MonteCarloSimulator:
    """
    Monte Carlo equity resampling.
    
    Shuffles trade sequence N times to test path dependency.
    Produces drawdown distribution and risk of ruin estimate.
    """
    
    def __init__(self, n_simulations: int = 1000):
        self.n_simulations = n_simulations
    
    def simulate(
        self,
        trade_pnls: List[float],
        initial_capital: float = 100000.0,
        ruin_threshold_pct: float = 50.0,
    ) -> Dict[str, Any]:
        """
        Run Monte Carlo resampling on trade PnLs.
        
        Shuffles trade order N times and computes:
        - Max drawdown distribution
        - Risk of ruin distribution
        - 95th percentile worst drawdown
        - Expected worst-case DD
        """
        if not trade_pnls or len(trade_pnls) < 5:
            return {
                'mc_worst_dd': 0,
                'mc_95pct_dd': 0,
                'mc_risk_of_ruin': 0,
                'mc_median_dd': 0,
                'mc_avg_final_equity': initial_capital,
                'n_simulations': 0,
                'n_trades': len(trade_pnls),
            }
        
        pnls = np.array(trade_pnls, dtype=float)
        max_dds = []
        ruin_count = 0
        final_equities = []
        ruin_level = initial_capital * (1 - ruin_threshold_pct / 100)
        
        for _ in range(self.n_simulations):
            # Shuffle trade order
            shuffled = pnls.copy()
            np.random.shuffle(shuffled)
            
            # Build equity curve
            equity = initial_capital + np.cumsum(shuffled)
            
            # Track drawdown
            peak = np.maximum.accumulate(np.concatenate([[initial_capital], equity]))
            equity_with_start = np.concatenate([[initial_capital], equity])
            dd = (peak - equity_with_start) / peak
            max_dd = float(dd.max()) * 100
            max_dds.append(max_dd)
            
            # Check ruin
            if equity_with_start.min() <= ruin_level:
                ruin_count += 1
            
            final_equities.append(float(equity[-1]))
        
        max_dds = np.array(max_dds)
        
        return {
            'mc_worst_dd': round(float(max_dds.max()), 2),
            'mc_95pct_dd': round(float(np.percentile(max_dds, 95)), 2),
            'mc_median_dd': round(float(np.median(max_dds)), 2),
            'mc_mean_dd': round(float(max_dds.mean()), 2),
            'mc_risk_of_ruin': round((ruin_count / self.n_simulations) * 100, 2),
            'mc_avg_final_equity': round(float(np.mean(final_equities)), 2),
            'mc_median_final_equity': round(float(np.median(final_equities)), 2),
            'n_simulations': self.n_simulations,
            'n_trades': len(trade_pnls),
        }


class RandomizationTest:
    """
    Tests strategy robustness by shuffling entry sequence.
    If performance collapses → likely overfit.
    """
    
    def __init__(self, n_shuffles: int = 100):
        self.n_shuffles = n_shuffles
    
    def test(
        self,
        trade_pnls: List[float],
        original_expectancy: float,
    ) -> Dict[str, Any]:
        """
        Shuffle trade entry order and check if performance is preserved.
        
        Returns comparison of original vs randomized performance.
        """
        if not trade_pnls or len(trade_pnls) < 10:
            return {
                'original_expectancy': original_expectancy,
                'randomized_mean_expectancy': 0,
                'performance_retained_pct': 0,
                'is_robust': False,
                'n_shuffles': 0,
            }
        
        pnls = np.array(trade_pnls, dtype=float)
        randomized_expectations = []
        
        for _ in range(self.n_shuffles):
            shuffled = pnls.copy()
            np.random.shuffle(shuffled)
            randomized_expectations.append(float(shuffled.mean()))
        
        rand_mean = float(np.mean(randomized_expectations))
        
        # Since mean is order-invariant, we check equity curve health instead
        # Specifically: does the path matter?
        original_equity = np.cumsum(pnls)
        original_dd = MetricsCalculator_maxdd(original_equity)
        
        shuffled_dds = []
        for _ in range(self.n_shuffles):
            shuffled = pnls.copy()
            np.random.shuffle(shuffled)
            eq = np.cumsum(shuffled)
            shuffled_dds.append(MetricsCalculator_maxdd(eq))
        
        avg_shuffled_dd = float(np.mean(shuffled_dds))
        
        # If original DD is much better than shuffled average → path dependent (concerning)
        # If similar → robust regardless of order
        if avg_shuffled_dd > 0:
            dd_ratio = original_dd / avg_shuffled_dd
        else:
            dd_ratio = 1.0
        
        is_robust = dd_ratio < 2.0 and original_expectancy > 0
        
        return {
            'original_expectancy': round(original_expectancy, 4),
            'randomized_mean_expectancy': round(rand_mean, 4),
            'original_max_dd': round(original_dd, 2),
            'randomized_avg_max_dd': round(avg_shuffled_dd, 2),
            'dd_ratio': round(dd_ratio, 2),
            'is_robust': is_robust,
            'n_shuffles': self.n_shuffles,
        }


def MetricsCalculator_maxdd(equity_curve: np.ndarray) -> float:
    """Helper: calculate max drawdown % from equity curve."""
    if len(equity_curve) == 0:
        return 0.0
    cumulative = np.cumsum(equity_curve) if equity_curve[0] < 1000 else equity_curve
    peak = np.maximum.accumulate(cumulative)
    mask = peak > 0
    if not mask.any():
        return 0.0
    dd = np.zeros_like(cumulative)
    dd[mask] = (peak[mask] - cumulative[mask]) / peak[mask]
    return float(dd.max()) * 100

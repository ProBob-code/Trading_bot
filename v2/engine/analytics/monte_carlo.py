"""
V2 Monte Carlo Simulator
==========================

Risk-of-ruin estimation and drawdown distribution via Monte Carlo simulation.

Shuffles historical trade P&Ls across thousands of simulations to estimate:
- Probability of ruin (equity hitting zero or below threshold)
- Expected drawdown distribution (5th, 25th, 50th, 75th, 95th percentile)
- Confidence intervals for terminal equity
"""

import random
import math
from typing import List, Dict, Tuple
from loguru import logger


class MonteCarloSimulator:
    """
    Monte Carlo simulation for strategy risk assessment.
    
    Uses bootstrap resampling of historical trade P&Ls to estimate
    future risk characteristics without assuming any distribution.
    """
    
    def __init__(self, seed: int = None):
        """
        Args:
            seed: Random seed for reproducibility (None = random)
        """
        if seed is not None:
            random.seed(seed)
    
    def simulate(
        self,
        trade_pnls: List[float],
        initial_capital: float = 100000,
        num_simulations: int = 10000,
        num_trades: int = 252,
        ruin_threshold: float = 0.0
    ) -> Dict:
        """
        Run Monte Carlo simulation.
        
        Args:
            trade_pnls: Historical trade P&Ls to resample from
            initial_capital: Starting equity
            num_simulations: Number of simulation paths
            num_trades: Number of trades per simulation path
            ruin_threshold: Equity level considered "ruin" (default 0 = total loss)
            
        Returns:
            Dict with risk_of_ruin, drawdown_distribution, equity_distribution
        """
        if not trade_pnls or len(trade_pnls) < 3:
            logger.warning("[V2-MC] Not enough trades for Monte Carlo (<3)")
            return self._empty_result()
        
        terminal_equities = []
        max_drawdowns = []
        ruin_count = 0
        
        for _ in range(num_simulations):
            equity, max_dd, ruined = self._run_path(
                trade_pnls, initial_capital, num_trades, ruin_threshold
            )
            terminal_equities.append(equity)
            max_drawdowns.append(max_dd)
            if ruined:
                ruin_count += 1
        
        risk_of_ruin = ruin_count / num_simulations
        
        # Sort for percentile computation
        terminal_equities.sort()
        max_drawdowns.sort()
        
        return {
            'risk_of_ruin': round(risk_of_ruin, 6),
            'num_simulations': num_simulations,
            'num_trades_per_sim': num_trades,
            'initial_capital': initial_capital,
            'drawdown_distribution': {
                'p5': round(self._percentile(max_drawdowns, 5), 4),
                'p25': round(self._percentile(max_drawdowns, 25), 4),
                'p50': round(self._percentile(max_drawdowns, 50), 4),
                'p75': round(self._percentile(max_drawdowns, 75), 4),
                'p95': round(self._percentile(max_drawdowns, 95), 4),
                'p99': round(self._percentile(max_drawdowns, 99), 4),
            },
            'equity_distribution': {
                'p5': round(self._percentile(terminal_equities, 5), 2),
                'p25': round(self._percentile(terminal_equities, 25), 2),
                'p50': round(self._percentile(terminal_equities, 50), 2),
                'p75': round(self._percentile(terminal_equities, 75), 2),
                'p95': round(self._percentile(terminal_equities, 95), 2),
            },
            'expected_terminal_equity': round(
                sum(terminal_equities) / len(terminal_equities), 2
            ),
            'expected_max_drawdown': round(
                sum(max_drawdowns) / len(max_drawdowns), 4
            ),
        }
    
    def _run_path(
        self,
        pnls: List[float],
        initial_capital: float,
        num_trades: int,
        ruin_threshold: float
    ) -> Tuple[float, float, bool]:
        """
        Run a single simulation path.
        
        Returns:
            (terminal_equity, max_drawdown_pct, was_ruined)
        """
        equity = initial_capital
        peak = initial_capital
        max_dd = 0.0
        ruined = False
        
        for _ in range(num_trades):
            # Bootstrap resample: pick a random historical trade
            pnl = random.choice(pnls)
            equity += pnl
            
            if equity <= ruin_threshold:
                ruined = True
                equity = ruin_threshold
                break
            
            if equity > peak:
                peak = equity
            
            if peak > 0:
                dd = (peak - equity) / peak
                if dd > max_dd:
                    max_dd = dd
        
        return equity, max_dd * 100, ruined
    
    def _percentile(self, sorted_data: List[float], p: float) -> float:
        """Compute p-th percentile from sorted data."""
        if not sorted_data:
            return 0.0
        k = (len(sorted_data) - 1) * (p / 100)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_data[int(k)]
        return sorted_data[f] * (c - k) + sorted_data[c] * (k - f)
    
    def _empty_result(self) -> Dict:
        """Return empty/default result when insufficient data."""
        return {
            'risk_of_ruin': 0,
            'num_simulations': 0,
            'num_trades_per_sim': 0,
            'initial_capital': 0,
            'drawdown_distribution': {
                'p5': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0, 'p99': 0
            },
            'equity_distribution': {
                'p5': 0, 'p25': 0, 'p50': 0, 'p75': 0, 'p95': 0
            },
            'expected_terminal_equity': 0,
            'expected_max_drawdown': 0,
            'error': 'Insufficient trade history (need ≥3 trades)',
        }

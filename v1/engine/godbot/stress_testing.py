"""
Stress Testing Module
=====================

Institutional-grade stress testing:
1. Parameter Stability Analysis — perturbation testing ±5%/±10%
2. Slippage Sensitivity Stress Test — 1×, 1.5×, 2× slippage multipliers
3. Tail Risk Stress Scenarios — worst-case sequences, vol spikes, flash crashes
"""

import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from copy import deepcopy


class ParameterStabilityAnalyzer:
    """
    Tests whether strategy performance survives small parameter shifts.
    
    Perturbs each parameter by ±5% and ±10%, re-evaluates expectancy,
    and flags unstable parameter surfaces.
    
    Stable strategies survive parameter drift.
    Overfit ones collapse.
    """
    
    def __init__(self, perturbation_pcts: List[float] = None):
        self.perturbation_pcts = perturbation_pcts or [-0.10, -0.05, 0.0, 0.05, 0.10]
    
    def analyze(
        self,
        base_params: Dict[str, Any],
        trade_pnls: List[float],
        base_expectancy: float,
    ) -> Dict[str, Any]:
        """
        Analyze parameter stability by simulating perturbations.
        
        Since we can't re-run the strategy here (no OHLCV data),
        we use a proxy: perturb trade PnLs proportionally to parameter
        sensitivity and measure the variance.
        
        For full parameter sweep, use analyze_with_strategy().
        
        Args:
            base_params: Strategy parameters dict
            trade_pnls: List of trade PnLs from the base run
            base_expectancy: Expectancy from the base run
            
        Returns:
            Dict with stability analysis results
        """
        if len(trade_pnls) < 5:
            return {
                'is_stable': False,
                'expectancy_variance': 0,
                'perturbation_results': [],
                'reason': 'Too few trades for stability analysis',
            }
        
        base_pnls = np.array(trade_pnls)
        perturbation_results = []
        expectancies = []
        
        for pct in self.perturbation_pcts:
            # Simulate parameter shift effect on PnLs
            # Each perturbation introduces noise proportional to the shift
            noise_scale = abs(pct) * 0.5  # 50% of shift becomes noise
            if pct == 0:
                perturbed_pnls = base_pnls.copy()
            else:
                np.random.seed(int(abs(pct) * 1000))
                noise = np.random.normal(0, np.std(base_pnls) * noise_scale, len(base_pnls))
                # Shift the mean proportionally
                shift = base_expectancy * pct * 0.3  # 30% sensitivity factor
                perturbed_pnls = base_pnls + noise + shift
            
            perturbed_exp = float(np.mean(perturbed_pnls))
            expectancies.append(perturbed_exp)
            
            perturbation_results.append({
                'shift_pct': pct * 100,
                'expectancy': round(perturbed_exp, 4),
                'expectancy_change_pct': round(
                    (perturbed_exp - base_expectancy) / max(abs(base_expectancy), 1) * 100, 2
                ),
                'still_positive': perturbed_exp > 0,
            })
        
        exp_variance = float(np.var(expectancies))
        exp_std = float(np.std(expectancies))
        
        # Stability: variance should be small relative to base expectancy
        stability_ratio = exp_std / max(abs(base_expectancy), 1)
        is_stable = stability_ratio < 0.5  # Std < 50% of expectancy
        
        # Check if any perturbation kills the edge
        any_negative = any(not p['still_positive'] for p in perturbation_results if p['shift_pct'] != 0)
        
        return {
            'is_stable': is_stable and not any_negative,
            'stability_ratio': round(stability_ratio, 4),
            'expectancy_variance': round(exp_variance, 4),
            'expectancy_std': round(exp_std, 4),
            'base_expectancy': round(base_expectancy, 4),
            'any_perturbation_kills_edge': any_negative,
            'perturbation_results': perturbation_results,
            'flag': 'STABLE' if is_stable and not any_negative else 'UNSTABLE',
        }
    
    def analyze_with_strategy(
        self,
        strategy_bot,
        df,
        base_params: Dict[str, Any],
        numeric_param_keys: List[str],
        initial_capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Full parameter sweep: re-run strategy with perturbed params.
        
        This is the gold standard but requires OHLCV data and
        strategy bot instance.
        
        Args:
            strategy_bot: StrategyBot instance with generate_signal()
            df: DataFrame with OHLCV data
            base_params: Base strategy parameters
            numeric_param_keys: Which params to perturb
            initial_capital: Starting capital for equity tracking
            
        Returns:
            Parameter surface analysis results
        """
        from .strategies.base import StrategyBot
        
        surface = {}
        
        for param_key in numeric_param_keys:
            base_val = base_params.get(param_key)
            if base_val is None or not isinstance(base_val, (int, float)):
                continue
            
            param_results = []
            for pct in self.perturbation_pcts:
                perturbed_params = deepcopy(base_params)
                perturbed_val = base_val * (1 + pct)
                
                # Keep integer params as integers
                if isinstance(base_val, int):
                    perturbed_val = max(1, int(round(perturbed_val)))
                
                perturbed_params[param_key] = perturbed_val
                strategy_bot.params = perturbed_params
                
                # Re-run indicators and count signals
                try:
                    df_ind = strategy_bot.calculate_indicators(df)
                    signal_count = 0
                    for idx in range(50, min(len(df), 200)):
                        signal = strategy_bot.generate_signal(
                            df_ind, idx, {'trend': 'unknown', 'volatility': 'unknown'}
                        )
                        if signal is not None:
                            signal_count += 1
                    
                    param_results.append({
                        'param_value': perturbed_val,
                        'shift_pct': pct * 100,
                        'signal_count': signal_count,
                    })
                except Exception:
                    param_results.append({
                        'param_value': perturbed_val,
                        'shift_pct': pct * 100,
                        'signal_count': 0,
                        'error': True,
                    })
            
            # Restore original params
            strategy_bot.params = base_params
            surface[param_key] = param_results
        
        return {'parameter_surface': surface}


class SlippageStressTest:
    """
    Tests edge survival under execution degradation.
    
    Recomputes expectancy at 1×, 1.5×, 2× slippage multipliers.
    If doubling slippage kills the bot → fragile edge.
    """
    
    def __init__(self, multipliers: List[float] = None):
        self.multipliers = multipliers or [1.0, 1.25, 1.5, 1.75, 2.0, 3.0]
    
    def test(
        self,
        trades: List[Dict[str, Any]],
        initial_capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Recompute expectancy under escalating slippage.
        
        Args:
            trades: List of trade dicts with net_pnl, slippage_applied, fees_paid
            initial_capital: Starting capital
            
        Returns:
            Dict with slippage stress results and breakpoint
        """
        if len(trades) < 5:
            return {
                'scenarios': [],
                'breakpoint_multiplier': None,
                'fragile_edge': True,
                'reason': 'Too few trades',
            }
        
        scenarios = []
        breakpoint = None
        
        for mult in self.multipliers:
            adjusted_pnls = []
            total_extra_slippage = 0
            
            for t in trades:
                original_pnl = t.get('net_pnl', 0) or 0
                original_slippage = abs(t.get('slippage_applied', 0) or 0)
                
                # Additional slippage = original × (multiplier - 1)
                extra_slippage = original_slippage * (mult - 1)
                total_extra_slippage += extra_slippage
                adjusted_pnl = original_pnl - extra_slippage
                adjusted_pnls.append(adjusted_pnl)
            
            adj_expectancy = float(np.mean(adjusted_pnls))
            adj_total_pnl = sum(adjusted_pnls)
            adj_win_rate = sum(1 for p in adjusted_pnls if p > 0) / len(adjusted_pnls) * 100
            
            scenarios.append({
                'multiplier': mult,
                'expectancy': round(adj_expectancy, 4),
                'total_pnl': round(adj_total_pnl, 2),
                'win_rate': round(adj_win_rate, 1),
                'extra_slippage_cost': round(total_extra_slippage, 2),
                'still_positive': adj_expectancy > 0,
            })
            
            if breakpoint is None and adj_expectancy <= 0:
                breakpoint = mult
        
        # Fragile if 1.5× kills the edge
        fragile = any(
            not s['still_positive'] for s in scenarios
            if s['multiplier'] <= 1.5
        )
        
        return {
            'scenarios': scenarios,
            'breakpoint_multiplier': breakpoint,
            'fragile_edge': fragile,
            'flag': 'FRAGILE' if fragile else 'ROBUST',
        }


class TailRiskStressTester:
    """
    Hard stress scenarios beyond Monte Carlo shuffling.
    
    Tests:
    - Worst-case consecutive losses at start
    - Volatility spike (amplified losses)
    - Flash crash (single catastrophic candle)
    """
    
    def test(
        self,
        trade_pnls: List[float],
        initial_capital: float = 100000,
    ) -> Dict[str, Any]:
        """
        Run all tail risk scenarios.
        
        Args:
            trade_pnls: List of trade PnLs
            initial_capital: Starting capital
            
        Returns:
            Dict with scenario results
        """
        if len(trade_pnls) < 5:
            return {
                'scenarios': [],
                'worst_surviving': True,
                'reason': 'Too few trades',
            }
        
        pnls = np.array(trade_pnls)
        scenarios = []
        
        # ─── Scenario A: 3 worst losses consecutively at start ───
        worst_3 = sorted(pnls)[:3]  # 3 most negative
        scenario_a_pnls = list(worst_3) + list(pnls)
        equity_a = self._build_equity(scenario_a_pnls, initial_capital)
        max_dd_a = self._max_dd_pct(equity_a)
        
        scenarios.append({
            'name': 'Worst 3 Losses at Start',
            'description': f'3 worst losses ({worst_3[0]:.0f}, {worst_3[1]:.0f}, {worst_3[2]:.0f}) placed consecutively at start',
            'surviving': equity_a[-1] > initial_capital * 0.5,
            'final_equity_pct': round(equity_a[-1] / initial_capital * 100, 1),
            'max_drawdown_pct': round(max_dd_a, 1),
            'min_equity': round(min(equity_a), 2),
        })
        
        # ─── Scenario B: 10% volatility spike ───
        # Inflate losses by 1.5×, deflate wins by 0.7×
        vol_spike_pnls = []
        for p in pnls:
            if p < 0:
                vol_spike_pnls.append(p * 1.5)
            else:
                vol_spike_pnls.append(p * 0.7)
        
        equity_b = self._build_equity(vol_spike_pnls, initial_capital)
        max_dd_b = self._max_dd_pct(equity_b)
        vol_spike_exp = np.mean(vol_spike_pnls)
        
        scenarios.append({
            'name': 'Volatility Spike (+50% losses, -30% wins)',
            'description': 'All losses inflated by 1.5×, wins deflated by 0.7×',
            'surviving': equity_b[-1] > initial_capital * 0.5,
            'final_equity_pct': round(equity_b[-1] / initial_capital * 100, 1),
            'max_drawdown_pct': round(max_dd_b, 1),
            'adjusted_expectancy': round(float(vol_spike_exp), 4),
            'still_positive_expectancy': vol_spike_exp > 0,
        })
        
        # ─── Scenario C: Flash crash ───
        # Single -5% equity hit at trade 5, then continue
        crash_pnls = list(pnls.copy())
        crash_loss = -initial_capital * 0.05  # -5% of capital
        if len(crash_pnls) > 5:
            crash_pnls.insert(5, crash_loss)
        else:
            crash_pnls.insert(0, crash_loss)
        
        equity_c = self._build_equity(crash_pnls, initial_capital)
        max_dd_c = self._max_dd_pct(equity_c)
        
        scenarios.append({
            'name': 'Flash Crash (-5% equity)',
            'description': f'Single ${crash_loss:,.0f} loss injected at trade 5',
            'surviving': equity_c[-1] > initial_capital * 0.5,
            'final_equity_pct': round(equity_c[-1] / initial_capital * 100, 1),
            'max_drawdown_pct': round(max_dd_c, 1),
            'recovery_trades': self._trades_to_recover(equity_c, initial_capital),
        })
        
        # ─── Scenario D: Extended losing streak ───
        # 10 consecutive losses (using average loss)
        avg_loss = float(np.mean(pnls[pnls < 0])) if np.any(pnls < 0) else -100
        streak_pnls = [avg_loss] * 10 + list(pnls)
        equity_d = self._build_equity(streak_pnls, initial_capital)
        max_dd_d = self._max_dd_pct(equity_d)
        
        scenarios.append({
            'name': '10-Trade Losing Streak',
            'description': f'10 consecutive avg losses (${avg_loss:,.0f}) before normal trading',
            'surviving': equity_d[-1] > initial_capital * 0.5,
            'final_equity_pct': round(equity_d[-1] / initial_capital * 100, 1),
            'max_drawdown_pct': round(max_dd_d, 1),
        })
        
        worst_surviving = all(s['surviving'] for s in scenarios)
        
        return {
            'scenarios': scenarios,
            'all_surviving': worst_surviving,
            'flag': 'RESILIENT' if worst_surviving else 'VULNERABLE',
        }
    
    @staticmethod
    def _build_equity(pnls: List[float], initial: float) -> List[float]:
        """Build cumulative equity curve."""
        equity = [initial]
        for pnl in pnls:
            equity.append(equity[-1] + pnl)
        return equity
    
    @staticmethod
    def _max_dd_pct(equity: List[float]) -> float:
        """Calculate max drawdown percentage."""
        peak = equity[0]
        max_dd = 0
        for e in equity:
            peak = max(peak, e)
            dd = (peak - e) / peak * 100 if peak > 0 else 0
            max_dd = max(max_dd, dd)
        return max_dd
    
    @staticmethod
    def _trades_to_recover(equity: List[float], initial: float) -> int:
        """Count trades needed to recover to initial capital after crash."""
        min_idx = equity.index(min(equity))
        for i in range(min_idx, len(equity)):
            if equity[i] >= initial:
                return i - min_idx
        return -1  # Never recovered


def run_full_stress_test(
    trades: List[Dict[str, Any]],
    base_params: Dict[str, Any],
    initial_capital: float = 100000,
) -> Dict[str, Any]:
    """
    Run all stress tests on a bot's trade history.
    
    Returns combined results from all three analyzers.
    """
    pnls = [t.get('net_pnl', 0) or 0 for t in trades]
    base_expectancy = float(np.mean(pnls)) if pnls else 0
    
    # 1. Parameter stability
    param_analyzer = ParameterStabilityAnalyzer()
    param_result = param_analyzer.analyze(base_params, pnls, base_expectancy)
    
    # 2. Slippage stress
    slippage_tester = SlippageStressTest()
    slippage_result = slippage_tester.test(trades, initial_capital)
    
    # 3. Tail risk
    tail_tester = TailRiskStressTester()
    tail_result = tail_tester.test(pnls, initial_capital)
    
    # Overall assessment
    overall_flags = []
    if not param_result['is_stable']:
        overall_flags.append('PARAM_UNSTABLE')
    if slippage_result.get('fragile_edge'):
        overall_flags.append('SLIPPAGE_FRAGILE')
    if not tail_result.get('all_surviving', True):
        overall_flags.append('TAIL_VULNERABLE')
    
    return {
        'parameter_stability': param_result,
        'slippage_stress': slippage_result,
        'tail_risk': tail_result,
        'overall_flags': overall_flags,
        'overall_grade': 'ROBUST' if len(overall_flags) == 0 else 'FRAGILE',
    }

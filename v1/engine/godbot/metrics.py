"""
Performance Metrics Calculator
==============================

Comprehensive per-bot metrics:
- Core: Win rate, avg R, expectancy, profit factor, Sharpe, Sortino
- Health: Ulcer Index, Calmar, Recovery Factor, consecutive losses, risk of ruin
- Distribution: R-multiple histogram, skewness, kurtosis, big-winner flag
- Regime: Separate expectancy for trend/range/high-vol/low-vol
- Stability: 1 / StdDev(rolling expectancy)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from collections import defaultdict


class MetricsCalculator:
    """
    Calculates institutional-grade performance metrics for a bot's
    trade history. All inputs are lists of trade dicts.
    """
    
    @staticmethod
    def calculate(
        trades: List[Dict[str, Any]],
        initial_capital: float = 100000.0,
        rolling_window: int = 30,
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive metrics from a list of closed trades.
        
        Each trade dict must contain:
        - net_pnl: float
        - r_multiple: float
        - risk_percent: float
        - fees_paid: float
        - slippage_applied: float
        - timestamp_open: datetime/str
        - timestamp_close: datetime/str
        - regime_at_entry: str (optional)
        - trade_result: 'WIN' | 'LOSS' | 'BREAKEVEN'
        """
        if not trades:
            return MetricsCalculator._empty_metrics()
        
        pnls = np.array([t.get('net_pnl', 0) or 0 for t in trades], dtype=float)
        r_multiples = np.array([t.get('r_multiple', 0) or 0 for t in trades], dtype=float)
        fees = np.array([t.get('fees_paid', 0) or 0 for t in trades], dtype=float)
        slippages = np.array([t.get('slippage_applied', 0) or 0 for t in trades], dtype=float)
        
        total_trades = len(trades)
        wins = pnls > 0
        losses = pnls < 0
        breakevens = pnls == 0
        
        win_count = int(wins.sum())
        loss_count = int(losses.sum())
        
        # ── Core Metrics ──
        win_rate = win_count / total_trades if total_trades > 0 else 0
        loss_rate = 1 - win_rate
        
        avg_win = float(pnls[wins].mean()) if wins.any() else 0
        avg_loss = float(abs(pnls[losses].mean())) if losses.any() else 0
        
        # Expectancy: E = (WR × AvgWin) − (LR × AvgLoss)
        expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        
        avg_r = float(r_multiples.mean()) if len(r_multiples) > 0 else 0
        
        # Profit factor
        gross_profit = float(pnls[wins].sum()) if wins.any() else 0
        gross_loss = float(abs(pnls[losses].sum())) if losses.any() else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0
        
        # Total PnL
        total_pnl = float(pnls.sum())
        total_return_pct = (total_pnl / initial_capital) * 100 if initial_capital > 0 else 0
        
        # ── Equity Curve ──
        equity = initial_capital + np.cumsum(pnls)
        peak = np.maximum.accumulate(equity)
        drawdowns = (peak - equity) / peak
        max_drawdown_pct = float(drawdowns.max()) * 100 if len(drawdowns) > 0 else 0
        max_drawdown_abs = float((peak - equity).max()) if len(equity) > 0 else 0
        
        # ── Sharpe & Sortino ──
        if len(pnls) > 1 and np.std(pnls) != 0:
            sharpe = float(np.mean(pnls) / np.std(pnls) * np.sqrt(252))
        else:
            sharpe = 0.0
        
        downside_pnls = pnls[pnls < 0]
        if len(downside_pnls) > 1 and np.std(downside_pnls) != 0:
            sortino = float(np.mean(pnls) / np.std(downside_pnls) * np.sqrt(252))
        else:
            sortino = 0.0
        
        # ── Calmar Ratio ──
        calmar = total_return_pct / max_drawdown_pct if max_drawdown_pct > 0 else 0
        
        # ── Ulcer Index ──
        if len(drawdowns) > 0:
            ulcer_index = float(np.sqrt(np.mean(drawdowns ** 2))) * 100
        else:
            ulcer_index = 0.0
        
        # ── Recovery Factor ──
        recovery_factor = total_pnl / max_drawdown_abs if max_drawdown_abs > 0 else 0
        
        # ── Consecutive Losses ──
        max_consec_losses = MetricsCalculator._max_consecutive(pnls < 0)
        max_consec_wins = MetricsCalculator._max_consecutive(pnls > 0)
        
        # ── Risk of Ruin ──
        risk_of_ruin = MetricsCalculator._estimate_risk_of_ruin(win_rate, avg_win, avg_loss, initial_capital)
        
        # ── Average Holding Time ──
        holding_seconds = []
        for t in trades:
            t_open = t.get('timestamp_open')
            t_close = t.get('timestamp_close')
            if t_open and t_close:
                try:
                    from datetime import datetime
                    if isinstance(t_open, str):
                        t_open = datetime.fromisoformat(t_open.replace('Z', '+00:00'))
                    if isinstance(t_close, str):
                        t_close = datetime.fromisoformat(t_close.replace('Z', '+00:00'))
                    diff = (t_close - t_open).total_seconds()
                    if diff > 0:
                        holding_seconds.append(diff)
                except (ValueError, TypeError):
                    pass
        avg_holding_seconds = int(np.mean(holding_seconds)) if holding_seconds else 0
        
        # ── Slippage Impact ──
        total_slippage = float(slippages.sum())
        total_fees_paid = float(fees.sum())
        gross_pnl = total_pnl + total_fees_paid + total_slippage
        net_expectancy = expectancy - (total_fees_paid + total_slippage) / max(total_trades, 1)
        
        # ── Distribution Analysis ──
        r_skewness = float(MetricsCalculator._skewness(r_multiples))
        r_kurtosis = float(MetricsCalculator._kurtosis(r_multiples))
        big_winner_pct = float((r_multiples > 3.0).sum() / total_trades * 100) if total_trades > 0 else 0
        big_winner_dependent = big_winner_pct > 30  # >30% of profit from >3R wins
        
        # ── Stability Score ──
        stability_score = MetricsCalculator._stability_score(r_multiples, rolling_window)
        
        # ── Regime Metrics ──
        regime_metrics = MetricsCalculator._regime_metrics(trades)
        
        # ── Capital Efficiency ──
        position_values = [
            abs(t.get('position_size', 0) or 0) * (t.get('entry_price', 0) or 0)
            for t in trades
        ]
        max_margin = max(position_values) if position_values else 0
        avg_capital_deployed = float(np.mean(position_values)) if position_values else 0
        return_per_max_margin = total_pnl / max_margin if max_margin > 0 else 0
        return_per_avg_capital = total_pnl / avg_capital_deployed if avg_capital_deployed > 0 else 0
        # Capital efficiency = return per unit of time-capital
        avg_holding_hrs = avg_holding_seconds / 3600 if avg_holding_seconds > 0 else 1
        capital_efficiency_ratio = (
            total_pnl / (avg_holding_hrs * avg_capital_deployed)
            if avg_capital_deployed > 0 else 0
        )
        
        return {
            # Core
            'total_trades': total_trades,
            'win_count': win_count,
            'loss_count': loss_count,
            'win_rate': round(win_rate * 100, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'avg_r': round(avg_r, 4),
            'expectancy': round(expectancy, 2),
            'profit_factor': round(profit_factor, 4),
            'total_pnl': round(total_pnl, 2),
            'total_return_pct': round(total_return_pct, 2),
            
            # Equity health
            'max_drawdown_pct': round(max_drawdown_pct, 2),
            'max_drawdown_abs': round(max_drawdown_abs, 2),
            'sharpe_ratio': round(sharpe, 4),
            'sortino_ratio': round(sortino, 4),
            'calmar_ratio': round(calmar, 4),
            'ulcer_index': round(ulcer_index, 4),
            'recovery_factor': round(recovery_factor, 4),
            'max_consecutive_losses': max_consec_losses,
            'max_consecutive_wins': max_consec_wins,
            'risk_of_ruin': round(risk_of_ruin * 100, 2),
            
            # Holding & costs
            'avg_holding_seconds': avg_holding_seconds,
            'total_slippage': round(total_slippage, 4),
            'total_fees': round(total_fees_paid, 4),
            'slippage_impact': round(total_slippage / max(gross_pnl, 1) * 100, 2) if gross_pnl != 0 else 0,
            'net_expectancy': round(net_expectancy, 2),
            
            # Distribution
            'r_skewness': round(r_skewness, 4),
            'r_kurtosis': round(r_kurtosis, 4),
            'big_winner_pct': round(big_winner_pct, 2),
            'big_winner_dependent': big_winner_dependent,
            
            # Stability
            'stability_score': round(stability_score, 4),
            
            # Regime
            'trend_expectancy': round(regime_metrics.get('trending', 0), 2),
            'range_expectancy': round(regime_metrics.get('ranging', 0), 2),
            'high_vol_expectancy': round(regime_metrics.get('high', 0), 2),
            'low_vol_expectancy': round(regime_metrics.get('low', 0), 2),
            
            # Capital efficiency
            'max_margin_used': round(max_margin, 2),
            'avg_capital_deployed': round(avg_capital_deployed, 2),
            'return_per_max_margin': round(return_per_max_margin, 4),
            'return_per_avg_capital': round(return_per_avg_capital, 4),
            'capital_efficiency_ratio': round(capital_efficiency_ratio, 6),
            
            # Flags
            'is_negative_expectancy': expectancy < 0,
            
            # Equity curve for charting
            'equity_curve': [round(e, 2) for e in equity.tolist()],
            'drawdown_curve': [round(d * 100, 2) for d in drawdowns.tolist()],
        }
    
    @staticmethod
    def _max_consecutive(bool_array: np.ndarray) -> int:
        """Max consecutive True values."""
        if len(bool_array) == 0:
            return 0
        max_run = 0
        current_run = 0
        for v in bool_array:
            if v:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        return max_run
    
    @staticmethod
    def _estimate_risk_of_ruin(
        win_rate: float,
        avg_win: float,
        avg_loss: float,
        capital: float,
        ruin_pct: float = 0.5,
    ) -> float:
        """
        Estimate risk of ruin using simplified formula.
        P(ruin) ≈ ((1-edge)/(1+edge))^(capital_units)
        """
        if avg_loss <= 0 or win_rate <= 0:
            return 1.0
        
        edge = (win_rate * avg_win - (1 - win_rate) * avg_loss) / avg_loss
        if edge <= 0:
            return 1.0
        
        if edge >= 1:
            return 0.0
        
        capital_units = capital / avg_loss if avg_loss > 0 else 100
        capital_units = min(capital_units, 1000)
        
        ratio = (1 - edge) / (1 + edge)
        ruin_units = capital * ruin_pct / avg_loss if avg_loss > 0 else 50
        ruin_units = min(ruin_units, 500)
        
        return min(1.0, ratio ** ruin_units)
    
    @staticmethod
    def _skewness(data: np.ndarray) -> float:
        if len(data) < 3:
            return 0.0
        n = len(data)
        mean = np.mean(data)
        std = np.std(data, ddof=1)
        if std == 0:
            return 0.0
        return float((n / ((n - 1) * (n - 2))) * np.sum(((data - mean) / std) ** 3))
    
    @staticmethod
    def _kurtosis(data: np.ndarray) -> float:
        if len(data) < 4:
            return 0.0
        return float(np.mean(((data - np.mean(data)) / max(np.std(data), 1e-10)) ** 4) - 3)
    
    @staticmethod
    def _stability_score(r_multiples: np.ndarray, window: int = 30) -> float:
        """Stability = 1 / StdDev(rolling expectancy)."""
        if len(r_multiples) < window:
            return 0.0
        
        rolling_exp = np.convolve(r_multiples, np.ones(window) / window, mode='valid')
        std_exp = np.std(rolling_exp)
        if std_exp == 0:
            return 100.0  # Perfect stability
        return min(100.0, 1.0 / std_exp)
    
    @staticmethod
    def _regime_metrics(trades: List[Dict]) -> Dict[str, float]:
        """Calculate expectancy per regime."""
        regime_pnls = defaultdict(list)
        for t in trades:
            regime = t.get('regime_at_entry', 'unknown')
            if regime:
                regime_pnls[regime].append(t.get('net_pnl', 0) or 0)
        
        result = {}
        for regime, pnls in regime_pnls.items():
            if pnls:
                result[regime] = float(np.mean(pnls))
        return result
    
    @staticmethod
    def _empty_metrics() -> Dict[str, Any]:
        return {
            'total_trades': 0, 'win_count': 0, 'loss_count': 0,
            'win_rate': 0, 'avg_win': 0, 'avg_loss': 0, 'avg_r': 0,
            'expectancy': 0, 'profit_factor': 0, 'total_pnl': 0,
            'total_return_pct': 0, 'max_drawdown_pct': 0,
            'max_drawdown_abs': 0, 'sharpe_ratio': 0, 'sortino_ratio': 0,
            'calmar_ratio': 0, 'ulcer_index': 0, 'recovery_factor': 0,
            'max_consecutive_losses': 0, 'max_consecutive_wins': 0,
            'risk_of_ruin': 0, 'avg_holding_seconds': 0,
            'total_slippage': 0, 'total_fees': 0, 'slippage_impact': 0,
            'net_expectancy': 0, 'r_skewness': 0, 'r_kurtosis': 0,
            'big_winner_pct': 0, 'big_winner_dependent': False,
            'stability_score': 0, 'trend_expectancy': 0,
            'range_expectancy': 0, 'high_vol_expectancy': 0,
            'low_vol_expectancy': 0,
            'max_margin_used': 0, 'avg_capital_deployed': 0,
            'return_per_max_margin': 0, 'return_per_avg_capital': 0,
            'capital_efficiency_ratio': 0,
            'is_negative_expectancy': False,
            'equity_curve': [], 'drawdown_curve': [],
        }

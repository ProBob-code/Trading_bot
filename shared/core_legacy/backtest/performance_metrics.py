import pandas as pd
import numpy as np
from typing import Dict, Any

class PerformanceMetrics:
    """
    Calculates HFR-level performance metrics.
    """

    @staticmethod
    def calculate_metrics(trade_log: pd.DataFrame, initial_capital: float) -> Dict[str, Any]:
        """
        Calculates a comprehensive set of metrics from a trade log.
        Expected columns: ['pnl', 'pnl_pct', 'pnl_r', 'exit_time']
        """
        if trade_log.empty:
            return {}

        total_pnl = trade_log['pnl'].sum()
        win_rate = (trade_log['pnl'] > 0).mean()
        
        wins = trade_log[trade_log['pnl'] > 0]['pnl']
        losses = trade_log[trade_log['pnl'] <= 0]['pnl']
        
        avg_win = wins.mean() if not wins.empty else 0
        avg_loss = abs(losses.mean()) if not losses.empty else 0
        profit_factor = wins.sum() / abs(losses.sum()) if not losses.empty and losses.sum() != 0 else np.inf
        
        expectancy_r = trade_log['pnl_r'].mean()
        
        # Equity Curve
        equity_curve = initial_capital + trade_log['pnl'].cumsum()
        max_drawdown = (equity_curve.cummax() - equity_curve).max()
        max_drawdown_pct = ((equity_curve.cummax() - equity_curve) / equity_curve.cummax()).max()
        
        # Sharpe, Sortino, Calmar (assuming daily returns or trade-based proxies)
        returns = trade_log['pnl_pct']
        sharpe = (returns.mean() / returns.std() * np.sqrt(252)) if returns.std() != 0 else 0
        
        downside_returns = returns[returns < 0]
        sortino = (returns.mean() / downside_returns.std() * np.sqrt(252)) if not downside_returns.empty and downside_returns.std() != 0 else 0
        
        total_return_pct = total_pnl / initial_capital
        calmar = total_return_pct / max_drawdown_pct if max_drawdown_pct != 0 else np.inf
        
        recovery_factor = total_pnl / max_drawdown if max_drawdown != 0 else np.inf

        return {
            "Total PnL": round(total_pnl, 2),
            "Total Return %": round(total_return_pct * 100, 2),
            "Win Rate": round(win_rate * 100, 2),
            "Avg Win": round(avg_win, 2),
            "Avg Loss": round(avg_loss, 2),
            "Profit Factor": round(profit_factor, 2),
            "Expectancy (R)": round(expectancy_r, 2),
            "Max Drawdown %": round(max_drawdown_pct * 100, 2),
            "Sharpe Ratio": round(sharpe, 2),
            "Sortino Ratio": round(sortino, 2),
            "Calmar Ratio": round(calmar, 2),
            "Recovery Factor": round(recovery_factor, 2)
        }

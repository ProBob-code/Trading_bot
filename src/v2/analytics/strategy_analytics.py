"""
V2 Strategy Analytics Engine
==============================

Fund-grade per-strategy analytics.

Computes 15 institutional metrics:
- total_trades, wins, losses, win_rate
- avg_win, avg_loss, expectancy, profit_factor
- sharpe_ratio, sortino_ratio, calmar_ratio
- recovery_factor, max_drawdown_pct, avg_r_multiple
- equity curve with max drawdown tracking

Called after every closed trade via update_after_trade().
"""

import math
from datetime import datetime
from typing import List, Dict, Optional
from loguru import logger


class StrategyAnalytics:
    """
    Institutional strategy analytics engine.
    
    Computes all metrics from trade history and persists to database.
    """
    
    def __init__(self, db_manager=None):
        """
        Args:
            db_manager: DatabaseManager instance for persistence
        """
        self.db = db_manager
    
    def compute_metrics(self, trades: List[Dict], initial_capital: float = 100000) -> Dict:
        """
        Compute all 15 institutional metrics from closed trades.
        
        Args:
            trades: List of closed trade dicts with 'realized_pnl' or 'net_pnl' key
            initial_capital: Starting capital for equity curve
            
        Returns:
            Dict with all metrics
        """
        if not trades:
            return self._empty_metrics()
        
        pnls = [t.get('net_pnl', t.get('realized_pnl', 0)) for t in trades]
        total = len(pnls)
        
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total if total > 0 else 0
        
        avg_win = sum(wins) / win_count if wins else 0
        avg_loss = abs(sum(losses) / loss_count) if losses else 0
        
        total_pnl = sum(pnls)
        
        # Expectancy = (win_rate × avg_win) − ((1 − win_rate) × avg_loss)
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Profit Factor = gross_wins / gross_losses
        gross_wins = sum(wins)
        gross_losses = abs(sum(losses))
        profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf') if gross_wins > 0 else 0
        
        # Equity curve
        equity_curve = self.compute_equity_curve(pnls, initial_capital)
        max_dd = self.compute_max_drawdown(equity_curve)
        
        # Returns for ratio calculations
        returns = self._compute_returns(equity_curve)
        
        # Sharpe Ratio (annualized, assuming daily trades)
        sharpe = self._compute_sharpe(returns)
        
        # Sortino Ratio (downside deviation only)
        sortino = self._compute_sortino(returns)
        
        # Calmar Ratio = annualized_return / max_drawdown
        calmar = self._compute_calmar(equity_curve, max_dd)
        
        # Recovery Factor = net_profit / max_drawdown_absolute
        max_dd_absolute = self._compute_max_drawdown_absolute(equity_curve)
        recovery_factor = total_pnl / max_dd_absolute if max_dd_absolute > 0 else 0
        
        # Average R-Multiple (normalized by avg_loss)
        avg_r = self._compute_avg_r_multiple(pnls, avg_loss)
        
        return {
            'total_trades': total,
            'wins': win_count,
            'losses': loss_count,
            'win_rate': round(win_rate, 4),
            'total_pnl': round(total_pnl, 2),
            'avg_win': round(avg_win, 2),
            'avg_loss': round(avg_loss, 2),
            'expectancy': round(expectancy, 4),
            'profit_factor': round(profit_factor, 4),
            'sharpe_ratio': round(sharpe, 4),
            'sortino_ratio': round(sortino, 4),
            'calmar_ratio': round(calmar, 4),
            'recovery_factor': round(recovery_factor, 4),
            'max_drawdown_pct': round(max_dd, 4),
            'avg_r_multiple': round(avg_r, 4),
        }
    
    def compute_equity_curve(self, pnls: List[float], initial_capital: float = 100000) -> List[float]:
        """
        Build cumulative equity curve from P&L series.
        
        Args:
            pnls: List of trade P&Ls in chronological order
            initial_capital: Starting equity
            
        Returns:
            List of equity values (length = len(pnls) + 1, starts with initial_capital)
        """
        curve = [initial_capital]
        equity = initial_capital
        for pnl in pnls:
            equity += pnl
            curve.append(equity)
        return curve
    
    def compute_max_drawdown(self, equity_curve: List[float]) -> float:
        """
        Compute maximum drawdown as percentage.
        
        Peak-to-trough maximum decline.
        
        Args:
            equity_curve: List of equity values
            
        Returns:
            Max drawdown as percentage (e.g., 15.5 = 15.5%)
        """
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            
            if peak > 0:
                dd = (peak - value) / peak
                if dd > max_dd:
                    max_dd = dd
        
        return max_dd * 100
    
    def _compute_max_drawdown_absolute(self, equity_curve: List[float]) -> float:
        """Compute max drawdown in absolute dollar terms."""
        if not equity_curve or len(equity_curve) < 2:
            return 0.0
        
        peak = equity_curve[0]
        max_dd = 0.0
        
        for value in equity_curve:
            if value > peak:
                peak = value
            dd = peak - value
            if dd > max_dd:
                max_dd = dd
        
        return max_dd
    
    def _compute_returns(self, equity_curve: List[float]) -> List[float]:
        """Compute period-over-period returns."""
        if len(equity_curve) < 2:
            return []
        
        returns = []
        for i in range(1, len(equity_curve)):
            if equity_curve[i - 1] > 0:
                ret = (equity_curve[i] - equity_curve[i - 1]) / equity_curve[i - 1]
                returns.append(ret)
        return returns
    
    def _compute_sharpe(self, returns: List[float], risk_free: float = 0.0, periods: int = 252) -> float:
        """Annualized Sharpe Ratio."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns) - risk_free / periods
        
        # Standard deviation
        variance = sum((r - mean_return) ** 2 for r in returns) / (len(returns) - 1)
        std = math.sqrt(variance) if variance > 0 else 0
        
        if std == 0:
            return 0.0
        
        return (mean_return / std) * math.sqrt(periods)
    
    def _compute_sortino(self, returns: List[float], risk_free: float = 0.0, periods: int = 252) -> float:
        """Annualized Sortino Ratio (downside deviation only)."""
        if not returns or len(returns) < 2:
            return 0.0
        
        mean_return = sum(returns) / len(returns) - risk_free / periods
        
        # Downside deviation (only negative returns)
        downside = [r for r in returns if r < 0]
        if not downside:
            return float('inf') if mean_return > 0 else 0.0
        
        downside_var = sum(r ** 2 for r in downside) / len(downside)
        downside_std = math.sqrt(downside_var) if downside_var > 0 else 0
        
        if downside_std == 0:
            return 0.0
        
        return (mean_return / downside_std) * math.sqrt(periods)
    
    def _compute_calmar(self, equity_curve: List[float], max_dd_pct: float) -> float:
        """Calmar Ratio = annualized return / max drawdown %."""
        if max_dd_pct <= 0 or len(equity_curve) < 2:
            return 0.0
        
        total_return = (equity_curve[-1] - equity_curve[0]) / equity_curve[0]
        # Assume ~252 trading days per year, normalize by number of data points
        n_periods = len(equity_curve) - 1
        annualized = total_return * (252 / max(n_periods, 1))
        
        return (annualized * 100) / max_dd_pct
    
    def _compute_avg_r_multiple(self, pnls: List[float], avg_loss: float) -> float:
        """Average R-Multiple (each trade's P&L normalized by avg risk unit)."""
        if avg_loss <= 0 or not pnls:
            return 0.0
        
        r_multiples = [p / avg_loss for p in pnls]
        return sum(r_multiples) / len(r_multiples)
    
    def _empty_metrics(self) -> Dict:
        """Return zeroed-out metrics dict."""
        return {
            'total_trades': 0, 'wins': 0, 'losses': 0,
            'win_rate': 0, 'total_pnl': 0,
            'avg_win': 0, 'avg_loss': 0,
            'expectancy': 0, 'profit_factor': 0,
            'sharpe_ratio': 0, 'sortino_ratio': 0,
            'calmar_ratio': 0, 'recovery_factor': 0,
            'max_drawdown_pct': 0, 'avg_r_multiple': 0,
        }
    
    def update_after_trade(
        self,
        user_id: int,
        strategy: str,
        pnl: float,
        account_value: float,
        initial_capital: float = 100000
    ):
        """
        Update strategy metrics and equity curve after a closed trade.
        
        Fetches all closed trades for this user+strategy from DB,
        recomputes metrics, and persists.
        
        Args:
            user_id: User ID
            strategy: Strategy name
            pnl: P&L from the just-closed trade
            account_value: Current account equity after trade
            initial_capital: Starting capital for equity curve
        """
        if not self.db:
            logger.warning("[V2-ANALYTICS] No DB manager — metrics not persisted")
            return
        
        try:
            # Get all closed trades for this strategy
            trades = self.db.v2_get_user_trades(
                user_id=user_id,
                strategy=strategy,
                trade_type='CLOSE'
            )
            
            # Compute metrics
            metrics = self.compute_metrics(trades, initial_capital)
            
            # Upsert to DB
            self.db.v2_upsert_strategy_metrics(user_id, strategy, metrics)
            
            # Append equity point
            self.db.v2_append_equity_point(user_id, strategy, account_value)
            
            logger.info(
                f"[V2-ANALYTICS] Updated {strategy} metrics: "
                f"trades={metrics['total_trades']}, exp={metrics['expectancy']:.4f}, "
                f"dd={metrics['max_drawdown_pct']:.2f}%"
            )
            
        except Exception as e:
            logger.error(f"[V2-ANALYTICS] Failed to update metrics: {e}")

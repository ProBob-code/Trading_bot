"""
Backtesting Engine
==================

Tests trading strategies on historical data before live deployment.
Calculates performance metrics: returns, Sharpe ratio, max drawdown, win rate.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from loguru import logger


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    
    # Trade metrics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    
    # Returns
    total_return: float = 0.0
    total_return_pct: float = 0.0
    avg_trade_return: float = 0.0
    
    # Risk metrics
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    
    # Trade list
    trades: List[Dict] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            'strategy': self.strategy_name,
            'symbol': self.symbol,
            'period': f"{self.start_date} to {self.end_date}",
            'total_trades': self.total_trades,
            'win_rate': f"{self.win_rate:.1f}%",
            'total_return': f"${self.total_return:.2f}",
            'total_return_pct': f"{self.total_return_pct:.2f}%",
            'max_drawdown': f"{self.max_drawdown_pct:.2f}%",
            'sharpe_ratio': f"{self.sharpe_ratio:.2f}",
            'trades': self.trades[-10:]  # Last 10 trades
        }


class BacktestEngine:
    """
    Backtesting engine for trading strategies.
    
    Usage:
        engine = BacktestEngine(initial_capital=100000)
        result = engine.run(strategy, data, params)
    """
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = []
    
    def run(
        self,
        strategy_name: str,
        data: pd.DataFrame,
        strategy_func,
        stop_loss_pct: float = 5.0,
        take_profit_pct: float = 10.0,
        position_size_pct: float = 10.0
    ) -> BacktestResult:
        """
        Run backtest on historical data.
        
        Args:
            strategy_name: Name of strategy being tested
            data: DataFrame with OHLCV data
            strategy_func: Function that returns signal ('BUY', 'SELL', 'HOLD')
            stop_loss_pct: Stop loss percentage
            take_profit_pct: Take profit percentage
            position_size_pct: Position size as % of capital
            
        Returns:
            BacktestResult with all metrics
        """
        self.capital = self.initial_capital
        self.position = 0
        self.entry_price = 0
        self.trades = []
        self.equity_curve = [self.initial_capital]
        
        if data.empty or len(data) < 50:
            logger.warning("Not enough data for backtest")
            return BacktestResult(
                strategy_name=strategy_name,
                symbol=data.get('symbol', 'UNKNOWN') if hasattr(data, 'get') else 'UNKNOWN',
                start_date='',
                end_date=''
            )
        
        symbol = 'UNKNOWN'
        start_date = str(data.index[0]) if hasattr(data.index[0], '__str__') else ''
        end_date = str(data.index[-1]) if hasattr(data.index[-1], '__str__') else ''
        
        # Iterate through data
        for i in range(50, len(data)):
            current_data = data.iloc[:i+1]
            current_price = data.iloc[i]['close']
            
            # Get signal from strategy
            try:
                signal = strategy_func(current_data)
            except Exception as e:
                signal = 'HOLD'
            
            # Check stop loss / take profit if in position
            if self.position != 0:
                pnl_pct = ((current_price / self.entry_price) - 1) * 100
                if self.position > 0:  # Long position
                    if pnl_pct <= -stop_loss_pct:
                        self._close_position(current_price, 'STOP_LOSS', data.index[i])
                    elif pnl_pct >= take_profit_pct:
                        self._close_position(current_price, 'TAKE_PROFIT', data.index[i])
                else:  # Short position
                    pnl_pct = -pnl_pct
                    if pnl_pct <= -stop_loss_pct:
                        self._close_position(current_price, 'STOP_LOSS', data.index[i])
                    elif pnl_pct >= take_profit_pct:
                        self._close_position(current_price, 'TAKE_PROFIT', data.index[i])
            
            # Process signal
            if signal == 'BUY' and self.position <= 0:
                if self.position < 0:
                    self._close_position(current_price, 'SIGNAL', data.index[i])
                self._open_position(current_price, 1, position_size_pct, data.index[i])
                
            elif signal == 'SELL' and self.position >= 0:
                if self.position > 0:
                    self._close_position(current_price, 'SIGNAL', data.index[i])
                self._open_position(current_price, -1, position_size_pct, data.index[i])
            
            # Update equity curve
            unrealized_pnl = 0
            if self.position != 0:
                unrealized_pnl = (current_price - self.entry_price) * self.position
            self.equity_curve.append(self.capital + unrealized_pnl)
        
        # Close any open position at end
        if self.position != 0:
            self._close_position(data.iloc[-1]['close'], 'END', data.index[-1])
        
        # Calculate metrics
        result = self._calculate_metrics(strategy_name, symbol, start_date, end_date)
        
        logger.info(f"📊 Backtest complete: {strategy_name} | {result.total_trades} trades | {result.win_rate:.1f}% win rate")
        
        return result
    
    def _open_position(self, price: float, direction: int, size_pct: float, timestamp):
        """Open a position."""
        position_value = self.capital * (size_pct / 100)
        quantity = position_value / price * direction
        
        self.position = quantity
        self.entry_price = price
    
    def _close_position(self, price: float, reason: str, timestamp):
        """Close current position."""
        if self.position == 0:
            return
        
        pnl = (price - self.entry_price) * self.position
        self.capital += pnl
        
        self.trades.append({
            'timestamp': str(timestamp),
            'entry': self.entry_price,
            'exit': price,
            'quantity': abs(self.position),
            'side': 'LONG' if self.position > 0 else 'SHORT',
            'pnl': pnl,
            'reason': reason
        })
        
        self.position = 0
        self.entry_price = 0
    
    def _calculate_metrics(self, strategy: str, symbol: str, start: str, end: str) -> BacktestResult:
        """Calculate backtest metrics."""
        result = BacktestResult(
            strategy_name=strategy,
            symbol=symbol,
            start_date=start,
            end_date=end,
            trades=self.trades,
            equity_curve=self.equity_curve
        )
        
        if not self.trades:
            return result
        
        # Trade metrics
        pnls = [t['pnl'] for t in self.trades]
        result.total_trades = len(self.trades)
        result.winning_trades = len([p for p in pnls if p > 0])
        result.losing_trades = len([p for p in pnls if p <= 0])
        result.win_rate = (result.winning_trades / result.total_trades) * 100 if result.total_trades > 0 else 0
        
        # Returns
        result.total_return = self.capital - self.initial_capital
        result.total_return_pct = ((self.capital / self.initial_capital) - 1) * 100
        result.avg_trade_return = result.total_return / result.total_trades if result.total_trades > 0 else 0
        
        # Max drawdown
        peak = self.initial_capital
        max_dd = 0
        for equity in self.equity_curve:
            if equity > peak:
                peak = equity
            dd = (peak - equity) / peak * 100
            if dd > max_dd:
                max_dd = dd
        result.max_drawdown_pct = max_dd
        
        # Sharpe ratio (simplified)
        if len(pnls) > 1:
            returns = np.array(pnls) / self.initial_capital * 100
            result.sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        
        return result


# Singleton instance
_backtest_engine = None

def get_backtest_engine(initial_capital: float = 100000) -> BacktestEngine:
    """Get or create backtest engine instance."""
    global _backtest_engine
    if _backtest_engine is None:
        _backtest_engine = BacktestEngine(initial_capital)
    return _backtest_engine

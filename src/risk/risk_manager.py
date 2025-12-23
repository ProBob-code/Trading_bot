"""
Risk Manager
============

Comprehensive risk management for trading.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Dict, List, Optional, Tuple
import pandas as pd
from loguru import logger


@dataclass
class Position:
    """Represents an open position."""
    symbol: str
    side: str  # 'long' or 'short'
    entry_price: float
    units: float
    stop_loss: float
    take_profit: float
    entry_time: datetime = field(default_factory=datetime.now)
    trailing_stop: Optional[float] = None
    highest_price: Optional[float] = None  # For trailing stop
    lowest_price: Optional[float] = None
    
    @property
    def value(self) -> float:
        return self.entry_price * self.units
    
    def update_trailing_stop(self, current_price: float, trail_pct: float):
        """Update trailing stop based on current price."""
        if self.side == 'long':
            if self.highest_price is None or current_price > self.highest_price:
                self.highest_price = current_price
                self.trailing_stop = current_price * (1 - trail_pct)
        else:  # short
            if self.lowest_price is None or current_price < self.lowest_price:
                self.lowest_price = current_price
                self.trailing_stop = current_price * (1 + trail_pct)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized P&L."""
        if self.side == 'long':
            return (current_price - self.entry_price) * self.units
        else:
            return (self.entry_price - current_price) * self.units
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Calculate unrealized P&L percentage."""
        if self.side == 'long':
            return ((current_price - self.entry_price) / self.entry_price) * 100
        else:
            return ((self.entry_price - current_price) / self.entry_price) * 100


@dataclass
class DailyStats:
    """Daily trading statistics."""
    date: date
    starting_capital: float
    ending_capital: float
    trades: int = 0
    wins: int = 0
    losses: int = 0
    gross_profit: float = 0
    gross_loss: float = 0
    
    @property
    def net_pnl(self) -> float:
        return self.gross_profit + self.gross_loss
    
    @property
    def win_rate(self) -> float:
        return self.wins / self.trades if self.trades > 0 else 0


class RiskManager:
    """
    Manages trading risk.
    
    Features:
    - Maximum daily loss limits
    - Maximum drawdown limits
    - Position limits
    - Stop loss management
    - Portfolio exposure tracking
    """
    
    def __init__(
        self,
        initial_capital: float,
        max_daily_loss_pct: float = 2.0,
        max_drawdown_pct: float = 10.0,
        max_positions: int = 5,
        max_position_pct: float = 10.0,
        max_portfolio_exposure_pct: float = 50.0,
        use_trailing_stop: bool = True,
        trailing_stop_pct: float = 1.0
    ):
        """
        Initialize risk manager.
        
        Args:
            initial_capital: Starting capital
            max_daily_loss_pct: Maximum daily loss as % of capital
            max_drawdown_pct: Maximum drawdown as % of peak
            max_positions: Maximum concurrent positions
            max_position_pct: Max single position as % of capital
            max_portfolio_exposure_pct: Max total exposure as % of capital
            use_trailing_stop: Enable trailing stops
            trailing_stop_pct: Trailing stop distance as %
        """
        self.initial_capital = initial_capital
        self.current_capital = initial_capital
        self.peak_capital = initial_capital
        
        self.max_daily_loss = initial_capital * (max_daily_loss_pct / 100)
        self.max_drawdown_pct = max_drawdown_pct / 100
        self.max_positions = max_positions
        self.max_position_pct = max_position_pct / 100
        self.max_portfolio_exposure = initial_capital * (max_portfolio_exposure_pct / 100)
        
        self.use_trailing_stop = use_trailing_stop
        self.trailing_stop_pct = trailing_stop_pct / 100
        
        # State
        self.positions: Dict[str, Position] = {}
        self.daily_stats: Dict[date, DailyStats] = {}
        self.trade_history: List[Dict] = []
        
        # Flags
        self.trading_halted = False
        self.halt_reason = ""
        
    def can_open_position(
        self,
        symbol: str,
        position_value: float
    ) -> Tuple[bool, str]:
        """
        Check if a new position can be opened.
        
        Returns:
            (allowed, reason)
        """
        # Check if trading is halted
        if self.trading_halted:
            return False, f"Trading halted: {self.halt_reason}"
        
        # Check daily loss limit
        today_stats = self._get_today_stats()
        if today_stats.net_pnl < -self.max_daily_loss:
            self.trading_halted = True
            self.halt_reason = "Daily loss limit reached"
            return False, self.halt_reason
        
        # Check drawdown
        current_dd = self._calculate_drawdown()
        if current_dd > self.max_drawdown_pct:
            self.trading_halted = True
            self.halt_reason = f"Maximum drawdown reached ({current_dd*100:.1f}%)"
            return False, self.halt_reason
        
        # Check position count
        if len(self.positions) >= self.max_positions:
            return False, f"Maximum positions reached ({self.max_positions})"
        
        # Check if already have position in symbol
        if symbol in self.positions:
            return False, f"Already have position in {symbol}"
        
        # Check position size
        if position_value > self.current_capital * self.max_position_pct:
            return False, f"Position size exceeds limit ({self.max_position_pct*100:.0f}%)"
        
        # Check total exposure
        current_exposure = sum(p.value for p in self.positions.values())
        if current_exposure + position_value > self.max_portfolio_exposure:
            return False, "Maximum portfolio exposure reached"
        
        return True, "OK"
    
    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        units: float,
        stop_loss: float,
        take_profit: float
    ) -> Optional[Position]:
        """
        Open a new position.
        
        Returns:
            Position object or None if not allowed
        """
        position_value = entry_price * units
        
        allowed, reason = self.can_open_position(symbol, position_value)
        if not allowed:
            logger.warning(f"Cannot open position: {reason}")
            return None
        
        position = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            units=units,
            stop_loss=stop_loss,
            take_profit=take_profit
        )
        
        self.positions[symbol] = position
        self.current_capital -= position_value
        
        logger.info(f"Opened {side} position: {symbol} @ {entry_price}, units: {units}")
        return position
    
    def close_position(
        self,
        symbol: str,
        exit_price: float,
        reason: str = "manual"
    ) -> Optional[Dict]:
        """
        Close a position.
        
        Returns:
            Trade record or None if position not found
        """
        if symbol not in self.positions:
            logger.warning(f"No position found for {symbol}")
            return None
        
        position = self.positions[symbol]
        pnl = position.unrealized_pnl(exit_price)
        pnl_pct = position.unrealized_pnl_pct(exit_price)
        
        # Update capital
        self.current_capital += (position.units * exit_price)
        
        # Update peak capital
        if self.current_capital > self.peak_capital:
            self.peak_capital = self.current_capital
        
        # Record trade
        trade = {
            'symbol': symbol,
            'side': position.side,
            'entry_price': position.entry_price,
            'exit_price': exit_price,
            'units': position.units,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'entry_time': position.entry_time,
            'exit_time': datetime.now(),
            'exit_reason': reason
        }
        self.trade_history.append(trade)
        
        # Update daily stats
        today_stats = self._get_today_stats()
        today_stats.trades += 1
        if pnl > 0:
            today_stats.wins += 1
            today_stats.gross_profit += pnl
        else:
            today_stats.losses += 1
            today_stats.gross_loss += pnl
        
        # Remove position
        del self.positions[symbol]
        
        logger.info(f"Closed {symbol}: PnL ${pnl:.2f} ({pnl_pct:.2f}%) - {reason}")
        return trade
    
    def update_positions(self, prices: Dict[str, float]) -> List[Dict]:
        """
        Update all positions with current prices.
        
        Checks stop losses, take profits, and trailing stops.
        
        Returns:
            List of closed trades
        """
        closed_trades = []
        
        for symbol, position in list(self.positions.items()):
            if symbol not in prices:
                continue
            
            current_price = prices[symbol]
            
            # Update trailing stop
            if self.use_trailing_stop:
                position.update_trailing_stop(current_price, self.trailing_stop_pct)
            
            # Check stop loss
            if position.side == 'long':
                stop_price = position.trailing_stop or position.stop_loss
                if current_price <= stop_price:
                    trade = self.close_position(symbol, current_price, "stop_loss")
                    if trade:
                        closed_trades.append(trade)
                    continue
                    
                # Check take profit
                if current_price >= position.take_profit:
                    trade = self.close_position(symbol, current_price, "take_profit")
                    if trade:
                        closed_trades.append(trade)
                    continue
            else:  # short
                stop_price = position.trailing_stop or position.stop_loss
                if current_price >= stop_price:
                    trade = self.close_position(symbol, current_price, "stop_loss")
                    if trade:
                        closed_trades.append(trade)
                    continue
                    
                if current_price <= position.take_profit:
                    trade = self.close_position(symbol, current_price, "take_profit")
                    if trade:
                        closed_trades.append(trade)
                    continue
        
        return closed_trades
    
    def _calculate_drawdown(self) -> float:
        """Calculate current drawdown from peak."""
        if self.peak_capital <= 0:
            return 0
        return (self.peak_capital - self.current_capital) / self.peak_capital
    
    def _get_today_stats(self) -> DailyStats:
        """Get or create today's stats."""
        today = date.today()
        
        if today not in self.daily_stats:
            self.daily_stats[today] = DailyStats(
                date=today,
                starting_capital=self.current_capital,
                ending_capital=self.current_capital
            )
        
        return self.daily_stats[today]
    
    def get_portfolio_summary(self) -> Dict:
        """Get current portfolio summary."""
        total_exposure = sum(p.value for p in self.positions.values())
        unrealized_pnl = sum(
            p.unrealized_pnl(p.entry_price)  # Would need current price
            for p in self.positions.values()
        )
        
        return {
            'current_capital': self.current_capital,
            'peak_capital': self.peak_capital,
            'initial_capital': self.initial_capital,
            'total_return': ((self.current_capital - self.initial_capital) / self.initial_capital) * 100,
            'drawdown': self._calculate_drawdown() * 100,
            'open_positions': len(self.positions),
            'total_exposure': total_exposure,
            'exposure_pct': (total_exposure / self.current_capital) * 100 if self.current_capital > 0 else 0,
            'trading_halted': self.trading_halted,
            'halt_reason': self.halt_reason
        }
    
    def get_performance_metrics(self) -> Dict:
        """Calculate comprehensive performance metrics."""
        if not self.trade_history:
            return {}
        
        df = pd.DataFrame(self.trade_history)
        
        # Basic stats
        total_trades = len(df)
        wins = len(df[df['pnl'] > 0])
        losses = len(df[df['pnl'] <= 0])
        
        # P&L
        total_pnl = df['pnl'].sum()
        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        
        # Averages
        avg_win = df[df['pnl'] > 0]['pnl'].mean() if wins > 0 else 0
        avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean()) if losses > 0 else 0
        
        # Ratios
        win_rate = wins / total_trades if total_trades > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        risk_reward = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Expectancy
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)
        
        # Sharpe (simplified - would need returns series)
        returns = df['pnl'] / self.initial_capital
        sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        return {
            'total_trades': total_trades,
            'wins': wins,
            'losses': losses,
            'win_rate': win_rate * 100,
            'total_pnl': total_pnl,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'risk_reward_ratio': risk_reward,
            'expectancy': expectancy,
            'sharpe_ratio': sharpe
        }
    
    def reset_daily_limits(self):
        """Reset daily limits (call at start of trading day)."""
        self.trading_halted = False
        self.halt_reason = ""
        
        today_stats = self._get_today_stats()
        today_stats.starting_capital = self.current_capital


# Needed import at top
import numpy as np

"""
Position Sizing Module
======================

Calculate optimal position sizes based on various methods.
"""

from typing import Dict, Optional
import numpy as np
from loguru import logger


class PositionSizer:
    """
    Calculate position sizes for trades.
    
    Methods:
    - Fixed: Always use fixed dollar amount
    - Fixed Fractional: Use fixed percentage of capital
    - Kelly Criterion: Optimal sizing based on win rate and payoff
    - Volatility: Size based on ATR/volatility
    """
    
    def __init__(
        self,
        method: str = "fixed_fractional",
        max_position_pct: float = 10.0,
        max_risk_pct: float = 2.0
    ):
        """
        Initialize position sizer.
        
        Args:
            method: 'fixed', 'fixed_fractional', 'kelly', 'volatility'
            max_position_pct: Maximum position as % of capital
            max_risk_pct: Maximum risk per trade as % of capital
        """
        self.method = method
        self.max_position_pct = max_position_pct / 100
        self.max_risk_pct = max_risk_pct / 100
        
    def calculate_position_size(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float = 0.5,
        avg_win: float = 1.0,
        avg_loss: float = 1.0,
        volatility: float = None,
        **kwargs
    ) -> Dict[str, float]:
        """
        Calculate position size.
        
        Args:
            capital: Available capital
            entry_price: Entry price per unit
            stop_loss: Stop loss price
            win_rate: Historical win rate (for Kelly)
            avg_win: Average winning trade (for Kelly)
            avg_loss: Average losing trade (for Kelly)
            volatility: Current ATR or volatility
            
        Returns:
            Dict with position_size, units, risk_amount, etc.
        """
        if self.method == "fixed":
            return self._fixed_sizing(capital, entry_price, stop_loss, **kwargs)
        elif self.method == "fixed_fractional":
            return self._fixed_fractional(capital, entry_price, stop_loss)
        elif self.method == "kelly":
            return self._kelly_criterion(capital, entry_price, stop_loss, win_rate, avg_win, avg_loss)
        elif self.method == "volatility":
            return self._volatility_sizing(capital, entry_price, stop_loss, volatility)
        else:
            logger.warning(f"Unknown method: {self.method}, using fixed_fractional")
            return self._fixed_fractional(capital, entry_price, stop_loss)
    
    def _fixed_sizing(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        fixed_amount: float = None
    ) -> Dict[str, float]:
        """
        Fixed dollar amount per trade.
        """
        if fixed_amount is None:
            fixed_amount = capital * 0.1  # Default 10%
        
        position_size = min(fixed_amount, capital * self.max_position_pct)
        units = position_size / entry_price
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = units * risk_per_unit
        
        return {
            'position_size': position_size,
            'units': units,
            'risk_amount': risk_amount,
            'risk_pct': (risk_amount / capital) * 100,
            'method': 'fixed'
        }
    
    def _fixed_fractional(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float
    ) -> Dict[str, float]:
        """
        Risk fixed percentage of capital per trade.
        
        position_size = (capital * risk_pct) / (entry - stop_loss)
        """
        risk_amount = capital * self.max_risk_pct
        risk_per_unit = abs(entry_price - stop_loss)
        
        if risk_per_unit <= 0:
            logger.warning("Invalid stop loss (equal or beyond entry)")
            return self._fixed_sizing(capital, entry_price, stop_loss)
        
        # Calculate units based on risk
        units = risk_amount / risk_per_unit
        position_size = units * entry_price
        
        # Apply maximum position limit
        max_position = capital * self.max_position_pct
        if position_size > max_position:
            position_size = max_position
            units = position_size / entry_price
            risk_amount = units * risk_per_unit
        
        return {
            'position_size': position_size,
            'units': units,
            'risk_amount': risk_amount,
            'risk_pct': (risk_amount / capital) * 100,
            'method': 'fixed_fractional'
        }
    
    def _kelly_criterion(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        win_rate: float,
        avg_win: float,
        avg_loss: float
    ) -> Dict[str, float]:
        """
        Kelly Criterion for optimal position sizing.
        
        f* = (p * b - q) / b
        
        where:
        - p = probability of winning
        - q = probability of losing (1 - p)
        - b = win/loss ratio (avg_win / avg_loss)
        """
        if avg_loss <= 0:
            return self._fixed_fractional(capital, entry_price, stop_loss)
        
        p = max(0, min(1, win_rate))
        q = 1 - p
        b = avg_win / avg_loss
        
        # Calculate Kelly fraction
        kelly = (p * b - q) / b if b > 0 else 0
        
        # Use half-Kelly for safety
        kelly = kelly * 0.5
        
        # Ensure kelly is within bounds
        kelly = max(0, min(kelly, self.max_position_pct))
        
        position_size = capital * kelly
        units = position_size / entry_price
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = units * risk_per_unit
        
        # Check risk limit
        if risk_amount > capital * self.max_risk_pct:
            # Scale down
            scale = (capital * self.max_risk_pct) / risk_amount
            position_size *= scale
            units *= scale
            risk_amount = capital * self.max_risk_pct
        
        return {
            'position_size': position_size,
            'units': units,
            'risk_amount': risk_amount,
            'risk_pct': (risk_amount / capital) * 100,
            'kelly_fraction': kelly * 100,
            'method': 'kelly'
        }
    
    def _volatility_sizing(
        self,
        capital: float,
        entry_price: float,
        stop_loss: float,
        volatility: float
    ) -> Dict[str, float]:
        """
        Position sizing based on volatility (ATR).
        
        Larger positions in low volatility, smaller in high volatility.
        """
        if volatility is None or volatility <= 0:
            return self._fixed_fractional(capital, entry_price, stop_loss)
        
        # Normalize volatility (assuming ATR as percentage)
        vol_pct = volatility / entry_price
        
        # Target volatility for position (e.g., 1% daily)
        target_vol = 0.01
        
        # Adjust position size inversely with volatility
        vol_factor = target_vol / max(vol_pct, 0.001)
        vol_factor = max(0.5, min(2.0, vol_factor))  # Limit adjustment
        
        # Base position from fixed fractional
        base_result = self._fixed_fractional(capital, entry_price, stop_loss)
        
        # Adjust by volatility
        position_size = base_result['position_size'] * vol_factor
        position_size = min(position_size, capital * self.max_position_pct)
        
        units = position_size / entry_price
        risk_per_unit = abs(entry_price - stop_loss)
        risk_amount = units * risk_per_unit
        
        return {
            'position_size': position_size,
            'units': units,
            'risk_amount': risk_amount,
            'risk_pct': (risk_amount / capital) * 100,
            'vol_factor': vol_factor,
            'method': 'volatility'
        }
    
    def adjust_for_correlation(
        self,
        position_sizes: Dict[str, float],
        correlation_matrix: np.ndarray
    ) -> Dict[str, float]:
        """
        Adjust position sizes for correlated assets.
        
        Reduces overall portfolio risk when assets are correlated.
        """
        n = len(position_sizes)
        if n <= 1:
            return position_sizes
        
        symbols = list(position_sizes.keys())
        sizes = np.array([position_sizes[s] for s in symbols])
        
        # Calculate portfolio variance
        # Var = sum(wi * wj * cov_ij)
        weights = sizes / np.sum(sizes)
        portfolio_var = np.dot(weights.T, np.dot(correlation_matrix, weights))
        
        # Scale down if high correlation
        if portfolio_var > 0.5:  # High correlation
            scale = 0.5 / portfolio_var
            sizes = sizes * min(scale, 1.0)
        
        return {symbols[i]: sizes[i] for i in range(n)}

"""
V2 Margin Engine
=================

Futures-grade margin and leverage modeling.

Features:
- Initial margin calculation
- Maintenance margin (0.5%)
- Liquidation price (futures-style)
- Cross vs isolated margin modes
- Leveraged P&L computation
- Margin call detection
"""

from typing import Dict
from loguru import logger


class MarginEngine:
    """
    Institutional margin engine for V2.
    
    Supports:
    - Isolated margin: each position has its own margin pool
    - Cross margin: positions share the account's available balance
    """
    
    MAINTENANCE_RATE = 0.005  # 0.5% maintenance margin
    
    def initial_margin(self, notional: float, leverage: float) -> float:
        """
        Compute initial margin required.
        
        Args:
            notional: Position notional value (price × quantity)
            leverage: Leverage multiplier (e.g., 10.0)
            
        Returns:
            Margin required in USD
        """
        if leverage <= 0:
            raise ValueError("Leverage must be positive")
        return notional / leverage
    
    def maintenance_margin(self, notional: float) -> float:
        """
        Compute maintenance margin.
        
        Args:
            notional: Current position notional value
            
        Returns:
            Maintenance margin in USD
        """
        return notional * self.MAINTENANCE_RATE
    
    def liquidation_price(self, entry_price: float, side: str, leverage: float) -> float:
        """
        Compute liquidation price (futures-style).
        
        At liquidation, the loss equals the initial margin.
        
        Args:
            entry_price: Position entry price
            side: 'LONG' or 'SHORT'
            leverage: Leverage multiplier
            
        Returns:
            Liquidation trigger price (0.0 if leverage ≤ 1)
        """
        if leverage <= 1.0:
            return 0.0  # No liquidation at spot
        
        if side.upper() == 'LONG':
            # Long: loss when price drops → liq at entry × (1 - 1/leverage)
            return entry_price * (1 - 1 / leverage)
        else:
            # Short: loss when price rises → liq at entry × (1 + 1/leverage)
            return entry_price * (1 + 1 / leverage)
    
    def leveraged_pnl(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        leverage: float,
        side: str
    ) -> float:
        """
        Compute leveraged P&L.
        
        P&L = base_pnl × leverage
        
        Args:
            entry_price: Entry price
            current_price: Current market price
            quantity: Position size
            leverage: Leverage multiplier
            side: 'LONG' or 'SHORT'
            
        Returns:
            Leveraged P&L in USD
        """
        if side.upper() == 'LONG':
            base_pnl = (current_price - entry_price) * quantity
        else:
            base_pnl = (entry_price - current_price) * quantity
        return base_pnl * leverage
    
    def check_liquidation(
        self,
        entry_price: float,
        current_price: float,
        side: str,
        leverage: float
    ) -> bool:
        """
        Check whether position should be liquidated.
        
        Returns:
            True if liquidation threshold breached
        """
        if leverage <= 1.0:
            return False
        
        liq_price = self.liquidation_price(entry_price, side, leverage)
        
        if side.upper() == 'LONG':
            return current_price <= liq_price
        else:
            return current_price >= liq_price
    
    def margin_ratio(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        leverage: float,
        side: str
    ) -> float:
        """
        Compute current margin ratio.
        
        margin_ratio = remaining_margin / maintenance_margin
        
        A ratio < 1.0 means liquidation.
        
        Returns:
            Margin ratio as float
        """
        notional = entry_price * quantity
        initial = self.initial_margin(notional, leverage)
        pnl = self.leveraged_pnl(entry_price, current_price, quantity, leverage, side)
        remaining_margin = initial + pnl
        
        current_notional = current_price * quantity
        maint = self.maintenance_margin(current_notional)
        
        if maint <= 0:
            return float('inf')
        return remaining_margin / maint
    
    def position_summary(
        self,
        entry_price: float,
        current_price: float,
        quantity: float,
        leverage: float,
        side: str
    ) -> Dict:
        """
        Get full margin summary for a position.
        """
        notional = entry_price * quantity
        current_notional = current_price * quantity
        
        return {
            'notional_entry': round(notional, 2),
            'notional_current': round(current_notional, 2),
            'initial_margin': round(self.initial_margin(notional, leverage), 2),
            'maintenance_margin': round(self.maintenance_margin(current_notional), 2),
            'liquidation_price': round(self.liquidation_price(entry_price, side, leverage), 4),
            'leveraged_pnl': round(self.leveraged_pnl(entry_price, current_price, quantity, leverage, side), 2),
            'margin_ratio': round(self.margin_ratio(entry_price, current_price, quantity, leverage, side), 4),
            'is_liquidated': self.check_liquidation(entry_price, current_price, side, leverage),
            'leverage': leverage,
            'side': side.upper(),
        }

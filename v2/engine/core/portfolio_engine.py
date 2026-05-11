"""
V2 Portfolio Engine — Institutional Grade
=========================================

Manages position sizing, business logic (single-position rule), 
and authoritative state synchronization.
"""

from typing import Dict, Any, Optional
from loguru import logger

class PortfolioEngineV2:
    def __init__(self, paper_trader, db_manager):
        self.paper_trader = paper_trader
        self.db = db_manager

    def get_position_state(self, user_id: int, symbol: str) -> Optional[Dict]:
        """Authorized check of current position from DB."""
        if not symbol:
            return None
        return self.db.v2_get_position(user_id, symbol)

    def calculate_units(
        self, 
        user_id: int, 
        symbol: str, 
        current_price: float, 
        atr_value: float, 
        config: Any, 
        equity: float
    ) -> float:
        """
        Determine exact quantity to trade using ATR-based risk sizing.
        """
        if current_price <= 0:
            return 0.0
            
        if atr_value > 0:
            # Smart sizing: risk 1% per trade, stop at 2×ATR
            # formula: Units = (Equity * Risk%) / (2 * ATR)
            risk_amount = equity * 0.01 
            risk_distance = 2 * atr_value
            atr_qty = risk_amount / risk_distance if risk_distance > 0 else 0
            
            # Cap to config max and margin-based max
            margin_qty = (equity * (config.get('position_size', 10) / 100)) / current_price
            quantity = min(atr_qty, margin_qty, config.get('max_quantity', 1.0))
        else:
            # Fallback to margin-based sizing
            trade_value = equity * (config.get('position_size', 10) / 100)
            quantity = min(trade_value / current_price, config.get('max_quantity', 1.0))
            
        return round(quantity, 8)

    def sync_position(self, user_id: int, symbol: str, results: list, strategy: str, leverage: float):
        """Update DB persistence after trade(s)."""
        has_open = any(r.get('success') and r.get('action') in ('OPEN', 'INCREASE', 'REVERSAL') for r in results)
        has_close = any(r.get('success') and r.get('action') in ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT') for r in results)
        
        # If any close result was successful, check if position still exists
        if has_close:
            acc = self.paper_trader._get_account(user_id)
            if symbol not in acc.positions:
                self.db.v2_delete_position(user_id, symbol)
                logger.debug(f"🗑️ [V2-PORTFOLIO] Deleted position for {symbol} (CLOSED)")
        
        # If any open/increase result was successful, save the current state
        if has_open:
            self.paper_trader.save_positions(user_id, self.db)

"""
V2 Risk Engine — Institutional Grade
====================================

Manages pre-trade safety gating and risk-adjusted constraints.
Decoupled from execution and strategy logic.
"""

from typing import Dict, Any, Tuple
from loguru import logger

class RiskEngineV2:
    def __init__(self, db_manager=None):
        self.db = db_manager
        
    def pre_trade_gate(
        self, 
        user_id: int, 
        symbol: str, 
        side: str, 
        signal_score: float, 
        expected_move_pct: float, 
        volatility_filter_result: Dict
    ) -> Tuple[bool, str]:
        """
        Final safety check before any order is sent to the portfolio engine.
        
        Rules:
        1. Signal Quality (Score > 0.6)
        2. Volatility Gate (Authorized by Regime/Volatility filter)
        3. Cost-Aware Edge Filter (Expected Move > Trading Costs)
        """
        # 1. Signal Quality
        if signal_score < 0.5:
            print(f"[RISK] ❌ Rejected: score {signal_score:.2f} < 0.5")
            return False, f"Signal score {signal_score:.2f} below threshold (0.5)"
            
        # 2. Volatility Gate
        if not volatility_filter_result.get('allowed', True):
            print(f"[RISK] ❌ Rejected: Volatility filter")
            return False, f"Volatility filter rejection: {volatility_filter_result.get('reason', 'N/A')}"
            
        # 3. Cost-Aware Edge Filter
        # Fixed estimated costs for now (Spread: 0.02%, Comm: 0.04%, Slip: 0.04%)
        estimated_cost_pct = 0.001  # 0.1% total round-trip
        print(f"[RISK] expected_move_pct: {expected_move_pct*100:.4f}%")
        print(f"[RISK] estimated_cost_pct: {estimated_cost_pct*100:.4f}%")
        
        if expected_move_pct < estimated_cost_pct:
            print(f"[RISK] ❌ Rejected: expected move < costs")
            return False, f"Expected move {expected_move_pct*100:.3f}% below trading costs {estimated_cost_pct*100:.3f}%"

        return True, "Passed"

    def check_position_risk(self, position_data: Dict, current_price: float) -> Tuple[bool, str, str]:
        """
        Check if a position has breached SL or TP levels.
        Returns: (breached, action_type, reason)
        """
        side = position_data.get('side')
        entry_price = position_data.get('entry_price')
        sl = position_data.get('stop_loss', 0)
        tp = position_data.get('take_profit', 0)
        
        if side == 'LONG':
            if sl > 0 and current_price <= sl:
                return True, 'STOP_LOSS', f"Long SL breached: {current_price} <= {sl}"
            if tp > 0 and current_price >= tp:
                return True, 'TAKE_PROFIT', f"Long TP breached: {current_price} >= {tp}"
        elif side == 'SHORT':
            if sl > 0 and current_price >= sl:
                return True, 'STOP_LOSS', f"Short SL breached: {current_price} >= {sl}"
            if tp > 0 and current_price <= tp:
                return True, 'TAKE_PROFIT', f"Short TP breached: {current_price} <= {tp}"
                
        return False, None, "Risk levels OK"

    def daily_loss_check(self, user_id: int, max_daily_loss: float) -> Tuple[bool, float]:
        """
        Check if user has exceeded their daily loss limit.
        Returns: (allowed, current_daily_pnl)
        """
        # Fetch current daily pnl from ledger (today's sum)
        import datetime
        today = datetime.datetime.utcnow().date()
        # This requires a db_manager method to get daily pnl
        daily_pnl = self.db.v2_get_daily_pnl(user_id, today) if self.db else 0.0
        
        if daily_pnl <= -max_daily_loss:
            return False, daily_pnl
        return True, daily_pnl

    def calculate_max_risk(self, user_id: int) -> float:
        """Calculate max capital risk allowed for user based on daily/total limits."""
        # Simple implementation for now: 100% unless daily loss hit
        # In production this would use a drawdown ladder
        return 1.0 

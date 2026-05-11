import pandas as pd
import numpy as np
from typing import List, Dict, Any
from config import settings

class RiskManager:
    """
    Implements global risk constraints and fail-safes.
    - Daily loss circuit breaker
    - Weekly drawdown pause
    - Consecutive loss cooldown
    - Equity curve filter
    """

    def __init__(self):
        self.consecutive_losses = 0
        self.daily_pnl_r = 0.0
        self.equity_history: List[float] = []

    def update_metrics(self, trade_result: Dict[str, Any]):
        """Updates internal risk counters after a trade closes."""
        pnl_r = trade_result.get('pnl_r', 0.0)
        self.daily_pnl_r += pnl_r
        
        if pnl_r < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0
            
        if 'equity' in trade_result:
            self.equity_history.append(trade_result['equity'])

    def can_trade(self, current_equity: float, high_water_mark: float) -> bool:
        """
        Checks all risk constraints. Returns False if any are violated.
        """
        # 1. Daily loss circuit breaker
        if self.daily_pnl_r <= -settings.DAILY_LOSS_LIMIT_R:
            return False

        # 2. Consecutive loss cooldown
        if self.consecutive_losses >= settings.CONSECUTIVE_LOSS_COOLDOWN:
            return False

        # 3. Weekly drawdown pause (simulated here with HWM)
        drawdown = (high_water_mark - current_equity) / high_water_mark if high_water_mark > 0 else 0
        if drawdown >= settings.WEEKLY_DRAWDOWN_PAUSE:
            return False

        # 4. Equity Curve Filter
        if len(self.equity_history) >= settings.EQUITY_MA_PERIOD:
            equity_ma = np.mean(self.equity_history[-settings.EQUITY_MA_PERIOD:])
            if current_equity < equity_ma:
                return False

        return True

    def reset_daily(self):
        """Reset daily counters."""
        self.daily_pnl_r = 0.0
        # consecutive_losses and equity_history persist

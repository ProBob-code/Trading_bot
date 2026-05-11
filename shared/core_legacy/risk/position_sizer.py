import math
from typing import Optional
from config import settings

class PositionSizer:
    """
    Calculates the position size for a trade based on risk parameters.
    Formula: position_size = (equity * risk_percent) / stop_distance
    """

    @staticmethod
    def calculate_size(equity: float, entry_price: float, stop_loss: float, risk_percent: Optional[float] = None) -> float:
        """
        Calculates the quantity to trade.
        """
        risk_pct = risk_percent if risk_percent is not None else settings.RISK_PER_TRADE
        
        amount_to_risk = equity * risk_pct
        stop_distance = abs(entry_price - stop_loss)
        
        if stop_distance == 0:
            return 0.0
            
        position_size = amount_to_risk / stop_distance
        
        # Kelly Criterion implementation (Fractional Kelly)
        # For simplicity, we cap the total exposure here if needed, 
        # but full Kelly logic usually depends on historical win rate/expectancy.
        # We'll apply the Fractional Kelly Cap as a maximum exposure limit.
        max_notional = equity * settings.FRACTIONAL_KELLY_CAP
        if (position_size * entry_price) > max_notional:
            position_size = max_notional / entry_price

        return position_size

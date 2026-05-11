import pandas as pd
import numpy as np
from typing import List, Dict, Any
from config import settings

class PortfolioManager:
    """
    Manages multi-asset exposure and portfolio-level risk.
    - Correlation matrix filtering
    - Max total open risk caps
    - Exposure clustering control
    """

    def __init__(self):
        self.open_positions: Dict[str, Dict[str, Any]] = {}

    def add_position(self, symbol: str, details: Dict[str, Any]):
        self.open_positions[symbol] = details

    def remove_position(self, symbol: str):
        if symbol in self.open_positions:
            del self.open_positions[symbol]

    def get_total_risk_pct(self, current_equity: float) -> float:
        """Calculates total percentage of equity currently at risk."""
        total_risk_amount = 0.0
        for symbol, pos in self.open_positions.items():
            entry = pos['entry_price']
            stop = pos['stop_loss']
            size = pos['size']
            risk = abs(entry - stop) * size
            total_risk_amount += risk
            
        return total_risk_amount / current_equity if current_equity > 0 else 0.0

    def check_correlation(self, asset_returns: pd.DataFrame) -> List[str]:
        """
        Identifies highly correlated assets that should have reduced exposure.
        Returns a list of symbols that are > threshold correlation.
        """
        if asset_returns.empty: return []
        
        corr_matrix = asset_returns.corr()
        to_reduce = []
        
        symbols = corr_matrix.columns
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                if corr_matrix.iloc[i, j] > settings.CORRELATION_THRESHOLD:
                    to_reduce.append(symbols[j])
                    
        return list(set(to_reduce))

    def can_add_exposure(self, symbol: str, current_equity: float, new_risk_pct: float) -> bool:
        """
        Checks if adding a new position violates portfolio-level risk caps.
        """
        current_risk = self.get_total_risk_pct(current_equity)
        if (current_risk + new_risk_pct) > settings.MAX_TOTAL_OPEN_RISK:
            return False
            
        return True

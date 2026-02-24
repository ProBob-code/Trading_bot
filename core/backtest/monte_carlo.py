import numpy as np
import pandas as pd
from typing import Dict, Any, List

class MonteCarloEngine:
    """
    Simulates thousands of equity curve variations to estimate risk parameters.
    """

    @staticmethod
    def run_simulation(trade_returns_r: List[float], num_simulations: int = 1000, initial_capital: float = 100000.0, risk_per_trade: float = 0.01) -> Dict[str, Any]:
        """
        Performs bootstrapped simulation of trade sequences.
        """
        if not trade_returns_r:
            return {}

        results = []
        max_drawdowns = []
        ruin_count = 0
        
        for _ in range(num_simulations):
            # Reshuffle returns with replacement
            sampled_returns = np.random.choice(trade_returns_r, size=len(trade_returns_r), replace=True)
            
            equity = initial_capital
            equity_curve = [initial_capital]
            
            for r_mult in sampled_returns:
                risk_amount = equity * risk_per_trade
                pnl = r_mult * risk_amount
                equity += pnl
                equity_curve.append(equity)
                
                if equity <= 0:
                    ruin_count += 1
                    break
            
            equity_curve = np.array(equity_curve)
            max_dd = (np.maximum.accumulate(equity_curve) - equity_curve).max()
            max_drawdowns.append(max_dd / np.maximum.accumulate(equity_curve).max() if max_dd > 0 else 0)
            results.append(equity)

        return {
            "Mean Final Equity": round(np.mean(results), 2),
            "Median Final Equity": round(np.median(results), 2),
            "Max Drawdown (95th Percentile)": round(np.percentile(max_drawdowns, 95) * 100, 2),
            "Max Drawdown (Max)": round(np.max(max_drawdowns) * 100, 2),
            "Risk of Ruin %": round((ruin_count / num_simulations) * 100, 2)
        }

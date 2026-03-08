"""
V2 Strategy Ranking Engine
============================

Composite scoring and ranking of strategies.

Score formula:
  score = 0.3 × expectancy + 0.3 × sharpe + 0.2 × profit_factor + 0.2 × inverse_drawdown
"""

from typing import List, Dict
from loguru import logger


class StrategyRanker:
    """
    Ranks strategies by composite institutional score.
    """
    
    # Score weights
    W_EXPECTANCY = 0.3
    W_SHARPE = 0.3
    W_PROFIT_FACTOR = 0.2
    W_INVERSE_DD = 0.2
    
    def composite_score(self, metrics: Dict) -> float:
        """
        Compute composite score for a single strategy.
        
        Components are normalized to comparable ranges:
        - Expectancy: raw value (typically -100 to +100)
        - Sharpe: raw value (typically -3 to +3)
        - Profit factor: capped at 10 (prevent infinity domination)
        - Inverse drawdown: 100 / (1 + max_drawdown_pct) to reward low drawdown
        
        Args:
            metrics: Strategy metrics dict
            
        Returns:
            Composite score (higher is better)
        """
        expectancy = metrics.get('expectancy', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        profit_factor = min(metrics.get('profit_factor', 0), 10)  # Cap at 10
        max_dd = metrics.get('max_drawdown_pct', 0)
        
        # Inverse drawdown: higher score for lower drawdown
        inverse_dd = 100 / (1 + max_dd) if max_dd >= 0 else 100
        
        # Normalize each component to [0, 100] range approximately
        norm_exp = self._normalize(expectancy, -50, 100)
        norm_sharpe = self._normalize(sharpe, -2, 3)
        norm_pf = self._normalize(profit_factor, 0, 5)
        norm_dd = self._normalize(inverse_dd, 0, 100)
        
        score = (
            self.W_EXPECTANCY * norm_exp +
            self.W_SHARPE * norm_sharpe +
            self.W_PROFIT_FACTOR * norm_pf +
            self.W_INVERSE_DD * norm_dd
        )
        
        return round(score, 4)
    
    def _normalize(self, value: float, min_val: float, max_val: float) -> float:
        """Normalize value to 0-100 scale."""
        if max_val <= min_val:
            return 50.0
        clamped = max(min_val, min(value, max_val))
        return ((clamped - min_val) / (max_val - min_val)) * 100
    
    def rank(self, all_metrics: List[Dict]) -> List[Dict]:
        """
        Rank strategies by composite score.
        
        Args:
            all_metrics: List of strategy metrics dicts
            
        Returns:
            Sorted list with rank and composite_score added
        """
        if not all_metrics:
            return []
        
        scored = []
        for m in all_metrics:
            entry = {**m}
            entry['composite_score'] = self.composite_score(m)
            scored.append(entry)
        
        # Sort descending by composite score
        scored.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank
        for i, entry in enumerate(scored):
            entry['rank'] = i + 1
        
        return scored
    
    def get_best_strategy(self, all_metrics: List[Dict]) -> Dict:
        """Get the top-ranked strategy."""
        ranked = self.rank(all_metrics)
        return ranked[0] if ranked else {}

"""
V2 Capital Allocator
=====================

Dynamic capital allocation across strategies based on performance.

Features:
- Per-strategy capital weighting
- Drawdown-based scaling (reduce allocation as drawdown increases)
- Auto-disable strategy if max_drawdown_limit breached
- Rebalancing logic
"""

from typing import Dict, List, Optional
from loguru import logger


class CapitalAllocator:
    """
    Allocates portfolio capital across multiple strategies.
    
    Uses strategy metrics to determine weights:
    - Better performing strategies get more capital
    - Strategies in drawdown get scaled down
    - Strategies exceeding max drawdown limit get disabled
    """
    
    MIN_ALLOCATION_PCT = 0.05  # Minimum 5% per active strategy
    
    def allocate(
        self,
        strategies: List[Dict],
        total_capital: float,
        strategy_profiles: Dict[str, Dict] = None
    ) -> Dict[str, Dict]:
        """
        Compute capital allocation for each strategy.
        
        Args:
            strategies: List of strategy metrics dicts (must have 'strategy', 'expectancy', 
                        'sharpe_ratio', 'max_drawdown_pct')
            total_capital: Total portfolio capital to allocate
            strategy_profiles: Optional overrides per strategy (max_drawdown_limit, etc.)
            
        Returns:
            Dict mapping strategy name → { weight, allocated_capital, status, reason }
        """
        if not strategies:
            return {}
        
        profiles = strategy_profiles or {}
        allocations = {}
        active_strategies = []
        
        # Step 1: Filter out strategies that exceed their drawdown limit
        for s in strategies:
            name = s.get('strategy', 'unknown')
            profile = profiles.get(name, {})
            max_dd_limit = profile.get('max_drawdown_limit', 25.0)  # Default 25%
            current_dd = s.get('max_drawdown_pct', 0)
            
            if current_dd >= max_dd_limit:
                allocations[name] = {
                    'weight': 0,
                    'allocated_capital': 0,
                    'status': 'DISABLED',
                    'reason': f'Drawdown {current_dd:.1f}% ≥ limit {max_dd_limit:.1f}%',
                }
                logger.warning(f"[V2-ALLOC] Strategy {name} DISABLED: DD={current_dd:.1f}%")
            else:
                active_strategies.append(s)
        
        if not active_strategies:
            return allocations
        
        # Step 2: Compute raw scores
        scores = {}
        for s in active_strategies:
            name = s.get('strategy', 'unknown')
            expectancy = max(s.get('expectancy', 0), 0)
            sharpe = max(s.get('sharpe_ratio', 0), 0)
            current_dd = s.get('max_drawdown_pct', 0)
            
            # Drawdown scaling: reduce weight as drawdown increases
            dd_scale = max(0.1, 1 - (current_dd / 100))
            
            # Composite score for allocation
            score = (expectancy * 0.4 + sharpe * 0.3 + 0.3) * dd_scale
            scores[name] = max(score, 0.01)  # Floor at tiny positive
        
        # Step 3: Normalize to weights
        total_score = sum(scores.values())
        
        for name, score in scores.items():
            raw_weight = score / total_score if total_score > 0 else 1.0 / len(scores)
            
            # Enforce minimum allocation
            weight = max(raw_weight, self.MIN_ALLOCATION_PCT)
            
            allocations[name] = {
                'weight': round(weight, 4),
                'allocated_capital': round(total_capital * weight, 2),
                'status': 'ACTIVE',
                'reason': None,
                'dd_scale': round(1 - (next(
                    s.get('max_drawdown_pct', 0) for s in active_strategies 
                    if s.get('strategy') == name
                ) / 100), 4),
            }
        
        # Step 4: Renormalize if min allocations pushed total over 100%
        total_weight = sum(a['weight'] for a in allocations.values() if a['status'] == 'ACTIVE')
        if total_weight > 1.0:
            for name, alloc in allocations.items():
                if alloc['status'] == 'ACTIVE':
                    alloc['weight'] = round(alloc['weight'] / total_weight, 4)
                    alloc['allocated_capital'] = round(total_capital * alloc['weight'], 2)
        
        return allocations
    
    def get_allocation_summary(self, allocations: Dict[str, Dict], total_capital: float) -> Dict:
        """Get a summary of the current allocation state."""
        active = {k: v for k, v in allocations.items() if v['status'] == 'ACTIVE'}
        disabled = {k: v for k, v in allocations.items() if v['status'] == 'DISABLED'}
        
        return {
            'total_capital': total_capital,
            'active_strategies': len(active),
            'disabled_strategies': len(disabled),
            'allocated_capital': sum(a['allocated_capital'] for a in active.values()),
            'allocations': allocations,
        }

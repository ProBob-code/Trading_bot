"""
Portfolio Allocation Layer
==========================

Turns the system from evaluation engine → capital allocator.

Allocation methods:
1. Kelly Fraction — optimal bet sizing (capped at half-Kelly)
2. Volatility Parity — weight inversely proportional to volatility
3. Risk Parity — weight inversely proportional to max drawdown
4. Composite Score Weighting — weight proportional to safety-filtered score
5. Dynamic Reallocation — recalculate every N trades
"""

import numpy as np
from typing import Dict, Any, List, Optional


class PortfolioAllocator:
    """
    Multi-bot capital allocation engine.
    
    Computes recommended allocation across bots using
    multiple frameworks, then blends into final recommendation.
    """
    
    def __init__(
        self,
        max_kelly_fraction: float = 0.25,  # Half-Kelly cap
        rebalance_frequency: int = 20,      # Trades between rebalances
        min_allocation_pct: float = 5.0,     # Minimum allocation per bot
        max_allocation_pct: float = 40.0,    # Maximum allocation per bot
    ):
        self.max_kelly_fraction = max_kelly_fraction
        self.rebalance_frequency = rebalance_frequency
        self.min_allocation_pct = min_allocation_pct
        self.max_allocation_pct = max_allocation_pct
    
    def allocate(
        self,
        bot_metrics: Dict[str, Dict[str, Any]],
        total_capital: float = 100000.0,
    ) -> Dict[str, Any]:
        """
        Compute portfolio allocation across all bots.
        
        Only SAFE and CAUTION bots receive allocation.
        DANGEROUS bots get 0%.
        
        Args:
            bot_metrics: Dict of bot_id → metrics dict
            total_capital: Total portfolio capital
            
        Returns:
            Allocation recommendations per bot + portfolio summary
        """
        if not bot_metrics:
            return {'allocations': [], 'portfolio_summary': {}}
        
        # Filter eligible bots (not DANGEROUS)
        eligible = {
            bid: m for bid, m in bot_metrics.items()
            if m.get('safety_label', 'CAUTION') != 'DANGEROUS'
        }
        
        if not eligible:
            return {
                'allocations': [{
                    'bot_id': bid,
                    'kelly_fraction': 0,
                    'vol_parity_weight': 0,
                    'risk_parity_weight': 0,
                    'score_weight': 0,
                    'recommended_allocation_pct': 0,
                    'recommended_capital': 0,
                    'reason': 'DANGEROUS — excluded',
                } for bid in bot_metrics],
                'portfolio_summary': {'status': 'All bots classified DANGEROUS'},
            }
        
        allocations = []
        
        # ─── 1. Kelly Fraction ───
        kelly_weights = {}
        for bid, m in eligible.items():
            kelly = self._kelly_fraction(
                win_rate=m.get('win_rate', 0) / 100,
                avg_win=m.get('avg_win', 0),
                avg_loss=m.get('avg_loss', 1),
            )
            kelly_weights[bid] = kelly
        
        # ─── 2. Volatility Parity ───
        vol_weights = self._volatility_parity(eligible)
        
        # ─── 3. Risk Parity ───
        risk_weights = self._risk_parity(eligible)
        
        # ─── 4. Composite Score Weighting ───
        score_weights = self._score_weighting(eligible)
        
        # ─── 5. Blend ───
        # Weights for blending: Kelly 20%, Vol Parity 25%, Risk Parity 25%, Score 30%
        blend_weights = {'kelly': 0.20, 'vol': 0.25, 'risk': 0.25, 'score': 0.30}
        
        blended = {}
        for bid in eligible:
            blended[bid] = (
                kelly_weights.get(bid, 0) * blend_weights['kelly'] +
                vol_weights.get(bid, 0) * blend_weights['vol'] +
                risk_weights.get(bid, 0) * blend_weights['risk'] +
                score_weights.get(bid, 0) * blend_weights['score']
            )
        
        # Normalize to 100%
        total_blend = sum(blended.values())
        if total_blend > 0:
            for bid in blended:
                blended[bid] = blended[bid] / total_blend * 100
        
        # Apply min/max constraints
        for bid in blended:
            blended[bid] = max(self.min_allocation_pct, min(self.max_allocation_pct, blended[bid]))
        
        # Re-normalize after constraints
        total_constrained = sum(blended.values())
        if total_constrained > 0:
            for bid in blended:
                blended[bid] = blended[bid] / total_constrained * 100
        
        # Build allocation output
        for bid in bot_metrics:
            if bid in eligible:
                alloc_pct = round(blended.get(bid, 0), 1)
                allocations.append({
                    'bot_id': bid,
                    'kelly_fraction': round(kelly_weights.get(bid, 0), 4),
                    'vol_parity_weight': round(vol_weights.get(bid, 0), 4),
                    'risk_parity_weight': round(risk_weights.get(bid, 0), 4),
                    'score_weight': round(score_weights.get(bid, 0), 4),
                    'recommended_allocation_pct': alloc_pct,
                    'recommended_capital': round(total_capital * alloc_pct / 100, 2),
                    'safety_label': bot_metrics[bid].get('safety_label', 'CAUTION'),
                })
            else:
                allocations.append({
                    'bot_id': bid,
                    'kelly_fraction': 0,
                    'vol_parity_weight': 0,
                    'risk_parity_weight': 0,
                    'score_weight': 0,
                    'recommended_allocation_pct': 0,
                    'recommended_capital': 0,
                    'safety_label': 'DANGEROUS',
                    'reason': 'DANGEROUS — excluded from allocation',
                })
        
        # Sort by allocation
        allocations.sort(key=lambda x: x['recommended_allocation_pct'], reverse=True)
        
        # Portfolio summary
        eligible_count = len(eligible)
        total_allocated = sum(a['recommended_allocation_pct'] for a in allocations)
        
        portfolio_summary = {
            'total_bots': len(bot_metrics),
            'eligible_bots': eligible_count,
            'excluded_bots': len(bot_metrics) - eligible_count,
            'total_capital': total_capital,
            'allocated_pct': round(total_allocated, 1),
            'blend_method': 'Kelly(20%) + VolParity(25%) + RiskParity(25%) + Score(30%)',
            'kelly_cap': self.max_kelly_fraction,
            'constraints': f'Min {self.min_allocation_pct}%, Max {self.max_allocation_pct}% per bot',
        }
        
        return {
            'allocations': allocations,
            'portfolio_summary': portfolio_summary,
        }
    
    def _kelly_fraction(
        self,
        win_rate: float,
        avg_win: float,
        avg_loss: float,
    ) -> float:
        """
        Kelly Criterion: f* = (W × B - L) / B
        where W = win rate, L = loss rate, B = avg_win/avg_loss
        
        Capped at half-Kelly for safety.
        """
        if avg_loss <= 0 or avg_win <= 0:
            return 0
        
        b = avg_win / avg_loss  # Win/loss ratio
        loss_rate = 1 - win_rate
        
        kelly = (win_rate * b - loss_rate) / b
        kelly = max(0, kelly)
        
        # Half-Kelly cap
        return min(kelly * 0.5, self.max_kelly_fraction)
    
    def _volatility_parity(self, bot_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Volatility parity: weight = 1 / individual_volatility.
        Lower volatility → higher allocation.
        """
        weights = {}
        for bid, m in bot_metrics.items():
            # Use equity volatility (from Sharpe denominator)
            # Proxy: 1 / (max_dd * some factor) or use sortino inverse
            dd = max(m.get('max_drawdown_pct', 10), 1)
            weights[bid] = 1.0 / dd
        
        # Normalize
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _risk_parity(self, bot_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Risk parity: weight = 1 / max_drawdown.
        Lower drawdown → higher allocation.
        """
        weights = {}
        for bid, m in bot_metrics.items():
            dd = max(m.get('max_drawdown_pct', 10), 0.5)
            ror = max(m.get('risk_of_ruin', 0), 0.01)
            # Combined risk score
            risk = dd * 0.7 + ror * 0.3
            weights[bid] = 1.0 / max(risk, 0.1)
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights
    
    def _score_weighting(self, bot_metrics: Dict[str, Dict]) -> Dict[str, float]:
        """
        Composite score weighting: weight proportional to composite score.
        SAFE bots get 1.5× bonus. CAUTION gets 1.0×.
        """
        weights = {}
        for bid, m in bot_metrics.items():
            score = max(m.get('composite_score', 0), 0)
            safety = m.get('safety_label', 'CAUTION')
            
            # Safety multiplier
            if safety == 'SAFE':
                score *= 1.5
            elif safety == 'CAUTION':
                score *= 1.0
            
            weights[bid] = score
        
        total = sum(weights.values())
        if total > 0:
            weights = {k: v / total for k, v in weights.items()}
        
        return weights

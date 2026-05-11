"""
V2 Volatility Filter
======================

Filters strategy execution based on market regime.

Rules:
- Block incompatible strategies per regime
- Reduce leverage during high volatility
- Log warnings when trades are blocked

Strategy-regime compatibility:
- TRENDING  → Ichimoku ✓, MACD+RSI ✓, ML Forecast ✓ | Bollinger ⚠
- RANGING   → Bollinger ✓, MACD+RSI ✓ | Ichimoku ✗, ML Forecast ⚠
- VOLATILE  → All reduced leverage, Combined only
"""

from typing import Dict, Optional
from loguru import logger

from .regime_detector import MarketRegime


# Strategy compatibility matrix
# True = allowed, False = blocked, 'reduced' = allowed with reduced leverage
STRATEGY_REGIME_MATRIX = {
    'ichimoku': {
        MarketRegime.TRENDING: True,
        MarketRegime.RANGING: False,
        MarketRegime.VOLATILE: 'reduced',
        MarketRegime.UNKNOWN: True,
    },
    'bollinger': {
        MarketRegime.TRENDING: 'reduced',
        MarketRegime.RANGING: True,
        MarketRegime.VOLATILE: 'reduced',
        MarketRegime.UNKNOWN: True,
    },
    'macd_rsi': {
        MarketRegime.TRENDING: True,
        MarketRegime.RANGING: True,
        MarketRegime.VOLATILE: 'reduced',
        MarketRegime.UNKNOWN: True,
    },
    'ml_forecast': {
        MarketRegime.TRENDING: True,
        MarketRegime.RANGING: 'reduced',
        MarketRegime.VOLATILE: False,
        MarketRegime.UNKNOWN: True,
    },
    'combined': {
        MarketRegime.TRENDING: True,
        MarketRegime.RANGING: True,
        MarketRegime.VOLATILE: True,
        MarketRegime.UNKNOWN: True,
    },
}

# Leverage reduction factor for 'reduced' status
VOLATILITY_LEVERAGE_SCALE = 0.5  # Cut leverage in half during high vol


class VolatilityFilter:
    """
    Pre-execution filter based on market regime.
    
    Intercepts before trade execution to:
    1. Block incompatible strategy+regime combinations
    2. Reduce leverage during high volatility
    3. Log regime context for audit trail
    """
    
    def __init__(self, strategy_matrix: Dict = None, leverage_scale: float = None):
        """
        Args:
            strategy_matrix: Custom strategy-regime compatibility matrix
            leverage_scale: Leverage reduction factor for 'reduced' status
        """
        self.matrix = strategy_matrix or STRATEGY_REGIME_MATRIX
        self.leverage_scale = leverage_scale or VOLATILITY_LEVERAGE_SCALE
    
    def filter(
        self,
        strategy: str,
        regime: str,
        leverage: float = 1.0
    ) -> Dict:
        """
        Check if a strategy should execute given the current regime.
        
        Args:
            strategy: Strategy name (e.g., 'ichimoku')
            regime: Current market regime (e.g., 'TRENDING')
            leverage: Requested leverage
            
        Returns:
            {
                allowed: bool,
                leverage: float (adjusted),
                reason: str or None,
                regime: str,
                strategy: str,
            }
        """
        strategy_lower = strategy.lower()
        
        # Look up compatibility
        compat = self.matrix.get(strategy_lower, {})
        status = compat.get(regime, True)  # Default: allow unknown strategies
        
        if status is False:
            reason = f"Strategy '{strategy}' blocked in {regime} regime"
            logger.warning(f"[V2-FILTER] 🚫 {reason}")
            return {
                'allowed': False,
                'leverage': 0,
                'reason': reason,
                'regime': regime,
                'strategy': strategy,
            }
        
        adjusted_leverage = leverage
        reason = None
        
        if status == 'reduced':
            adjusted_leverage = leverage * self.leverage_scale
            reason = f"Leverage reduced {leverage}× → {adjusted_leverage}× (regime={regime})"
            logger.info(f"[V2-FILTER] ⚠️ {reason}")
        
        return {
            'allowed': True,
            'leverage': adjusted_leverage,
            'reason': reason,
            'regime': regime,
            'strategy': strategy,
        }
    
    def get_compatible_strategies(self, regime: str) -> Dict[str, str]:
        """
        Get all strategies and their compatibility status for a given regime.
        
        Returns:
            Dict mapping strategy → 'allowed' | 'reduced' | 'blocked'
        """
        result = {}
        for strategy, compat in self.matrix.items():
            status = compat.get(regime, True)
            if status is True:
                result[strategy] = 'allowed'
            elif status == 'reduced':
                result[strategy] = 'reduced'
            else:
                result[strategy] = 'blocked'
        return result

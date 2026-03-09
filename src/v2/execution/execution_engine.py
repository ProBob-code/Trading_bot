"""
V2 Execution Engine
====================

Institutional-grade execution with:
- ATR-aware slippage simulation
- Volume-based spread model
- Commission model
- Deterministic mode for backtesting reproducibility

This is the SINGLE SOURCE OF TRUTH for fill prices.
No frontend price manipulation — backend authoritative.
"""

import random
from dataclasses import dataclass
from typing import Optional, Tuple
from loguru import logger


class SlippageModel:
    """
    ATR-aware slippage simulation.
    
    Slippage scales with volatility:
    - Higher volatility → more slippage
    - Deterministic mode for reproducible backtests
    """
    
    BASE_SLIPPAGE = 0.0005  # 0.05% base
    
    def compute(
        self,
        side: str,
        base_price: float,
        volatility: float = 0.02,
        deterministic: bool = False
    ) -> Tuple[float, float]:
        """
        Compute slippage-adjusted price.
        
        Args:
            side: 'BUY' or 'SELL'
            base_price: Market price before slippage
            volatility: ATR / price ratio (e.g., 0.02 = 2% daily range)
            deterministic: If True, use fixed multiplier (no randomness)
            
        Returns:
            (slippage_pct, slippage_amount)
        """
        # Scale by volatility — normalize around 2% daily range
        volatility_multiplier = max(1.0, volatility / 0.02)
        
        # Random factor (or fixed 1.0 for deterministic mode)
        random_factor = 1.0 if deterministic else random.uniform(0.5, 1.5)
        
        slippage_pct = self.BASE_SLIPPAGE * volatility_multiplier * random_factor
        
        # BUY → worse entry (higher), SELL → worse exit (lower)
        direction = 1 if side.upper() == 'BUY' else -1
        slippage_amount = base_price * slippage_pct * direction
        
        return slippage_pct, slippage_amount


class SpreadModel:
    """
    Volume-based bid/ask spread simulation.
    
    Spread tiers:
    - Low volume  (<1M):    0.30% (illiquid)
    - Medium volume (<100M): 0.10%
    - High volume  (≥100M):  0.03% (liquid)
    """
    
    TIERS = [
        (1e6,   0.003),   # <1M   → 0.30%
        (100e6, 0.001),   # <100M → 0.10%
        (float('inf'), 0.0003),  # ≥100M → 0.03%
    ]
    
    def compute(self, volume: float = 100e6) -> float:
        """
        Compute spread as decimal fraction based on 24h volume.
        
        Args:
            volume: 24-hour trading volume in USD
            
        Returns:
            spread as decimal (e.g., 0.001 = 0.1%)
        """
        for threshold, spread in self.TIERS:
            if volume < threshold:
                return spread
        return self.TIERS[-1][1]


class CommissionModel:
    """
    Commission model — flat percentage of trade value.
    """
    
    DEFAULT_RATE = 0.0004  # 0.04% (Binance taker fee)
    
    def __init__(self, rate: float = None):
        self.rate = rate if rate is not None else self.DEFAULT_RATE
    
    def compute(self, fill_price: float, quantity: float) -> float:
        """
        Compute commission.
        
        Args:
            fill_price: Execution price per unit
            quantity: Number of units
            
        Returns:
            Commission amount in USD
        """
        return fill_price * quantity * self.rate


@dataclass
class ExecutionResult:
    """Complete execution output — stored in trade ledger."""
    fill_price: float
    commission: float
    spread_pct: float
    slippage_pct: float
    volatility_input: float
    volume_input: float
    market_price: float
    side: str
    quantity: float
    
    def to_dict(self) -> dict:
        return {
            'fill_price': round(self.fill_price, 8),
            'commission': round(self.commission, 8),
            'spread_pct': round(self.spread_pct * 100, 6),
            'slippage_pct': round(self.slippage_pct * 100, 6),
            'volatility_input': round(self.volatility_input, 6),
            'volume_input': round(self.volume_input, 2),
            'market_price': round(self.market_price, 8),
            'side': self.side,
            'quantity': self.quantity,
        }


class ExecutionEngine:
    """
    Institutional-grade execution engine.
    
    Single source of truth for all fill prices.
    Combines slippage, spread, and commission models.
    Supports deterministic mode for reproducible backtesting.
    """
    
    def __init__(
        self,
        slippage_model: Optional[SlippageModel] = None,
        spread_model: Optional[SpreadModel] = None,
        commission_model: Optional[CommissionModel] = None,
        deterministic: bool = False
    ):
        self.slippage_model = slippage_model or SlippageModel()
        self.spread_model = spread_model or SpreadModel()
        self.commission_model = commission_model or CommissionModel()
        self.deterministic = deterministic
        
        mode_label = "DETERMINISTIC" if deterministic else "STOCHASTIC"
        logger.info(f"[V2-EXEC] ExecutionEngine initialized ({mode_label})")
    
    def execute(
        self,
        side: str,
        market_price: float,
        quantity: float,
        volatility: float = 0.02,
        volume: float = 100e6,
        deterministic: Optional[bool] = None
    ) -> ExecutionResult:
        """
        Execute a trade with realistic market microstructure.
        
        Args:
            side: 'BUY' or 'SELL'
            market_price: Current market price
            quantity: Trade quantity
            volatility: ATR/price ratio for slippage scaling
            volume: 24h volume for spread tier
            deterministic: Override instance-level deterministic flag
            
        Returns:
            ExecutionResult with all execution details
        """
        det = deterministic if deterministic is not None else self.deterministic
        side_upper = side.upper()
        
        # 1. Compute spread
        spread_pct = self.spread_model.compute(volume)
        
        # 2. Compute slippage
        slippage_pct, _ = self.slippage_model.compute(
            side_upper, market_price, volatility, det
        )
        
        # 3. Apply to price
        if side_upper == 'BUY':
            fill_price = market_price * (1 + spread_pct + slippage_pct)
        else:
            fill_price = market_price * (1 - spread_pct - slippage_pct)
        
        # 4. Compute commission
        commission = self.commission_model.compute(fill_price, quantity)
        
        logger.debug(
            f"[V2-EXEC] {side_upper} {quantity} @ mkt={market_price:.4f} → "
            f"fill={fill_price:.4f} (spread={spread_pct*100:.4f}%, "
            f"slip={slippage_pct*100:.4f}%, comm=${commission:.4f})"
        )
        
        return ExecutionResult(
            fill_price=fill_price,
            commission=commission,
            spread_pct=spread_pct,
            slippage_pct=slippage_pct,
            volatility_input=volatility,
            volume_input=volume,
            market_price=market_price,
            side=side_upper,
            quantity=quantity,
        )

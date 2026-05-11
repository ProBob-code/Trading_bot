"""
Strategy Bot Base Class
=======================

Abstract base for all paper trading strategy bots.
Enforces: entry/SL/TP definition, R:R floor, position sizing,
negative expectancy gating, time-based exits, trade reasoning.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone
import uuid
import pandas as pd
import numpy as np


@dataclass
class TradeSignal:
    """
    A validated trade signal produced by a strategy.
    
    Every signal MUST define entry, SL, TP, side, and reasoning.
    Position size is calculated externally via risk module.
    """
    side: str                          # "buy" or "sell"
    entry_price: float = 0.0
    sl_price: float = 0.0
    tp_price: float = 0.0
    confidence: float = 0.0            # 0.0 to 1.0
    reason: str = ""                   # Human-readable trade justification
    regime: str = ""                   # Regime at signal time
    indicator_snapshot: Dict[str, float] = field(default_factory=dict)
    order_type: str = "market"         # market | limit | stop
    timestamp: Optional[datetime] = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now(timezone.utc)
    
    @property
    def risk_distance(self) -> float:
        """Distance from entry to SL."""
        return abs(self.entry_price - self.sl_price)
    
    @property
    def reward_distance(self) -> float:
        """Distance from entry to TP."""
        return abs(self.tp_price - self.entry_price)
    
    @property
    def rr_ratio(self) -> float:
        """Risk:Reward ratio."""
        if self.risk_distance > 0:
            return self.reward_distance / self.risk_distance
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'side': self.side,
            'entry_price': self.entry_price,
            'sl_price': self.sl_price,
            'tp_price': self.tp_price,
            'confidence': self.confidence,
            'reason': self.reason,
            'regime': self.regime,
            'indicator_snapshot': self.indicator_snapshot,
            'order_type': self.order_type,
            'rr_ratio': round(self.rr_ratio, 2),
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
        }


class StrategyBot(ABC):
    """
    Abstract base class for paper trading strategy bots.
    
    All strategies must implement:
    - calculate_indicators(): add required indicators to DataFrame
    - generate_signal(): produce a TradeSignal or None for each bar
    
    The base class enforces:
    - Minimum R:R ratio (rejects signals below threshold)
    - Negative expectancy gating (stops generating signals if bot is failing)
    - Trade reasoning requirement
    - Time-based exit tracking
    """
    
    def __init__(
        self,
        name: str,
        min_rr: float = 2.0,
        expectancy_gate: bool = True,
        min_expectancy: float = -0.5,
        params: Dict[str, Any] = None,
    ):
        self.name = name
        self.min_rr = min_rr
        self.expectancy_gate = expectancy_gate
        self.min_expectancy = min_expectancy
        self.params = params or {}
        
        # Performance tracking for expectancy gate
        self._trade_results: List[float] = []
        self._is_gated = False
    
    @abstractmethod
    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add required technical indicators to the DataFrame.
        Must return a copy with new columns added.
        """
        pass
    
    @abstractmethod
    def generate_signal(
        self,
        df: pd.DataFrame,
        idx: int,
        regime: Dict[str, str],
    ) -> Optional[TradeSignal]:
        """
        Generate a trade signal for bar at index `idx`.
        
        Args:
            df: DataFrame with OHLCV + calculated indicators
            idx: Current bar index
            regime: Current regime dict from RegimeDetector
            
        Returns:
            TradeSignal or None if no signal
        """
        pass
    
    def validate_signal(self, signal: TradeSignal) -> tuple[bool, str]:
        """
        Validate a signal against R:R, expectancy gate, and sanity checks.
        
        Returns (is_valid, reason).
        """
        if signal is None:
            return False, "No signal"
        
        # Entry price must be positive
        if signal.entry_price <= 0:
            return False, f"Invalid entry price: {signal.entry_price}"
        
        # SL sanity
        if signal.side == "buy":
            if signal.sl_price >= signal.entry_price:
                return False, f"Long SL ({signal.sl_price}) must be < entry ({signal.entry_price})"
            if signal.tp_price <= signal.entry_price:
                return False, f"Long TP ({signal.tp_price}) must be > entry ({signal.entry_price})"
        elif signal.side == "sell":
            if signal.sl_price <= signal.entry_price:
                return False, f"Short SL ({signal.sl_price}) must be > entry ({signal.entry_price})"
            if signal.tp_price >= signal.entry_price:
                return False, f"Short TP ({signal.tp_price}) must be < entry ({signal.entry_price})"
        
        # R:R floor
        if signal.rr_ratio < self.min_rr:
            return False, (
                f"R:R {signal.rr_ratio:.2f} below minimum {self.min_rr} "
                f"(risk={signal.risk_distance:.4f}, reward={signal.reward_distance:.4f})"
            )
        
        # Expectancy gate
        if self.expectancy_gate and self._is_gated:
            return False, f"Bot gated: negative expectancy ({self.current_expectancy:.2f}R)"
        
        # Must have reasoning
        if not signal.reason.strip():
            return False, "Signal must include trade reasoning"
        
        return True, "OK"
    
    def record_result(self, r_multiple: float):
        """Record a trade result for expectancy tracking."""
        self._trade_results.append(r_multiple)
        # Check gate after 10+ trades
        if len(self._trade_results) >= 10:
            if self.current_expectancy < self.min_expectancy:
                self._is_gated = True
            else:
                self._is_gated = False
    
    @property
    def current_expectancy(self) -> float:
        """Rolling expectancy in R-multiples."""
        if not self._trade_results:
            return 0.0
        return float(np.mean(self._trade_results))
    
    @property
    def is_gated(self) -> bool:
        return self._is_gated
    
    def reset(self):
        """Reset performance tracking."""
        self._trade_results.clear()
        self._is_gated = False
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__} name={self.name} gated={self._is_gated}>"

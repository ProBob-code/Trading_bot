"""
Bot Configuration
=================

Dataclass for multi-bot paper trading configuration.
Each bot has isolated capital, strategy, and risk parameters.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from enum import Enum
import uuid


class ExecutionMode(Enum):
    """Order execution delay model."""
    INSTANT = "instant"
    NEXT_BAR_OPEN = "next_bar_open"
    RANDOM_DELAY = "random_delay_ms"


class OrderType(Enum):
    """Supported order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"


@dataclass
class BotConfig:
    """
    Configuration for a single paper trading bot.
    
    Each bot runs an independent strategy on a single instrument
    with isolated capital and risk parameters.
    """
    # Identity
    bot_id: str = ""
    name: str = ""
    instrument: str = "BTCUSDT"
    strategy: str = "breakout"
    timeframe: str = "1h"
    
    # Capital (isolated per bot)
    virtual_capital: float = 100000.0
    
    # Risk parameters
    risk_per_trade_pct: float = 1.0          # % of capital risked per trade
    min_rr: float = 2.0                       # Minimum R:R ratio (reject below)
    max_concurrent_trades: int = 3            # Max open trades for this bot
    max_risk_pct: float = 5.0                 # Hard cap on risk %
    
    # Execution simulation
    execution_mode: str = "next_bar_open"     # instant | next_bar_open | random_delay_ms
    slippage_base_pct: float = 0.05           # Base slippage %
    spread_pct: float = 0.02                  # Spread simulation %
    maker_fee_pct: float = 0.02               # Maker fee %
    taker_fee_pct: float = 0.06               # Taker fee %
    
    # Time-based exit
    max_bars_in_trade: Optional[int] = None   # Close at market if exceeded
    
    # Liquidity
    max_position_volume_pct: float = 5.0      # Max % of 20-bar avg volume
    
    # Strategy-specific parameters
    params: Dict[str, Any] = field(default_factory=dict)
    
    # Status
    enabled: bool = True
    
    def __post_init__(self):
        if not self.bot_id:
            self.bot_id = f"{self.strategy}_{self.instrument}_{uuid.uuid4().hex[:8]}"
        if not self.name:
            self.name = f"{self.strategy.title()} on {self.instrument}"
        # Clamp risk
        self.risk_per_trade_pct = min(self.risk_per_trade_pct, self.max_risk_pct)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            'bot_id': self.bot_id,
            'name': self.name,
            'instrument': self.instrument,
            'strategy': self.strategy,
            'timeframe': self.timeframe,
            'virtual_capital': self.virtual_capital,
            'risk_per_trade_pct': self.risk_per_trade_pct,
            'min_rr': self.min_rr,
            'max_concurrent_trades': self.max_concurrent_trades,
            'max_risk_pct': self.max_risk_pct,
            'execution_mode': self.execution_mode,
            'slippage_base_pct': self.slippage_base_pct,
            'spread_pct': self.spread_pct,
            'maker_fee_pct': self.maker_fee_pct,
            'taker_fee_pct': self.taker_fee_pct,
            'max_bars_in_trade': self.max_bars_in_trade,
            'max_position_volume_pct': self.max_position_volume_pct,
            'params': self.params,
            'enabled': self.enabled,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BotConfig':
        """Create from dict."""
        known_fields = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in known_fields}
        return cls(**filtered)


def create_default_bots(instrument: str = "BTCUSDT", capital_each: float = 100000.0):
    """Create default set of 4 strategy bots for an instrument."""
    return [
        BotConfig(
            instrument=instrument,
            strategy="breakout",
            name=f"Breakout on {instrument}",
            virtual_capital=capital_each,
            params={'lookback': 20, 'volume_factor': 1.5},
        ),
        BotConfig(
            instrument=instrument,
            strategy="mean_reversion",
            name=f"Mean Reversion on {instrument}",
            virtual_capital=capital_each,
            params={'bb_period': 20, 'bb_std': 2.0, 'rsi_period': 14},
        ),
        BotConfig(
            instrument=instrument,
            strategy="ichimoku",
            name=f"Ichimoku on {instrument}",
            virtual_capital=capital_each,
            params={'conversion': 9, 'base': 26, 'span_b': 52},
        ),
        BotConfig(
            instrument=instrument,
            strategy="ml_forecast",
            name=f"ML Forecast on {instrument}",
            virtual_capital=capital_each,
            params={'lookback': 60, 'confidence_threshold': 0.65},
        ),
    ]

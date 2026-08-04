"""
V3 Strategy Engine — drop-in replacement for strategy_engine.py
================================================================

NEW FILE — strategy_engine.py is untouched. This engine serves the COMBINED
registry (all 5 legacy v2 strategies + 3 new v3 quant strategies), so every
existing bot config keeps working and the new quant strategies become
selectable everywhere the old ones were.

Also fixes a latent bug: api_server.py's backtest endpoint calls
`strategy_engine.get_signal(...)`, which never existed on the old engine
(only `analyze`). Here `get_signal` is provided as a proper alias.
"""

from typing import Dict, Optional

import pandas as pd
from datetime import datetime
from loguru import logger

# Reuse the old engine's implementation (indicators etc.) via inheritance —
# zero duplication, old file stays the single source for legacy behavior.
from shared.logic.strategies.strategy_engine import StrategyEngine
from shared.logic.strategies.v3_quant_strategies import (
    REGISTRY, Signal, resolve_sensitivity)


class StrategyEngineV3(StrategyEngine):
    """Multi-strategy engine over the combined v2 + v3 registry."""

    def __init__(self, min_confluence: int = 2):
        super().__init__(min_confluence=min_confluence)
        self.registry = REGISTRY
        self.strategies = [s['id'] for s in REGISTRY]
        logger.info(
            f"✅ StrategyEngineV3 initialized with {len(self.strategies)} "
            f"strategies: {self.strategies}")

    def analyze(self, df: pd.DataFrame, strategy: str = 'quant_alpha',
                settings: Optional[Dict] = None,
                sensitivity: str = 'conservative') -> Signal:
        if df.empty or len(df) < 50:
            return Signal('none', 'HOLD', 0.0, 0.0,
                          ['Insufficient data'], datetime.now())

        current_price = float(df['close'].iloc[-1])
        strategy_def = next(
            (s for s in self.registry if s['id'] == strategy), None)

        if strategy_def is None:
            logger.warning(f"⚠️ Unknown strategy requested: {strategy}")
            return Signal(strategy, 'HOLD', 0.0, current_price,
                          ['Unknown strategy'], datetime.now())

        # Per-bot Signal Sensitivity → entry-gate thresholds (edge_min, cost_mult,
        # ml_conf). Strategies read these from **kwargs['params'] and ignore them
        # if they don't apply.
        params = resolve_sensitivity(sensitivity)

        try:
            return strategy_def['logic'](df, current_price,
                                         self.min_confluence, params=params)
        except Exception as e:
            logger.error(f"❌ Error executing strategy {strategy}: {e}")
            return Signal(strategy, 'HOLD', 0.0, current_price,
                          [f"Error: {str(e)}"], datetime.now())

    # api_server.py's backtest endpoint expects this name
    def get_signal(self, df: pd.DataFrame, strategy: str = 'quant_alpha',
                   settings: Optional[Dict] = None,
                   sensitivity: str = 'conservative') -> Signal:
        return self.analyze(df, strategy, settings, sensitivity)


def get_strategy_engine(min_confluence: int = 2) -> StrategyEngineV3:
    """Factory — same signature as the old strategy_engine factory."""
    return StrategyEngineV3(min_confluence=min_confluence)

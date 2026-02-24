"""
GodBot Trade – Quantitative Paper Trading Framework
====================================================

Institutional-grade multi-bot paper trading simulation.
PAPER_MODE is enforced — no live execution possible.
"""

from enum import Enum

# ══════════════════════════════════════════════════════════════
# HARD SAFETY FLAG — DO NOT CHANGE
# ══════════════════════════════════════════════════════════════
PAPER_MODE = True

class RunMode(Enum):
    """System run mode."""
    BACKTEST = "backtest"
    FORWARD_PAPER = "forward_paper"

# Block broker API imports when PAPER_MODE is active
if PAPER_MODE:
    import sys
    class _BrokerBlocker:
        """Prevents accidental import of live broker modules."""
        _BLOCKED = frozenset({
            'alpaca_trade_api',
            'kiteconnect',
            'ccxt',
        })
        def find_module(self, name, path=None):
            if name in self._BLOCKED:
                return self
            return None
        def load_module(self, name):
            raise ImportError(
                f"🚫 PAPER_MODE is active. Import of live broker '{name}' is BLOCKED. "
                f"Disable PAPER_MODE only after 3+ months positive expectancy."
            )
    sys.meta_path.insert(0, _BrokerBlocker())

__version__ = "1.0.0"
__all__ = ["PAPER_MODE", "RunMode"]

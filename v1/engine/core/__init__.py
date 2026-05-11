"""
Trading Engine Module
=====================

Automated trading engines for different strategies.
"""

from .ichimoku_auto_trader import IchimokuAutoTrader, IchimokuSignal, TradeRecord, TradingState

__all__ = ['IchimokuAutoTrader', 'IchimokuSignal', 'TradeRecord', 'TradingState']

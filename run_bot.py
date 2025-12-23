#!/usr/bin/env python
"""
Trading Bot - Entry Point
=========================

Simple entry point for running the trading bot.

Usage:
    python run_bot.py                     # Run backtest with defaults
    python run_bot.py --mode paper        # Paper trading
    python run_bot.py --symbols AAPL MSFT # Specific symbols
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.main import TradingBot, main


if __name__ == "__main__":
    main()

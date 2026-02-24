"""
Configuration settings for the Systematic Trading Engine.
All parameters are centralized here to avoid magic numbers.
"""

from typing import List, Dict

# 1. RISK PARAMETERS
RISK_PER_TRADE = 0.01  # 1% risk per trade
FRACTIONAL_KELLY_CAP = 0.25  # Max 25% of Kelly Criterion
DAILY_LOSS_LIMIT_R = 3.0  # Stop trading after 3R loss in a day
WEEKLY_DRAWDOWN_PAUSE = 0.05  # Pause trading if weekly drawdown > 5%
CONSECUTIVE_LOSS_COOLDOWN = 5  # Pause after 5 consecutive losses
EQUITY_MA_PERIOD = 20  # SMA period for equity curve filter

# 2. STRATEGY PARAMETERS
REWARD_RISK_RATIO = 3.0  # Default 1:3 R:R (can be 1.5 for 2:3)
ATR_PERIOD = 14
STOP_ATR_MULTIPLIER = 2.0  # Stop loss at 2 * ATR
SIGNAL_SCORE_THRESHOLD = 7.0  # Minimum score (0-10) to enter trade

# 3. REGIME PARAMETERS
ADX_TREND_THRESHOLD = 25
ATR_VOLATILITY_PERCENTILE = 0.7  # High volatility threshold
VOLUME_LIQUIDITY_PERCENTILE = 0.3  # Low liquidity threshold

# 4. TRADING ASSETS
TRADING_ASSETS = ["BTC/USDT", "ETH/USDT", "SOL/USDT"]
MAX_TOTAL_OPEN_RISK = 0.04  # Max 4% total portfolio risk open at once
CORRELATION_THRESHOLD = 0.8  # Reduce exposure if correlation > 0.8

# 5. EXECUTION & BACKTESTING
DEFAULT_SLIPPAGE_BPS = 5.0  # 5 basis points
DEFAULT_TRANSACTION_FEE = 0.001  # 0.1% fee
DEFAULT_SPREAD_BPS = 2.0  # 2 basis points

BACKTEST_INITIAL_CAPITAL = 100000.0
BACKTEST_DATA_PATH = "data/historical/"

# 6. SYSTEM MODES
PAPER_TRADING = True
LOG_LEVEL = "INFO"

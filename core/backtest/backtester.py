import pandas as pd
from typing import List, Dict, Any, Optional
from core.data.data_loader import DataLoader
from core.regime.regime_detector import RegimeDetector
from core.signal.signal_engine import SignalEngine
from core.risk.position_sizer import PositionSizer
from core.risk.risk_manager import RiskManager
from core.portfolio.portfolio_manager import PortfolioManager
from config import settings

class Backtester:
    """
    Event-driven simulation of the trading strategy.
    Ensures no lookahead bias by shifting data.
    """

    def __init__(self, initial_capital: float = settings.BACKTEST_INITIAL_CAPITAL):
        self.initial_capital = initial_capital
        self.equity = initial_capital
        self.high_water_mark = initial_capital
        self.trade_log = []
        
        # Components
        self.loader = DataLoader()
        self.regime_detector = RegimeDetector()
        self.signal_engine = SignalEngine()
        self.risk_manager = RiskManager()
        self.portfolio_manager = PortfolioManager()

    def run(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Runs the backtest simulation over the provided dataframe."""
        
        # 0. Pre-calculate Indicators
        df = self.regime_detector.add_indicators(df)
        df = self.signal_engine.calculate_indicators(df)
        
        # We process bar by bar
        for i in range(100, len(df)): # Start later to allow for MA/ATR warmup
            current_bar = df.iloc[i]
            
            # 1. Check Regime
            regime = self.regime_detector.detect(df, i)
            
            # 2. Check Risk Circuit Breaker
            if not self.risk_manager.can_trade(self.equity, self.high_water_mark):
                continue

            # 3. Generate Signal
            signal = self.signal_engine.get_signal(df, i, regime)
            
            if signal['type'] != 'NONE':
                # Sizing
                size = PositionSizer.calculate_size(self.equity, signal['entry_price'], signal['stop_loss'])
                
                if size > 0:
                    # Apply Slippage and Spread to Entry
                    entry_price = signal['entry_price'] + (settings.DEFAULT_SLIPPAGE_BPS + settings.DEFAULT_SPREAD_BPS/2) * 0.0001 * signal['entry_price'] if signal['type'] == 'LONG' else \
                                  signal['entry_price'] - (settings.DEFAULT_SLIPPAGE_BPS + settings.DEFAULT_SPREAD_BPS/2) * 0.0001 * signal['entry_price']
                    
                    # Simulate Outcome (Using next N bars to find SL/TP or exit)
                    # For simplicity in this backtester, we check the current/future bars to see which is hit first
                    outcome = self._simulate_trade_outcome(df.iloc[i:], signal, entry_price, size)
                    
                    if outcome:
                        self.equity += outcome['pnl']
                        self.high_water_mark = max(self.high_water_mark, self.equity)
                        
                        outcome['equity'] = self.equity
                        self.trade_log.append(outcome)
                        self.risk_manager.update_metrics(outcome)

        return pd.DataFrame(self.trade_log)

    def _simulate_trade_outcome(self, future_bars: pd.DataFrame, signal: Dict[str, Any], entry_price: float, size: float) -> Optional[Dict[str, Any]]:
        """Simulates the trade until SL, TP, or timeout is hit."""
        sl = signal['stop_loss']
        tp = signal['take_profit']
        
        for idx, bar in future_bars.iterrows():
            # Check LONG
            if signal['type'] == 'LONG':
                if bar['low'] <= sl: # Hit SL
                    exit_price = sl - (settings.DEFAULT_SLIPPAGE_BPS * 0.0001 * sl)
                    pnl = (exit_price - entry_price) * size - (entry_price * size * settings.DEFAULT_TRANSACTION_FEE)
                    return {"symbol": "TEST", "side": "LONG", "entry": entry_price, "exit": exit_price, "pnl": pnl, "pnl_r": -1.0, "pnl_pct": pnl/self.equity}
                if bar['high'] >= tp: # Hit TP
                    exit_price = tp - (settings.DEFAULT_SLIPPAGE_BPS * 0.0001 * tp)
                    pnl = (exit_price - entry_price) * size - (entry_price * size * settings.DEFAULT_TRANSACTION_FEE)
                    return {"symbol": "TEST", "side": "LONG", "entry": entry_price, "exit": exit_price, "pnl": pnl, "pnl_r": settings.REWARD_RISK_RATIO, "pnl_pct": pnl/self.equity}
            
            # Check SHORT
            if signal['type'] == 'SHORT':
                if bar['high'] >= sl: # Hit SL
                    exit_price = sl + (settings.DEFAULT_SLIPPAGE_BPS * 0.0001 * sl)
                    pnl = (entry_price - exit_price) * size - (entry_price * size * settings.DEFAULT_TRANSACTION_FEE)
                    return {"symbol": "TEST", "side": "SHORT", "entry": entry_price, "exit": exit_price, "pnl": pnl, "pnl_r": -1.0, "pnl_pct": pnl/self.equity}
                if bar['low'] <= tp: # Hit TP
                    exit_price = tp + (settings.DEFAULT_SLIPPAGE_BPS * 0.0001 * tp)
                    pnl = (entry_price - exit_price) * size - (entry_price * size * settings.DEFAULT_TRANSACTION_FEE)
                    return {"symbol": "TEST", "side": "SHORT", "entry": entry_price, "exit": exit_price, "pnl": pnl, "pnl_r": settings.REWARD_RISK_RATIO, "pnl_pct": pnl/self.equity}
                    
        return None

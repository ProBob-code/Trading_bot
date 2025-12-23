"""
Intraday Paper Trading Simulation
=================================

Simulates a full trading day with automatic buy/sell based on strategy signals.
Uses 5-minute or 30-minute candle data.

Usage:
    python intraday_simulation.py
    
Features:
- Simulates trading through each candle of the day
- Automatic entry on signal
- Automatic exit on stop-loss, take-profit, or EOD
- Real-time P&L tracking
- Support for multiple stocks

Author: Trading Bot
"""

import sys
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from enum import Enum

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.indicators.technical import TechnicalIndicators
from src.indicators.custom import CustomIndicators


class PositionSide(Enum):
    LONG = "long"
    SHORT = "short"


@dataclass
class IntradayPosition:
    """Represents an open intraday position."""
    symbol: str
    side: PositionSide
    entry_price: float
    quantity: int
    stop_loss: float
    take_profit: float
    entry_time: datetime
    exit_price: Optional[float] = None
    exit_time: Optional[datetime] = None
    pnl: float = 0.0
    exit_reason: str = ""


@dataclass
class IntradayTrader:
    """Intraday paper trading engine."""
    capital: float = 10000.0
    max_position_pct: float = 50.0  # Max 50% of capital per trade
    max_positions: int = 2
    
    positions: List[IntradayPosition] = field(default_factory=list)
    closed_positions: List[IntradayPosition] = field(default_factory=list)
    cash: float = 0.0
    
    # Intraday settings
    market_open: str = "09:15"  # NSE market open
    market_close: str = "15:30"  # NSE market close
    square_off_time: str = "15:15"  # Square off 15 min before close
    
    def __post_init__(self):
        self.cash = self.capital
    
    def can_open_position(self) -> bool:
        """Check if we can open a new position."""
        return len(self.positions) < self.max_positions
    
    def get_position_size(self, price: float) -> int:
        """Calculate position size based on capital."""
        max_value = self.cash * (self.max_position_pct / 100)
        return int(max_value / price)
    
    def open_position(
        self,
        symbol: str,
        side: PositionSide,
        price: float,
        stop_loss: float,
        take_profit: float,
        timestamp: datetime
    ) -> Optional[IntradayPosition]:
        """Open a new position."""
        if not self.can_open_position():
            return None
        
        quantity = self.get_position_size(price)
        if quantity <= 0:
            return None
        
        cost = quantity * price
        if cost > self.cash:
            return None
        
        self.cash -= cost
        
        position = IntradayPosition(
            symbol=symbol,
            side=side,
            entry_price=price,
            quantity=quantity,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_time=timestamp
        )
        self.positions.append(position)
        
        return position
    
    def close_position(
        self,
        position: IntradayPosition,
        price: float,
        timestamp: datetime,
        reason: str
    ):
        """Close a position and calculate P&L."""
        if position.side == PositionSide.LONG:
            pnl = (price - position.entry_price) * position.quantity
        else:  # SHORT
            pnl = (position.entry_price - price) * position.quantity
        
        position.exit_price = price
        position.exit_time = timestamp
        position.pnl = pnl
        position.exit_reason = reason
        
        self.cash += (position.quantity * price) + pnl
        
        self.positions.remove(position)
        self.closed_positions.append(position)
        
        return pnl
    
    def check_stops(self, symbol: str, high: float, low: float, timestamp: datetime):
        """Check if any position hit stop-loss or take-profit."""
        for position in self.positions[:]:  # Copy list to allow modification
            if position.symbol != symbol:
                continue
            
            if position.side == PositionSide.LONG:
                # Check stop-loss (price went below SL)
                if low <= position.stop_loss:
                    self.close_position(position, position.stop_loss, timestamp, "STOP_LOSS")
                    print(f"      [SL HIT] {symbol} LONG closed @ Rs.{position.stop_loss:.2f}")
                # Check take-profit (price went above TP)
                elif high >= position.take_profit:
                    self.close_position(position, position.take_profit, timestamp, "TAKE_PROFIT")
                    print(f"      [TP HIT] {symbol} LONG closed @ Rs.{position.take_profit:.2f}")
            else:  # SHORT
                # Check stop-loss (price went above SL)
                if high >= position.stop_loss:
                    self.close_position(position, position.stop_loss, timestamp, "STOP_LOSS")
                    print(f"      [SL HIT] {symbol} SHORT closed @ Rs.{position.stop_loss:.2f}")
                # Check take-profit (price went below TP)
                elif low <= position.take_profit:
                    self.close_position(position, position.take_profit, timestamp, "TAKE_PROFIT")
                    print(f"      [TP HIT] {symbol} SHORT closed @ Rs.{position.take_profit:.2f}")
    
    def square_off_all(self, prices: Dict[str, float], timestamp: datetime):
        """Square off all positions at end of day."""
        for position in self.positions[:]:
            price = prices.get(position.symbol, position.entry_price)
            self.close_position(position, price, timestamp, "EOD_SQUAREOFF")
            print(f"      [EOD] {position.symbol} {position.side.value} closed @ Rs.{price:.2f}")
    
    def get_total_pnl(self) -> float:
        """Get total P&L from all closed positions."""
        return sum(p.pnl for p in self.closed_positions)
    
    def get_equity(self, prices: Dict[str, float]) -> float:
        """Get current equity (cash + open positions value)."""
        equity = self.cash
        for position in self.positions:
            price = prices.get(position.symbol, position.entry_price)
            if position.side == PositionSide.LONG:
                unrealized = (price - position.entry_price) * position.quantity
            else:
                unrealized = (position.entry_price - price) * position.quantity
            equity += unrealized
        return equity


class IntradayStrategy:
    """Fast intraday strategy for 5-min/30-min charts."""
    
    def __init__(
        self,
        ema_fast: int = 9,
        ema_slow: int = 21,
        rsi_period: int = 14,
        rsi_oversold: float = 30,
        rsi_overbought: float = 70,
        atr_period: int = 14,
        atr_multiplier: float = 1.5
    ):
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
    
    def add_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add indicators needed for intraday trading."""
        df = df.copy()
        
        # EMAs for trend
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        
        # RSI for momentum
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(self.rsi_period).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR for stops
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(self.atr_period).mean()
        
        # VWAP
        df['cum_vol'] = df['volume'].cumsum()
        df['cum_vol_price'] = (df['volume'] * (df['high'] + df['low'] + df['close']) / 3).cumsum()
        df['vwap'] = df['cum_vol_price'] / df['cum_vol']
        
        # Trend direction
        df['trend'] = (df['ema_fast'] > df['ema_slow']).astype(int) - (df['ema_fast'] < df['ema_slow']).astype(int)
        
        return df
    
    def generate_signal(self, df: pd.DataFrame, idx: int) -> Optional[Dict]:
        """Generate signal for current bar."""
        if idx < max(self.ema_slow, self.rsi_period, self.atr_period) + 1:
            return None
        
        current = df.iloc[idx]
        prev = df.iloc[idx - 1]
        
        close = current['close']
        ema_fast = current['ema_fast']
        ema_slow = current['ema_slow']
        rsi = current['rsi']
        atr = current['atr']
        
        signal = None
        
        # BUY Signal: EMA crossover up (relaxed conditions)
        if prev['ema_fast'] <= prev['ema_slow'] and ema_fast > ema_slow:
            # Only require reasonable RSI (not overbought)
            if rsi < 75:
                stop_loss = close - (atr * self.atr_multiplier)
                take_profit = close + (atr * self.atr_multiplier * 2)  # 2:1 R:R
                
                signal = {
                    'side': PositionSide.LONG,
                    'price': close,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': f"EMA crossover BUY, RSI={rsi:.1f}"
                }
        
        # SELL Signal: EMA crossover down (relaxed conditions)
        elif prev['ema_fast'] >= prev['ema_slow'] and ema_fast < ema_slow:
            # Only require reasonable RSI (not oversold)
            if rsi > 25:
                stop_loss = close + (atr * self.atr_multiplier)
                take_profit = close - (atr * self.atr_multiplier * 2)  # 2:1 R:R
                
                signal = {
                    'side': PositionSide.SHORT,
                    'price': close,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'reason': f"EMA crossover SELL, RSI={rsi:.1f}"
                }
        
        return signal


def load_intraday_data(file_path: str, symbol: str) -> pd.DataFrame:
    """Load intraday data from Excel file."""
    try:
        # Read raw to find header
        raw = pd.read_excel(file_path, header=None)
        
        # Find the row with "Exchange Date" or similar
        header_row = None
        for i in range(50):
            row = raw.iloc[i].astype(str).str.lower()
            if any('date' in str(val) for val in row.values):
                header_row = i
                break
        
        if header_row is None:
            header_row = 29  # Default
        
        # Load with header
        df = pd.read_excel(file_path, header=header_row)
        
        # Find datetime column
        date_col = None
        for col in df.columns:
            col_str = str(col).lower()
            if 'date' in col_str or 'time' in col_str:
                date_col = col
                break
        
        if date_col is None:
            date_col = df.columns[0]
        
        # Rename columns
        col_mapping = {}
        for col in df.columns:
            col_lower = str(col).lower()
            if 'open' in col_lower:
                col_mapping[col] = 'open'
            elif 'high' in col_lower:
                col_mapping[col] = 'high'
            elif 'low' in col_lower:
                col_mapping[col] = 'low'
            elif 'close' in col_lower:
                col_mapping[col] = 'close'
            elif 'vol' in col_lower:
                col_mapping[col] = 'volume'
        
        col_mapping[date_col] = 'datetime'
        df.rename(columns=col_mapping, inplace=True)
        
        # Convert datetime
        df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
        df = df.dropna(subset=['datetime'])
        df.set_index('datetime', inplace=True)
        df = df.sort_index()
        
        # Convert numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        df = df[['open', 'high', 'low', 'close', 'volume']].dropna()
        
        print(f"  [{symbol}] Loaded {len(df)} candles ({df.index.min()} to {df.index.max()})")
        return df
        
    except Exception as e:
        print(f"  [{symbol}] Error loading: {e}")
        return pd.DataFrame()


def run_intraday_simulation(
    stocks: Dict[str, str],
    capital: float = 10000,
    simulation_date: Optional[str] = None
):
    """
    Run intraday paper trading simulation.
    
    Args:
        stocks: Dict of {symbol: file_path}
        capital: Starting capital
        simulation_date: Date to simulate (YYYY-MM-DD), or None for latest
    """
    print("\n" + "=" * 70)
    print("  INTRADAY PAPER TRADING SIMULATION")
    print("=" * 70)
    print(f"\nStarting Capital: Rs.{capital:,.2f}")
    print(f"Stocks: {', '.join(stocks.keys())}")
    print("-" * 70)
    
    # Initialize trader and strategy
    trader = IntradayTrader(capital=capital, max_positions=len(stocks))
    strategy = IntradayStrategy()
    
    # Load all stock data
    stock_data = {}
    for symbol, file_path in stocks.items():
        print(f"\nLoading {symbol}...")
        df = load_intraday_data(file_path, symbol)
        if not df.empty:
            df = strategy.add_indicators(df)
            stock_data[symbol] = df
    
    if not stock_data:
        print("\n[ERROR] No data loaded!")
        return
    
    # Find the latest common date
    all_dates = set()
    for symbol, df in stock_data.items():
        dates = df.index.normalize().unique()
        all_dates.update(dates)
    
    if simulation_date:
        sim_date = pd.Timestamp(simulation_date)
    else:
        # Use the latest date that has data
        sim_date = max(all_dates)
    
    print(f"\n{'=' * 70}")
    print(f"  SIMULATION DATE: {sim_date.date()}")
    print(f"{'=' * 70}")
    
    # Filter data for simulation date
    day_data = {}
    for symbol, df in stock_data.items():
        day_df = df[df.index.normalize() == sim_date]
        if not day_df.empty:
            day_data[symbol] = day_df
            print(f"  [{symbol}] {len(day_df)} candles for {sim_date.date()}")
    
    if not day_data:
        print(f"\n[ERROR] No data for date {sim_date.date()}")
        return
    
    # Get all timestamps for the day
    all_timestamps = set()
    for df in day_data.values():
        all_timestamps.update(df.index)
    sorted_timestamps = sorted(all_timestamps)
    
    print(f"\n  Trading from {sorted_timestamps[0].time()} to {sorted_timestamps[-1].time()}")
    print(f"  Total candles: {len(sorted_timestamps)}")
    print("-" * 70)
    
    # Simulate trading through each candle
    trade_count = 0
    
    for timestamp in sorted_timestamps:
        time_str = timestamp.strftime("%H:%M")
        
        # Get current prices
        current_prices = {}
        for symbol, df in day_data.items():
            if timestamp in df.index:
                current_prices[symbol] = df.loc[timestamp, 'close']
        
        # Check stops for open positions
        for symbol, df in day_data.items():
            if timestamp in df.index:
                bar = df.loc[timestamp]
                trader.check_stops(symbol, bar['high'], bar['low'], timestamp)
        
        # Check for new signals
        for symbol, df in day_data.items():
            if timestamp not in df.index:
                continue
            
            # Get index of current bar
            idx = df.index.get_loc(timestamp)
            
            # Skip if we already have position in this symbol
            if any(p.symbol == symbol for p in trader.positions):
                continue
            
            # Generate signal
            signal = strategy.generate_signal(df, idx)
            
            if signal and trader.can_open_position():
                position = trader.open_position(
                    symbol=symbol,
                    side=signal['side'],
                    price=signal['price'],
                    stop_loss=signal['stop_loss'],
                    take_profit=signal['take_profit'],
                    timestamp=timestamp
                )
                
                if position:
                    trade_count += 1
                    side_str = "BUY" if signal['side'] == PositionSide.LONG else "SELL"
                    print(f"[{time_str}] [{symbol}] {side_str} {position.quantity} @ Rs.{position.entry_price:.2f}")
                    print(f"          SL: Rs.{position.stop_loss:.2f} | TP: Rs.{position.take_profit:.2f}")
                    print(f"          Reason: {signal['reason']}")
    
    # Square off remaining positions at end of day
    if trader.positions:
        print(f"\n[15:30] SQUARING OFF {len(trader.positions)} OPEN POSITIONS")
        final_prices = {s: df.iloc[-1]['close'] for s, df in day_data.items()}
        trader.square_off_all(final_prices, sorted_timestamps[-1])
    
    # Print summary
    print("\n" + "=" * 70)
    print("  END OF DAY SUMMARY")
    print("=" * 70)
    
    total_pnl = trader.get_total_pnl()
    final_equity = trader.capital + total_pnl
    return_pct = (total_pnl / capital) * 100
    
    print(f"\n  Starting Capital: Rs.{capital:,.2f}")
    print(f"  Final Equity:     Rs.{final_equity:,.2f}")
    print(f"  P&L:              Rs.{total_pnl:+,.2f} ({return_pct:+.2f}%)")
    print(f"  Total Trades:     {len(trader.closed_positions)}")
    
    if trader.closed_positions:
        wins = sum(1 for p in trader.closed_positions if p.pnl > 0)
        losses = len(trader.closed_positions) - wins
        print(f"  Wins/Losses:      {wins}/{losses}")
        print(f"  Win Rate:         {wins/len(trader.closed_positions)*100:.1f}%")
        
        print("\n  TRADE LOG:")
        print("-" * 70)
        for p in trader.closed_positions:
            side = "LONG" if p.side == PositionSide.LONG else "SHORT"
            pnl_str = f"Rs.{p.pnl:+,.2f}"
            print(f"  {p.symbol}: {side} {p.quantity} @ Rs.{p.entry_price:.2f} -> Rs.{p.exit_price:.2f} | {p.exit_reason:12} | {pnl_str}")
    
    print("\n" + "=" * 70)
    
    return trader


# ============================================================
# CONFIGURATION - EDIT THESE
# ============================================================

# Available stocks (add your Excel files here)
AVAILABLE_STOCKS = {
    'ADANIENT_5m': 'stock_data/Adani enterprise 5 min.xlsx',
    'ADANIENT_30m': 'stock_data/Adani enterprise 30 min.xlsx',
    'ASIANPAINT_5m': 'stock_data/Asian paints 5 min.xlsx',
    'ASIANPAINT_30m': 'stock_data/Asian paints 30 min.xlsx',
}

# Default stocks to trade (pick from AVAILABLE_STOCKS)
DEFAULT_STOCKS = {
    'ADANIENT': 'stock_data/Adani enterprise 5 min.xlsx',
    'ASIANPAINT': 'stock_data/Asian paints 5 min.xlsx',
}


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Intraday Paper Trading Simulation")
    parser.add_argument('--capital', type=float, default=10000, help='Starting capital (default: 10000)')
    parser.add_argument('--date', type=str, default=None, help='Simulation date (YYYY-MM-DD)')
    parser.add_argument('--interval', type=str, default='5m', choices=['5m', '30m'], help='Candle interval')
    
    args = parser.parse_args()
    
    # Select stocks based on interval
    if args.interval == '30m':
        stocks = {
            'ADANIENT': 'stock_data/Adani enterprise 30 min.xlsx',
            'ASIANPAINT': 'stock_data/Asian paints 30 min.xlsx',
        }
    else:
        stocks = DEFAULT_STOCKS
    
    run_intraday_simulation(stocks, args.capital, args.date)

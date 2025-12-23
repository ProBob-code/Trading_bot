"""
Paper Trading Script
====================

Simulates live trading with the strategy.
Checks for signals periodically and executes paper trades.

Usage:
    python paper_trade.py
"""

import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.historical_loader import HistoricalLoader
from src.strategy.ta_strategy import TAStrategy
from src.risk.position_sizer import PositionSizer
from src.risk.risk_manager import RiskManager
from src.execution.order_manager import OrderManager
from src.execution.brokers.paper_trader import PaperTrader


# Configuration
CAPITAL = 10000  # Rs.10,000 starting capital
POSITION_SIZE_PCT = 50  # Use 50% of capital per trade
MAX_POSITIONS = 2  # Max concurrent positions

# Stock files to monitor
STOCKS = {
    'ADANIENT': {
        'file': 'stock_data/Adani enterprise annual.xlsx',
        'header_row': 29,
        'date_column': 'Exchange Date'
    },
    'ASIANPAINT': {
        'file': 'stock_data/Asian Paints Annual.xlsx',
        'header_row': 30,  # Different header row
        'date_column': None  # Will use first column
    }
}


def load_stock_data(symbol: str, config: dict):
    """Load stock data from Excel file."""
    import pandas as pd
    
    loader = HistoricalLoader()
    
    try:
        df = loader.load_excel(
            config['file'],
            header_row=config['header_row'],
            date_column=config.get('date_column', 'Exchange Date')
        )
        return df
    except Exception as e:
        print(f"[WARN] Could not load {symbol}: {e}")
        return None


def run_paper_trading():
    """Run paper trading simulation."""
    print("\n" + "=" * 60)
    print("  PAPER TRADING SESSION")
    print("=" * 60)
    print(f"\nStarting Capital: Rs.{CAPITAL:,.2f}")
    print(f"Position Size: {POSITION_SIZE_PCT}%")
    print(f"Max Positions: {MAX_POSITIONS}")
    print("-" * 60)
    
    # Initialize components
    broker = PaperTrader(initial_capital=CAPITAL)
    order_manager = OrderManager(broker)
    broker.set_order_manager(order_manager)
    
    risk_manager = RiskManager(
        initial_capital=CAPITAL,
        max_daily_loss_pct=5.0,
        max_drawdown_pct=15.0,
        max_positions=MAX_POSITIONS
    )
    
    position_sizer = PositionSizer(
        method="fixed_fractional",
        max_risk_pct=2.0
    )
    
    strategy = TAStrategy(
        ma_fast=9,
        ma_slow=21,
        ma_signal=44,
        min_confluence=3
    )
    
    print("\n[INFO] Loading stock data and checking signals...\n")
    
    signals_found = []
    
    for symbol, config in STOCKS.items():
        df = load_stock_data(symbol, config)
        
        if df is None or df.empty:
            print(f"[{symbol}] No data available")
            continue
        
        # Get latest price
        latest = df.iloc[-1]
        current_price = latest['close']
        
        # Set price for paper trader
        broker.set_prices({symbol: current_price})
        
        # Generate signal
        signal = strategy.get_latest_signal(df, symbol)
        
        if signal:
            signals_found.append({
                'symbol': symbol,
                'signal': signal,
                'price': current_price
            })
            
            print(f"[{symbol}] SIGNAL: {signal.signal_type.value.upper()}")
            print(f"         Price: Rs.{current_price:.2f}")
            print(f"         Confidence: {signal.confidence:.0%}")
            print(f"         Stop Loss: Rs.{signal.stop_loss:.2f}")
            print(f"         Take Profit: Rs.{signal.take_profit:.2f}")
            print(f"         Reason: {signal.reason[:100]}...")
            
            # Check if we can open position
            position_size = CAPITAL * (POSITION_SIZE_PCT / 100)
            can_trade, reason = risk_manager.can_open_position(symbol, position_size)
            
            if can_trade:
                # Calculate position size
                sizing = position_sizer.calculate_position_size(
                    capital=broker.get_account_info()['cash'],
                    entry_price=current_price,
                    stop_loss=signal.stop_loss
                )
                
                units = int(sizing['units'])
                if units > 0:
                    # Create and submit order
                    side = 'sell' if signal.signal_type.value == 'sell' else 'buy'
                    order = order_manager.create_order(
                        symbol=symbol,
                        side=side,
                        quantity=units,
                        order_type='market'
                    )
                    order_manager.submit_order(order)
                    
                    print(f"         [ORDER] {side.upper()} {units} @ Rs.{current_price:.2f}")
            else:
                print(f"         [SKIP] Cannot trade: {reason}")
        else:
            print(f"[{symbol}] No signal (neutral)")
        
        print()
    
    # Show account summary
    print("-" * 60)
    print("\nACCOUNT SUMMARY")
    print("-" * 60)
    
    account = broker.get_account_info()
    print(f"Cash: Rs.{account['cash']:,.2f}")
    print(f"Positions Value: Rs.{account['positions_value']:,.2f}")
    print(f"Total Value: Rs.{account['total_value']:,.2f}")
    
    pnl = account['total_value'] - CAPITAL
    pnl_pct = (pnl / CAPITAL) * 100
    print(f"P&L: Rs.{pnl:+,.2f} ({pnl_pct:+.2f}%)")
    
    positions = broker.get_positions()
    if positions:
        print("\nOpen Positions:")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} @ Rs.{pos['avg_price']:.2f}")
    
    print("\n" + "=" * 60)
    return signals_found


if __name__ == "__main__":
    run_paper_trading()

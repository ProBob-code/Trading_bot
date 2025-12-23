"""
Quick Start Demo
================

Demonstrates basic usage of the trading bot components.
Run this to verify everything is working.

Usage:
    python demo.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from datetime import datetime, timedelta
import pandas as pd


# Alpha Vantage API Key (free tier: 25 requests/day)
ALPHA_VANTAGE_API_KEY = "9N8KYMVUSI2VRBOZ"

# Local data files in stock_data folder
# Available: Adani enterprise (30min, 5min, annual), Asian Paints (30min, 5min, annual)
LOCAL_DATA_FILE = "stock_data/Adani enterprise annual.xlsx"
DEMO_SYMBOL = "ADANIENT"  # Adani Enterprises (NIFTY 50)


def demo_data_loading():
    """Demo: Load and display data from local files."""
    print("\n" + "="*60)
    print("1. DATA LOADING DEMO (Local Excel Files)")
    print("="*60)
    
    from src.data.historical_loader import HistoricalLoader
    import os
    
    # Use historical loader for local Excel files
    loader = HistoricalLoader()
    
    # Get the full path to the data file
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, LOCAL_DATA_FILE)
    
    print(f"\nLoading {DEMO_SYMBOL} data from: {LOCAL_DATA_FILE}")
    # Excel file has header at row 29 (0-indexed), with Exchange Date as date column
    df = loader.load_excel(file_path, header_row=29, date_column="Exchange Date")
    
    if not df.empty:
        print(f"[OK] Loaded {len(df)} bars")
        print(f"  Date range: {df.index.min()} to {df.index.max()}")
        print(f"\nLatest data:")
        print(df.tail(3).to_string())
    else:
        print("[FAIL] No data loaded - check file path")
    
    return df


def demo_indicators(df: pd.DataFrame):
    """Demo: Calculate technical indicators."""
    print("\n" + "="*60)
    print("2. TECHNICAL INDICATORS DEMO")
    print("="*60)
    
    from src.indicators.technical import TechnicalIndicators
    from src.indicators.custom import CustomIndicators
    
    # Add all standard indicators
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Add custom indicators
    df = CustomIndicators.add_supertrend(df)
    
    print(f"\n[OK] Added {len(df.columns) - 5} indicators")
    print(f"\nIndicators: {', '.join([c for c in df.columns if c not in ['open','high','low','close','volume']])[:200]}...")
    
    print(f"\nLatest values:")
    latest = df.iloc[-1]
    print(f"  Close: ${latest['close']:.2f}")
    print(f"  RSI: {latest['rsi']:.1f}")
    print(f"  MACD: {latest['macd']:.4f}")
    print(f"  BB Upper/Lower: ${latest['bb_upper']:.2f} / ${latest['bb_lower']:.2f}")
    print(f"  Supertrend Dir: {'Bullish' if latest['supertrend_dir'] == 1 else 'Bearish'}")
    
    return df


def demo_strategy(df: pd.DataFrame):
    """Demo: Generate trading signals."""
    print("\n" + "="*60)
    print("3. STRATEGY & SIGNAL GENERATION DEMO")
    print("="*60)
    
    from src.strategy.ta_strategy import TAStrategy
    
    strategy = TAStrategy(
        ma_fast=9,
        ma_slow=21,
        ma_signal=44,
        min_confluence=3
    )
    
    # Generate signal for Indian stock
    signal = strategy.get_latest_signal(df, DEMO_SYMBOL)
    
    if signal:
        print(f"\n[OK] Signal Generated for {DEMO_SYMBOL}:")
        print(f"  Type: {signal.signal_type.value.upper()}")
        print(f"  Price: Rs.{signal.price:.2f}")
        print(f"  Confidence: {signal.confidence:.0%}")
        print(f"  Stop Loss: Rs.{signal.stop_loss:.2f}" if signal.stop_loss else "")
        print(f"  Take Profit: Rs.{signal.take_profit:.2f}" if signal.take_profit else "")
        print(f"\n  Reason: {signal.reason[:150]}...")
    
    return signal


def demo_risk_management():
    """Demo: Position sizing and risk management."""
    print("\n" + "="*60)
    print("4. RISK MANAGEMENT DEMO")
    print("="*60)
    
    from src.risk.position_sizer import PositionSizer
    from src.risk.risk_manager import RiskManager
    
    # Position sizing
    sizer = PositionSizer(method="fixed_fractional", max_risk_pct=2.0)
    
    capital = 100000
    entry_price = 150.0
    stop_loss = 145.0
    
    sizing = sizer.calculate_position_size(capital, entry_price, stop_loss)
    
    print(f"\nPosition Sizing (Fixed Fractional):")
    print(f"  Capital: ${capital:,.2f}")
    print(f"  Entry: ${entry_price:.2f}, Stop: ${stop_loss:.2f}")
    print(f"  -> Position Size: ${sizing['position_size']:,.2f}")
    print(f"  -> Units: {sizing['units']:.2f}")
    print(f"  -> Risk: ${sizing['risk_amount']:.2f} ({sizing['risk_pct']:.1f}%)")
    
    # Risk manager
    rm = RiskManager(
        initial_capital=capital,
        max_daily_loss_pct=2.0,
        max_drawdown_pct=10.0,
        max_positions=5
    )
    
    print(f"\nRisk Manager Status:")
    allowed, reason = rm.can_open_position("AAPL", sizing['position_size'])
    print(f"  Can open position: {allowed} - {reason}")
    
    return sizing


def demo_paper_trading():
    """Demo: Paper trading execution."""
    print("\n" + "="*60)
    print("5. PAPER TRADING DEMO")
    print("="*60)
    
    from src.execution.order_manager import OrderManager
    from src.execution.brokers.paper_trader import PaperTrader
    
    # Setup
    broker = PaperTrader(initial_capital=100000)
    order_manager = OrderManager(broker)
    broker.set_order_manager(order_manager)
    
    # Set current price
    broker.set_prices({"AAPL": 150.0})
    
    # Create and submit order
    order = order_manager.create_order(
        symbol="AAPL",
        side="buy",
        quantity=10,
        order_type="market"
    )
    
    print(f"\nCreated Order: {order.order_id}")
    order_manager.submit_order(order)
    
    # Check status
    print(f"Order Status: {order.status.value}")
    print(f"Filled: {order.filled_quantity} @ ${order.filled_price:.2f}")
    
    # Check account
    account = broker.get_account_info()
    print(f"\nAccount Status:")
    print(f"  Cash: ${account['cash']:,.2f}")
    print(f"  Positions Value: ${account['positions_value']:,.2f}")
    print(f"  Total Value: ${account['total_value']:,.2f}")
    
    # Show positions
    positions = broker.get_positions()
    if positions:
        print(f"\nPositions:")
        for pos in positions:
            print(f"  {pos['symbol']}: {pos['quantity']} @ Rs.{pos['avg_price']:.2f}")


def demo_ml_features():
    """Demo: ML feature engineering."""
    print("\n" + "="*60)
    print("6. ML FEATURE ENGINEERING DEMO")
    print("="*60)
    
    from src.ml.feature_engineering import FeatureEngineer
    from src.data.historical_loader import HistoricalLoader
    from src.indicators.technical import TechnicalIndicators
    import os
    
    # Load data from local Excel file
    loader = HistoricalLoader()
    base_path = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(base_path, LOCAL_DATA_FILE)
    
    print(f"\nLoading {DEMO_SYMBOL} data for ML features...")
    # Excel file has header at row 29 (0-indexed), with Exchange Date as date column
    df = loader.load_excel(file_path, header_row=29, date_column="Exchange Date")
    
    if df.empty:
        print("No data available - check file path")
        return
    
    # Add indicators first
    df = TechnicalIndicators.add_all_indicators(df)
    
    # Create features
    fe = FeatureEngineer()
    df_features = fe.create_features(df)
    
    print(f"\n[OK] Created {len(fe.get_feature_names())} features")
    print(f"\nSample features:")
    for feat in fe.get_feature_names()[:10]:
        print(f"  • {feat}")
    print("  ...")
    
    # Create target
    df_features = fe.create_target(df_features, horizon=5, target_type='direction')
    
    # Prepare train/test
    X_train, X_test, y_train, y_test = fe.prepare_data(df_features)
    
    print(f"\nData split:")
    print(f"  Training: {len(X_train)} samples")
    print(f"  Testing: {len(X_test)} samples")
    print(f"  Features per sample: {len(X_train.columns)}")


def main():
    """Run all demos."""
    print("\n" + "#"*60)
    print("  TRADING BOT - QUICK START DEMO")
    print("#"*60)
    
    try:
        # Run demos
        df = demo_data_loading()
        
        if df is not None and not df.empty:
            df = demo_indicators(df)
            demo_strategy(df)
        
        demo_risk_management()
        demo_paper_trading()
        demo_ml_features()
        
        print("\n" + "="*60)
        print("[OK] ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("="*60)
        print("\nNext steps:")
        print("  1. Edit config.yaml with your settings")
        print("  2. Run: python run_bot.py --mode backtest")
        print("  3. Review results and tune strategy")
        print("  4. Run: python run_bot.py --mode paper")
        print()
        
    except ImportError as e:
        print(f"\n[FAIL] Missing dependency: {e}")
        print("  Run: pip install -r requirements.txt")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

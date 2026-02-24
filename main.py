import pandas as pd
from core.data.data_loader import DataLoader
from core.backtest.backtester import Backtester
from core.backtest.performance_metrics import PerformanceMetrics
from core.backtest.monte_carlo import MonteCarloEngine
from config import settings

def main():
    print("="*60, flush=True)
    print("SYSTEMATIC TRADING ENGINE - PRODUCTION GRADE", flush=True)
    print("="*60, flush=True)

    # 1. Initialize Components
    loader = DataLoader()
    backtester = Backtester(initial_capital=settings.BACKTEST_INITIAL_CAPITAL)
    metrics_calc = PerformanceMetrics()
    mc_engine = MonteCarloEngine()

    # 2. Generate/Load Data
    print("\n[1/4] Loading Historical Data...", flush=True)
    symbol = "BTC/USDT"
    df = loader.generate_mock_data(symbol, periods=500, freq='1H')
    print(f"Loaded {len(df)} bars for {symbol}", flush=True)

    # 3. Run Backtest
    print("\n[2/4] Running Event-Driven Backtest...", flush=True)
    trade_log = backtester.run(df, symbol)
    
    if trade_log.empty:
        print("No trades generated. Check strategy thresholds.", flush=True)
        return

    # 4. Calculate Performance Metrics
    print("\n[3/4] Calculating Performance Metrics...", flush=True)
    results = metrics_calc.calculate_metrics(trade_log, settings.BACKTEST_INITIAL_CAPITAL)
    
    for metric, value in results.items():
        print(f"  {metric:25}: {value}", flush=True)

    # 5. Run Monte Carlo Simulation
    print("\n[4/4] Running Monte Carlo Risk Analysis...", flush=True)
    returns_r = trade_log['pnl_r'].tolist()
    mc_results = mc_engine.run_simulation(
        returns_r, 
        num_simulations=1000, 
        initial_capital=settings.BACKTEST_INITIAL_CAPITAL,
        risk_per_trade=settings.RISK_PER_TRADE
    )
    
    for metric, value in mc_results.items():
        print(f"  {metric:30}: {value}", flush=True)

    print("\n" + "="*60, flush=True)
    print("SIMULATION COMPLETE", flush=True)
    print("="*60, flush=True)

if __name__ == "__main__":
    main()

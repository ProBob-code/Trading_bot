"""
Trading Bot - Main Entry Point
===============================

Main application that ties together all components.
"""

import sys
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import pandas as pd
from loguru import logger

from .config.settings import get_settings, Settings
from .data.data_provider import get_data_provider, YFinanceProvider
from .data.historical_loader import HistoricalLoader
from .indicators.technical import TechnicalIndicators
from .indicators.custom import CustomIndicators
from .ml.feature_engineering import FeatureEngineer
from .ml.forecasters import XGBoostForecaster
from .ml.model_manager import ModelManager
from .strategy.ta_strategy import TAStrategy
from .strategy.hybrid_strategy import HybridStrategy
from .strategy.base_strategy import Signal, SignalType
from .risk.position_sizer import PositionSizer
from .risk.risk_manager import RiskManager
from .execution.order_manager import OrderManager
from .execution.brokers.paper_trader import PaperTrader


class TradingBot:
    """
    Main trading bot application.
    
    Orchestrates:
    - Data fetching
    - Signal generation
    - Risk management
    - Order execution
    """
    
    def __init__(self, config_path: str = None):
        """
        Initialize trading bot.
        
        Args:
            config_path: Path to configuration file
        """
        self.settings = get_settings(config_path)
        self.is_running = False
        
        # Configure logging
        self._setup_logging()
        
        # Initialize components
        self._init_components()
        
        logger.info(f"Trading Bot initialized in {self.settings.mode} mode")
        
    def _setup_logging(self):
        """Configure logging."""
        logger.remove()  # Remove default handler
        
        # Console logging
        logger.add(
            sys.stdout,
            level=self.settings.log_level,
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{message}</cyan>"
        )
        
        # File logging
        logger.add(
            self.settings.log_file,
            rotation=self.settings.get_raw("logging.rotation", "10 MB"),
            level="DEBUG"
        )
        
    def _init_components(self):
        """Initialize all components."""
        s = self.settings
        
        # Data provider (only initialize if not using local files)
        if s.data_provider == "local":
            self.data_provider = None  # Will use historical_loader instead
        elif s.data_provider == "alphavantage":
            # Get API key from config
            api_key = s.get_raw("data.alphavantage.api_key", "")
            self.data_provider = get_data_provider("alphavantage", api_key=api_key)
        else:
            self.data_provider = get_data_provider(s.data_provider)
        self.historical_loader = HistoricalLoader(s.cache_dir)
        
        # Strategy
        if s.strategy_name == "ta_only":
            self.strategy = TAStrategy(
                ma_fast=s.technical.ma_fast,
                ma_slow=s.technical.ma_slow,
                ma_signal=s.technical.ma_signal,
                rsi_period=s.technical.rsi_period,
                rsi_overbought=s.technical.rsi_overbought,
                rsi_oversold=s.technical.rsi_oversold,
                bb_period=s.technical.bb_period,
                bb_std=s.technical.bb_std,
                atr_period=s.technical.atr_period
            )
        else:
            # Hybrid strategy (default)
            ta_strategy = TAStrategy(
                ma_fast=s.technical.ma_fast,
                ma_slow=s.technical.ma_slow,
                ma_signal=s.technical.ma_signal
            )
            self.strategy = HybridStrategy(ta_strategy=ta_strategy, mode="confirm")
        
        # Risk management
        self.position_sizer = PositionSizer(
            method=s.risk.position_sizing,
            max_position_pct=s.risk.max_position_pct,
            max_risk_pct=s.risk.max_daily_loss_pct
        )
        
        self.risk_manager = RiskManager(
            initial_capital=s.backtest.initial_capital,
            max_daily_loss_pct=s.risk.max_daily_loss_pct,
            max_drawdown_pct=s.risk.max_drawdown_pct,
            max_positions=s.risk.max_positions,
            max_position_pct=s.risk.max_position_pct
        )
        
        # Execution
        if s.mode in ["backtest", "paper"]:
            self.broker = PaperTrader(
                initial_capital=s.backtest.initial_capital,
                slippage_pct=s.backtest.slippage * 100,
                commission_pct=s.backtest.commission * 100
            )
        else:
            # Live trading - would initialize real broker here
            logger.warning("Live trading not fully implemented, using paper trader")
            self.broker = PaperTrader(initial_capital=s.backtest.initial_capital)
        
        self.order_manager = OrderManager(self.broker)
        self.broker.set_order_manager(self.order_manager)
        
        # ML components
        self.model_manager = ModelManager()
        self.feature_engineer = FeatureEngineer()
        
        # Data cache
        self.market_data: Dict[str, pd.DataFrame] = {}
        
    def fetch_data(self, symbols: List[str] = None) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for symbols.
        
        Args:
            symbols: List of symbols (default: from settings)
            
        Returns:
            Dictionary of symbol -> DataFrame
        """
        symbols = symbols or self.settings.symbols
        s = self.settings
        
        logger.info(f"Fetching data for {len(symbols)} symbols...")
        
        # Check if using local files
        if s.data_provider == "local":
            # Load from local Excel files
            local_files = s.get_raw("market.local_data_files", {})
            
            for symbol in symbols:
                try:
                    file_path = local_files.get(symbol)
                    if not file_path:
                        logger.warning(f"  {symbol}: No local file configured")
                        continue
                    
                    # Load using historical loader with correct header row
                    # Most export files have header at row 29
                    df = self.historical_loader.load_excel(
                        file_path,
                        header_row=29,
                        date_column="Exchange Date"
                    )
                    
                    if not df.empty:
                        # Add indicators
                        df = self.strategy.calculate_indicators(df)
                        self.market_data[symbol] = df
                        logger.info(f"  {symbol}: {len(df)} bars loaded from local file")
                    else:
                        logger.warning(f"  {symbol}: No data in file")
                        
                except Exception as e:
                    logger.error(f"  {symbol}: Error loading local file - {e}")
        else:
            # Use data provider (API)
            end_date = datetime.now()
            start_date = end_date - timedelta(days=s.history_days)
            
            for symbol in symbols:
                try:
                    df = self.data_provider.get_historical_data(
                        symbol=symbol,
                        start_date=start_date,
                        end_date=end_date,
                        interval=s.interval
                    )
                    
                    if not df.empty:
                        # Add indicators
                        df = self.strategy.calculate_indicators(df)
                        self.market_data[symbol] = df
                        logger.info(f"  {symbol}: {len(df)} bars loaded")
                    else:
                        logger.warning(f"  {symbol}: No data returned")
                        
                except Exception as e:
                    logger.error(f"  {symbol}: Error - {e}")
        
        return self.market_data
    
    def generate_signals(self) -> Dict[str, Signal]:
        """
        Generate signals for all symbols.
        
        Returns:
            Dictionary of symbol -> Signal
        """
        signals = {}
        
        for symbol, df in self.market_data.items():
            if len(df) < 50:
                continue
                
            signal = self.strategy.get_latest_signal(df, symbol)
            if signal:
                signals[symbol] = signal
                
                if signal.signal_type != SignalType.HOLD:
                    logger.info(
                        f"Signal: {signal.signal_type.value.upper()} {symbol} "
                        f"@ {signal.price:.2f} (conf: {signal.confidence:.2f})"
                    )
        
        return signals
    
    def execute_signals(self, signals: Dict[str, Signal]):
        """
        Execute trading signals.
        
        Args:
            signals: Dictionary of symbol -> Signal
        """
        for symbol, signal in signals.items():
            df = self.market_data.get(symbol)
            if df is None:
                continue
            
            current_price = df['close'].iloc[-1]
            
            # Update broker prices
            self.broker.set_prices({symbol: current_price})
            
            if signal.signal_type == SignalType.BUY:
                self._execute_buy(symbol, signal, current_price, df)
            elif signal.signal_type == SignalType.SELL:
                self._execute_sell(symbol, signal, current_price)
    
    def _execute_buy(
        self,
        symbol: str,
        signal: Signal,
        current_price: float,
        df: pd.DataFrame
    ):
        """Execute a buy signal."""
        # Check if we can open position
        capital = self.risk_manager.current_capital
        
        # Calculate position size
        stop_loss = signal.stop_loss or (current_price * 0.98)
        
        sizing = self.position_sizer.calculate_position_size(
            capital=capital,
            entry_price=current_price,
            stop_loss=stop_loss,
            volatility=df['atr'].iloc[-1] if 'atr' in df.columns else None
        )
        
        position_value = sizing['position_size']
        units = sizing['units']
        
        # Risk check
        allowed, reason = self.risk_manager.can_open_position(symbol, position_value)
        if not allowed:
            logger.warning(f"Cannot open position for {symbol}: {reason}")
            return
        
        # Create and submit order
        order = self.order_manager.create_order(
            symbol=symbol,
            side='buy',
            quantity=units,
            order_type='market',
            stop_loss=stop_loss,
            take_profit=signal.take_profit
        )
        
        if self.order_manager.submit_order(order):
            logger.info(f"Executed BUY: {units:.4f} {symbol} @ ~{current_price:.2f}")
    
    def _execute_sell(self, symbol: str, signal: Signal, current_price: float):
        """Execute a sell signal."""
        # Check if we have a position
        positions = self.broker.get_positions()
        position = next((p for p in positions if p['symbol'] == symbol), None)
        
        if position is None:
            logger.warning(f"No position to sell for {symbol}")
            return
        
        # Create sell order
        order = self.order_manager.create_order(
            symbol=symbol,
            side='sell',
            quantity=position['quantity'],
            order_type='market'
        )
        
        if self.order_manager.submit_order(order):
            logger.info(f"Executed SELL: {position['quantity']:.4f} {symbol} @ ~{current_price:.2f}")
    
    def run_backtest(
        self,
        start_date: str = None,
        end_date: str = None
    ) -> Dict:
        """
        Run backtest on historical data.
        
        Args:
            start_date: Backtest start date (default: from settings)
            end_date: Backtest end date (default: from settings)
            
        Returns:
            Backtest results dictionary
        """
        s = self.settings
        start_date = start_date or s.backtest.start_date
        end_date = end_date or s.backtest.end_date
        
        logger.info(f"Starting backtest: {start_date} to {end_date}")
        
        # Fetch data
        self.fetch_data()
        
        if not self.market_data:
            logger.error("No data available for backtest")
            return {}
        
        # Run through each bar
        all_trades = []
        
        for symbol, df in self.market_data.items():
            logger.info(f"Backtesting {symbol}...")
            
            # Filter by date range
            df_filtered = df.loc[start_date:end_date]
            
            if len(df_filtered) < 50:
                continue
            
            # Simple backtest using strategy's built-in method
            trades = self.strategy.backtest(
                df_filtered,
                initial_capital=s.backtest.initial_capital,
                position_size_pct=s.risk.max_position_pct / 100,
                symbol=symbol
            )
            
            if len(trades) > 0:
                trades['symbol'] = symbol
                all_trades.append(trades)
        
        # Combine results
        if all_trades:
            all_trades_df = pd.concat(all_trades, ignore_index=True)
        else:
            all_trades_df = pd.DataFrame()
        
        # Calculate metrics
        results = self._calculate_backtest_metrics(all_trades_df)
        
        logger.info(f"Backtest complete: {results.get('total_trades', 0)} trades, "
                   f"{results.get('net_pnl', 0):.2f} P&L")
        
        return results
    
    def _calculate_backtest_metrics(self, trades: pd.DataFrame) -> Dict:
        """Calculate performance metrics from trades."""
        if trades.empty:
            return {'total_trades': 0}
        
        total_pnl = trades['pnl'].sum()
        wins = trades[trades['pnl'] > 0]
        losses = trades[trades['pnl'] <= 0]
        
        return {
            'total_trades': len(trades),
            'wins': len(wins),
            'losses': len(losses),
            'win_rate': len(wins) / len(trades) * 100 if len(trades) > 0 else 0,
            'net_pnl': total_pnl,
            'gross_profit': wins['pnl'].sum() if len(wins) > 0 else 0,
            'gross_loss': losses['pnl'].sum() if len(losses) > 0 else 0,
            'avg_win': wins['pnl'].mean() if len(wins) > 0 else 0,
            'avg_loss': losses['pnl'].mean() if len(losses) > 0 else 0,
            'profit_factor': abs(wins['pnl'].sum() / losses['pnl'].sum()) if len(losses) > 0 and losses['pnl'].sum() != 0 else float('inf'),
            'trades': trades.to_dict('records')
        }
    
    def run_live(self, interval_seconds: int = 60):
        """
        Run live/paper trading loop.
        
        Args:
            interval_seconds: Seconds between iterations
        """
        logger.info(f"Starting {'paper' if self.settings.mode == 'paper' else 'live'} trading...")
        
        self.is_running = True
        
        try:
            while self.is_running:
                try:
                    # Fetch latest data
                    self.fetch_data()
                    
                    # Generate signals
                    signals = self.generate_signals()
                    
                    # Execute signals
                    self.execute_signals(signals)
                    
                    # Update positions
                    prices = {s: df['close'].iloc[-1] for s, df in self.market_data.items()}
                    self.broker.set_prices(prices)
                    
                    # Log status
                    account = self.broker.get_account_info()
                    logger.info(f"Portfolio: ${account['total_value']:.2f} "
                               f"(PnL: ${account['pnl']:.2f})")
                    
                except Exception as e:
                    logger.error(f"Error in trading loop: {e}")
                
                # Wait for next iteration
                time.sleep(interval_seconds)
                
        except KeyboardInterrupt:
            logger.info("Trading stopped by user")
            self.stop()
    
    def stop(self):
        """Stop the trading bot."""
        self.is_running = False
        
        # Cancel all pending orders
        self.order_manager.cancel_all_orders()
        
        # Log final status
        account = self.broker.get_account_info()
        logger.info(f"Final portfolio value: ${account['total_value']:.2f}")
        
        logger.info("Trading bot stopped")
    
    def get_status(self) -> Dict:
        """Get current bot status."""
        account = self.broker.get_account_info()
        positions = self.broker.get_positions()
        
        return {
            'mode': self.settings.mode,
            'is_running': self.is_running,
            'account': account,
            'positions': positions,
            'symbols': self.settings.symbols,
            'strategy': self.strategy.name
        }


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Trading Bot')
    parser.add_argument('--mode', choices=['backtest', 'paper', 'live'], 
                       default='backtest', help='Trading mode')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--symbols', nargs='+', help='Symbols to trade')
    
    args = parser.parse_args()
    
    # Initialize bot
    bot = TradingBot(config_path=args.config)
    
    if args.symbols:
        bot.settings.symbols = args.symbols
    
    # Run based on mode
    if args.mode == 'backtest':
        results = bot.run_backtest()
        print("\n=== Backtest Results ===")
        for key, value in results.items():
            if key != 'trades':
                print(f"{key}: {value}")
    else:
        bot.run_live()


if __name__ == "__main__":
    main()

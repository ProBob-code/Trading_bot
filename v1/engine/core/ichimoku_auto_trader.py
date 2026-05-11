"""
Ichimoku Cloud Automated Trader
================================

Automated trading engine using Ichimoku Cloud strategy with ML/TA confluence.
Uses best Ichimoku reading strategies for buy/sell decisions.
"""

import threading
import time
from datetime import datetime
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum
import pandas as pd
import numpy as np
from loguru import logger


class TradingState(Enum):
    """Trading engine states."""
    STOPPED = "stopped"
    RUNNING = "running"
    PAUSED = "paused"


@dataclass
class IchimokuSignal:
    """Ichimoku-based trading signal."""
    signal_type: str  # 'BUY', 'SELL', 'HOLD'
    confidence: float  # 0.0 to 1.0
    timestamp: datetime
    price: float
    reasons: List[str] = field(default_factory=list)
    
    # Ichimoku component states
    kumo_breakout: Optional[str] = None  # 'bullish', 'bearish', 'in_cloud'
    tk_cross: Optional[str] = None  # 'bullish', 'bearish'
    cloud_twist: Optional[str] = None  # 'bullish', 'bearish'
    chikou_confirm: Optional[str] = None  # 'bullish', 'bearish'


@dataclass
class TradeRecord:
    """Record of an executed trade."""
    trade_id: str
    timestamp: datetime
    symbol: str
    side: str  # 'BUY' or 'SELL'
    quantity: int
    price: float
    signal_confidence: float
    signal_reasons: List[str]
    pnl: float = 0.0
    is_closed: bool = False


class IchimokuAutoTrader:
    """
    Automated trading engine using Ichimoku Cloud strategy.
    
    Ichimoku Strategy Rules:
    ========================
    
    STRONG BUY (High Confluence - 4/4 signals):
    - Price above Kumo (cloud)
    - Tenkan-sen above Kijun-sen (TK bullish cross)
    - Senkou Span A above Senkou Span B (bullish cloud)
    - Chikou Span above price from 26 periods ago
    
    MODERATE BUY (3/4 signals):
    - At least 3 of the above conditions met
    
    STRONG SELL (High Confluence - 4/4 signals):
    - Price below Kumo (cloud)
    - Tenkan-sen below Kijun-sen (TK bearish cross)
    - Senkou Span A below Senkou Span B (bearish cloud)
    - Chikou Span below price from 26 periods ago
    """
    
    def __init__(
        self,
        paper_trader,
        order_manager,
        on_trade_callback: Optional[Callable] = None,
        on_signal_callback: Optional[Callable] = None,
        check_interval_seconds: float = 2.0,
        min_confluence: int = 3,
        position_size_pct: float = 0.10,
        max_positions: int = 5
    ):
        """
        Initialize the Ichimoku auto-trader.
        
        Args:
            paper_trader: PaperTrader instance for execution
            order_manager: OrderManager instance
            on_trade_callback: Called when trade is executed
            on_signal_callback: Called when signal is generated
            check_interval_seconds: How often to check for signals
            min_confluence: Minimum Ichimoku signals for trade (1-4)
            position_size_pct: Position size as % of available capital
            max_positions: Maximum concurrent positions
        """
        self.paper_trader = paper_trader
        self.order_manager = order_manager
        self.on_trade_callback = on_trade_callback
        self.on_signal_callback = on_signal_callback
        self.check_interval = check_interval_seconds
        self.min_confluence = min_confluence
        self.position_size_pct = position_size_pct
        self.max_positions = max_positions
        
        # State
        self.state = TradingState.STOPPED
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        
        # Data
        self.df: Optional[pd.DataFrame] = None
        self.symbol: str = "UNKNOWN"
        self.current_bar_index: int = 0
        
        # Trading stats
        self.trade_history: List[TradeRecord] = []
        self.signal_history: List[IchimokuSignal] = []
        self.start_time: Optional[datetime] = None
        self.stop_time: Optional[datetime] = None
        self.initial_capital: float = 0
        
        # Performance tracking
        self.total_trades: int = 0
        self.winning_trades: int = 0
        self.losing_trades: int = 0
        self.total_pnl: float = 0
    
    def set_data(self, df: pd.DataFrame, symbol: str):
        """Set the data to trade on."""
        self.df = df.copy()
        self.symbol = symbol
        self.current_bar_index = 50  # Start after enough data for Ichimoku
    
    def start(self):
        """Start automated trading."""
        if self.state == TradingState.RUNNING:
            logger.warning("Auto-trader already running")
            return False
        
        if self.df is None or len(self.df) < 52:
            logger.error("Not enough data to start auto-trading")
            return False
        
        self.state = TradingState.RUNNING
        self.start_time = datetime.now()
        self.initial_capital = self.paper_trader.get_account_info()['total_value']
        self._stop_event.clear()
        
        # Start trading thread
        self._thread = threading.Thread(target=self._trading_loop, daemon=True)
        self._thread.start()
        
        logger.info(f"🤖 Auto-trader STARTED for {self.symbol}")
        return True
    
    def stop(self) -> Dict:
        """Stop automated trading and return report."""
        if self.state != TradingState.RUNNING:
            logger.warning("Auto-trader not running")
            return self.generate_report()
        
        self.state = TradingState.STOPPED
        self._stop_event.set()
        self.stop_time = datetime.now()
        
        if self._thread:
            self._thread.join(timeout=5)
        
        logger.info(f"🛑 Auto-trader STOPPED for {self.symbol}")
        return self.generate_report()
    
    def is_running(self) -> bool:
        """Check if auto-trader is running."""
        return self.state == TradingState.RUNNING
    
    def _trading_loop(self):
        """Main trading loop running in background thread."""
        logger.info("Trading loop started")
        
        while not self._stop_event.is_set() and self.current_bar_index < len(self.df):
            try:
                # Get current data slice
                current_df = self.df.iloc[:self.current_bar_index + 1].copy()
                
                # Ensure Ichimoku indicators are calculated
                current_df = self._ensure_ichimoku_indicators(current_df)
                
                # Update paper trader with current price
                current_price = current_df['close'].iloc[-1]
                self.paper_trader.set_prices({self.symbol: current_price})
                
                # Generate Ichimoku signal
                signal = self._analyze_ichimoku(current_df)
                self.signal_history.append(signal)
                
                if self.on_signal_callback:
                    self.on_signal_callback(signal)
                
                # Execute trade if signal is actionable
                if signal.signal_type in ['BUY', 'SELL']:
                    self._execute_signal(signal, current_df)
                
                # Move to next bar
                self.current_bar_index += 1
                
                # Wait for next check
                time.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Error in trading loop: {e}")
                time.sleep(1)
        
        logger.info("Trading loop ended")
    
    def _ensure_ichimoku_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure Ichimoku indicators are calculated."""
        if 'ichimoku_tenkan' not in df.columns:
            # Tenkan-sen (Conversion Line) - 9 period
            high_9 = df['high'].rolling(9).max()
            low_9 = df['low'].rolling(9).min()
            df['ichimoku_tenkan'] = (high_9 + low_9) / 2
            
            # Kijun-sen (Base Line) - 26 period
            high_26 = df['high'].rolling(26).max()
            low_26 = df['low'].rolling(26).min()
            df['ichimoku_kijun'] = (high_26 + low_26) / 2
            
            # Senkou Span A (Leading Span A) - (Tenkan + Kijun) / 2, shifted 26 periods ahead
            df['ichimoku_span_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(26)
            
            # Senkou Span B (Leading Span B) - 52 period, shifted 26 periods ahead
            high_52 = df['high'].rolling(52).max()
            low_52 = df['low'].rolling(52).min()
            df['ichimoku_span_b'] = ((high_52 + low_52) / 2).shift(26)
            
            # Chikou Span (Lagging Span) - Close shifted 26 periods back
            df['ichimoku_chikou'] = df['close'].shift(-26)
            
            # RSI for additional confirmation
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _analyze_ichimoku(self, df: pd.DataFrame) -> IchimokuSignal:
        """
        Analyze Ichimoku Cloud and generate trading signal.
        
        Uses 4 key Ichimoku signals:
        1. Kumo Breakout - Price above/below cloud
        2. TK Cross - Tenkan crosses Kijun
        3. Cloud Twist - Span A vs Span B
        4. Chikou Confirmation - Lagging span position
        """
        row = df.iloc[-1]
        close = row['close']
        timestamp = df.index[-1] if hasattr(df.index[-1], 'strftime') else datetime.now()
        
        bullish_signals = []
        bearish_signals = []
        reasons = []
        
        # Get Ichimoku values
        tenkan = row.get('ichimoku_tenkan', np.nan)
        kijun = row.get('ichimoku_kijun', np.nan)
        span_a = row.get('ichimoku_span_a', np.nan)
        span_b = row.get('ichimoku_span_b', np.nan)
        
        # 1. KUMO BREAKOUT - Price vs Cloud
        kumo_breakout = None
        if not pd.isna(span_a) and not pd.isna(span_b):
            cloud_top = max(span_a, span_b)
            cloud_bottom = min(span_a, span_b)
            
            if close > cloud_top:
                bullish_signals.append("kumo_breakout")
                kumo_breakout = "bullish"
                reasons.append(f"Price ₹{close:.2f} above cloud ₹{cloud_top:.2f}")
            elif close < cloud_bottom:
                bearish_signals.append("kumo_breakout")
                kumo_breakout = "bearish"
                reasons.append(f"Price ₹{close:.2f} below cloud ₹{cloud_bottom:.2f}")
            else:
                kumo_breakout = "in_cloud"
                reasons.append("Price inside cloud - neutral zone")
        
        # 2. TK CROSS - Tenkan vs Kijun
        tk_cross = None
        if not pd.isna(tenkan) and not pd.isna(kijun):
            if tenkan > kijun:
                bullish_signals.append("tk_cross")
                tk_cross = "bullish"
                reasons.append(f"TK Cross bullish: Tenkan ₹{tenkan:.2f} > Kijun ₹{kijun:.2f}")
            else:
                bearish_signals.append("tk_cross")
                tk_cross = "bearish"
                reasons.append(f"TK Cross bearish: Tenkan ₹{tenkan:.2f} < Kijun ₹{kijun:.2f}")
        
        # 3. CLOUD TWIST - Span A vs Span B (future cloud direction)
        cloud_twist = None
        if not pd.isna(span_a) and not pd.isna(span_b):
            if span_a > span_b:
                bullish_signals.append("cloud_twist")
                cloud_twist = "bullish"
                reasons.append("Cloud twist bullish: Span A > Span B")
            else:
                bearish_signals.append("cloud_twist")
                cloud_twist = "bearish"
                reasons.append("Cloud twist bearish: Span A < Span B")
        
        # 4. CHIKOU CONFIRMATION - Lagging span vs price 26 periods ago
        chikou_confirm = None
        if len(df) > 26:
            chikou = row.get('ichimoku_chikou', np.nan)
            price_26_ago = df['close'].iloc[-27] if len(df) > 26 else np.nan
            
            if not pd.isna(chikou) and not pd.isna(price_26_ago):
                if chikou > price_26_ago:
                    bullish_signals.append("chikou")
                    chikou_confirm = "bullish"
                    reasons.append("Chikou Span above price (bullish confirmation)")
                else:
                    bearish_signals.append("chikou")
                    chikou_confirm = "bearish"
                    reasons.append("Chikou Span below price (bearish confirmation)")
        
        # 5. RSI Filter (avoid overbought/oversold extremes)
        rsi = row.get('rsi', 50)
        if not pd.isna(rsi):
            if rsi > 70:
                reasons.append(f"RSI overbought: {rsi:.1f}")
                # Remove one bullish signal if RSI is too high
                if bullish_signals:
                    bullish_signals = bullish_signals[:-1]
            elif rsi < 30:
                reasons.append(f"RSI oversold: {rsi:.1f}")
                # Remove one bearish signal if RSI is too low
                if bearish_signals:
                    bearish_signals = bearish_signals[:-1]
        
        # Generate final signal based on confluence
        n_bullish = len(bullish_signals)
        n_bearish = len(bearish_signals)
        
        if n_bullish >= self.min_confluence and n_bullish > n_bearish:
            signal_type = "BUY"
            confidence = n_bullish / 4.0
        elif n_bearish >= self.min_confluence and n_bearish > n_bullish:
            signal_type = "SELL"
            confidence = n_bearish / 4.0
        else:
            signal_type = "HOLD"
            confidence = 0.0
        
        return IchimokuSignal(
            signal_type=signal_type,
            confidence=confidence,
            timestamp=timestamp,
            price=close,
            reasons=reasons,
            kumo_breakout=kumo_breakout,
            tk_cross=tk_cross,
            cloud_twist=cloud_twist,
            chikou_confirm=chikou_confirm
        )
    
    def _execute_signal(self, signal: IchimokuSignal, df: pd.DataFrame):
        """Execute a trading signal."""
        account = self.paper_trader.get_account_info()
        positions = self.paper_trader.get_positions()
        current_price = signal.price
        
        # Check position limits
        if len(positions) >= self.max_positions and signal.signal_type == "BUY":
            logger.info("Max positions reached, skipping BUY signal")
            return
        
        if signal.signal_type == "BUY":
            # Calculate position size
            available_capital = account['cash']
            position_value = available_capital * self.position_size_pct
            quantity = int(position_value / current_price)
            
            if quantity < 1:
                logger.warning("Insufficient funds for trade")
                return
            
            # Check if already have position in this symbol
            existing_pos = [p for p in positions if p['symbol'] == self.symbol]
            if existing_pos:
                logger.info(f"Already have position in {self.symbol}, skipping")
                return
            
            # Execute BUY
            self.paper_trader.set_prices({self.symbol: current_price})
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side='buy',
                quantity=quantity,
                order_type='market'
            )
            
            if self.order_manager.submit_order(order):
                trade_id = f"AUTO_{datetime.now().strftime('%H%M%S')}"
                trade = TradeRecord(
                    trade_id=trade_id,
                    timestamp=datetime.now(),
                    symbol=self.symbol,
                    side="BUY",
                    quantity=quantity,
                    price=current_price,
                    signal_confidence=signal.confidence,
                    signal_reasons=signal.reasons
                )
                self.trade_history.append(trade)
                self.total_trades += 1
                
                logger.info(f"🟢 AUTO BUY: {quantity} x {self.symbol} @ ₹{current_price:.2f}")
                
                if self.on_trade_callback:
                    self.on_trade_callback(trade)
        
        elif signal.signal_type == "SELL":
            # Check if we have a position to sell
            positions = self.paper_trader.get_positions()
            pos = next((p for p in positions if p['symbol'] == self.symbol), None)
            
            if not pos:
                logger.info(f"No position in {self.symbol} to sell")
                return
            
            quantity = pos['quantity']
            entry_price = pos['avg_price']
            
            # Execute SELL
            self.paper_trader.set_prices({self.symbol: current_price})
            order = self.order_manager.create_order(
                symbol=self.symbol,
                side='sell',
                quantity=quantity,
                order_type='market'
            )
            
            if self.order_manager.submit_order(order):
                pnl = (current_price - entry_price) * quantity
                trade_id = f"AUTO_{datetime.now().strftime('%H%M%S')}"
                trade = TradeRecord(
                    trade_id=trade_id,
                    timestamp=datetime.now(),
                    symbol=self.symbol,
                    side="SELL",
                    quantity=quantity,
                    price=current_price,
                    signal_confidence=signal.confidence,
                    signal_reasons=signal.reasons,
                    pnl=pnl,
                    is_closed=True
                )
                self.trade_history.append(trade)
                self.total_trades += 1
                self.total_pnl += pnl
                
                if pnl > 0:
                    self.winning_trades += 1
                else:
                    self.losing_trades += 1
                
                logger.info(f"🔴 AUTO SELL: {quantity} x {self.symbol} @ ₹{current_price:.2f} | P&L: ₹{pnl:+.2f}")
                
                if self.on_trade_callback:
                    self.on_trade_callback(trade)
    
    def generate_report(self) -> Dict:
        """Generate comprehensive trading report."""
        account = self.paper_trader.get_account_info()
        final_capital = account['total_value']
        
        # Calculate metrics
        duration = (self.stop_time or datetime.now()) - (self.start_time or datetime.now())
        duration_minutes = duration.total_seconds() / 60
        
        # Win rate
        closed_trades = [t for t in self.trade_history if t.is_closed]
        win_rate = (self.winning_trades / len(closed_trades) * 100) if closed_trades else 0
        
        # Profit factor
        gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
        gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average trade
        avg_trade_pnl = self.total_pnl / len(closed_trades) if closed_trades else 0
        
        # Max drawdown approximation
        equity_curve = [self.initial_capital]
        running_pnl = 0
        for trade in self.trade_history:
            running_pnl += trade.pnl
            equity_curve.append(self.initial_capital + running_pnl)
        
        peak = equity_curve[0]
        max_drawdown = 0
        for value in equity_curve:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak * 100
            if drawdown > max_drawdown:
                max_drawdown = drawdown
        
        # Return on investment
        roi = ((final_capital - self.initial_capital) / self.initial_capital) * 100 if self.initial_capital > 0 else 0
        
        # Signal analysis
        buy_signals = [s for s in self.signal_history if s.signal_type == "BUY"]
        sell_signals = [s for s in self.signal_history if s.signal_type == "SELL"]
        
        # Ichimoku signal breakdown
        kumo_bullish = sum(1 for s in self.signal_history if s.kumo_breakout == "bullish")
        kumo_bearish = sum(1 for s in self.signal_history if s.kumo_breakout == "bearish")
        tk_bullish = sum(1 for s in self.signal_history if s.tk_cross == "bullish")
        tk_bearish = sum(1 for s in self.signal_history if s.tk_cross == "bearish")
        
        return {
            # Summary
            'symbol': self.symbol,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'stop_time': (self.stop_time or datetime.now()).isoformat(),
            'duration_minutes': round(duration_minutes, 2),
            
            # Capital
            'initial_capital': self.initial_capital,
            'final_capital': final_capital,
            'total_pnl': self.total_pnl,
            'roi_percent': round(roi, 2),
            
            # Trade Stats
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'win_rate': round(win_rate, 2),
            'profit_factor': round(profit_factor, 2) if profit_factor != float('inf') else 'N/A',
            'avg_trade_pnl': round(avg_trade_pnl, 2),
            'max_drawdown_pct': round(max_drawdown, 2),
            
            # P&L Breakdown
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            
            # Signal Stats
            'total_signals': len(self.signal_history),
            'buy_signals': len(buy_signals),
            'sell_signals': len(sell_signals),
            
            # Ichimoku Analysis
            'ichimoku_stats': {
                'kumo_bullish': kumo_bullish,
                'kumo_bearish': kumo_bearish,
                'tk_bullish': tk_bullish,
                'tk_bearish': tk_bearish,
            },
            
            # Trade History
            'trades': [
                {
                    'trade_id': t.trade_id,
                    'timestamp': t.timestamp.isoformat(),
                    'side': t.side,
                    'quantity': t.quantity,
                    'price': t.price,
                    'pnl': t.pnl,
                    'confidence': t.signal_confidence,
                    'reasons': t.signal_reasons
                }
                for t in self.trade_history
            ]
        }
    
    def get_status(self) -> Dict:
        """Get current trading status."""
        account = self.paper_trader.get_account_info()
        positions = self.paper_trader.get_positions()
        
        return {
            'state': self.state.value,
            'symbol': self.symbol,
            'current_bar': self.current_bar_index,
            'total_bars': len(self.df) if self.df is not None else 0,
            'progress_pct': (self.current_bar_index / len(self.df) * 100) if self.df is not None and len(self.df) > 0 else 0,
            'total_trades': self.total_trades,
            'current_pnl': account['pnl'],
            'positions': positions,
            'latest_signal': self.signal_history[-1] if self.signal_history else None
        }

"""
Paper Trader
=============

Simulated broker for paper trading and backtesting.
"""

from datetime import datetime
from typing import Dict, List, Optional
import random
from loguru import logger

from .base_broker import BaseBroker
from ..order_manager import Order, OrderStatus, OrderType, OrderSide


class PaperTrader(BaseBroker):
    """
    Paper trading broker for testing strategies.
    
    Simulates order execution with configurable slippage and latency.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        slippage_pct: float = 0.05,
        commission_pct: float = 0.1,
        simulate_latency: bool = False,
        **kwargs
    ):
        """
        Initialize paper trader.
        
        Args:
            initial_capital: Starting capital
            slippage_pct: Simulated slippage as percentage (0.05 = 0.05%)
            commission_pct: Commission as percentage of trade value
            simulate_latency: Add random latency to simulations
        """
        super().__init__(**kwargs)
        
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.slippage_pct = slippage_pct / 100
        self.commission_pct = commission_pct / 100
        self.simulate_latency = simulate_latency
        
        # State
        self.positions: Dict[str, Dict] = {}  # LONG positions
        self.short_positions: Dict[str, Dict] = {}  # SHORT positions
        self.closed_positions: List[Dict] = []  # Track closed positions
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []
        
        # Mock prices (set externally)
        self.current_prices: Dict[str, float] = {}
        
        # Order callback
        self.order_manager = None
        
        self.is_connected = True
        
    def set_order_manager(self, order_manager):
        """Set order manager for fill callbacks."""
        self.order_manager = order_manager
        
    def set_prices(self, prices: Dict[str, float]):
        """
        Update current prices.
        
        Call this with real-time prices to simulate order fills.
        """
        self.current_prices = prices
        
        # Check pending orders
        self._check_pending_orders()
    
    def connect(self) -> bool:
        """Connect (always succeeds for paper trading)."""
        self.is_connected = True
        logger.info("Paper trader connected")
        return True
    
    def disconnect(self):
        """Disconnect."""
        self.is_connected = False
        logger.info("Paper trader disconnected")
    
    def get_account_info(self) -> Dict:
        """Get account information."""
        # Calculate LONG positions value
        long_value = sum(
            pos['quantity'] * self.current_prices.get(symbol, pos['avg_price'])
            for symbol, pos in self.positions.items()
        )
        
        # Calculate SHORT positions P&L (profit when price drops)
        short_pnl = 0
        for symbol, pos in self.short_positions.items():
            current_price = self.current_prices.get(symbol, pos['entry_price'])
            # Profit = (entry - current) * quantity
            short_pnl += (pos['entry_price'] - current_price) * pos['quantity']
        
        total_value = self.cash + long_value + short_pnl
        
        # Buying power = cash minus short margin obligations
        # When we short, cash goes UP (proceeds), but we owe the shares back.
        # True buying power is cash minus the current cost to cover all shorts.
        short_margin = sum(
            pos['quantity'] * self.current_prices.get(symbol, pos['entry_price'])
            for symbol, pos in self.short_positions.items()
        )
        buying_power = self.cash - short_margin
        
        return {
            'account_id': 'PAPER_ACCOUNT',
            'cash': self.cash,
            'long_value': long_value,
            'short_pnl': short_pnl,
            'total_value': total_value,
            'buying_power': buying_power,
            'initial_capital': self.initial_capital,
            'pnl': total_value - self.initial_capital,
            'pnl_pct': ((total_value - self.initial_capital) / self.initial_capital) * 100
        }
    
    def get_positions(self) -> List[Dict]:
        """Get current positions."""
        positions = []
        
        for symbol, pos in self.positions.items():
            current_price = self.current_prices.get(symbol, pos['avg_price'])
            market_value = pos['quantity'] * current_price
            unrealized_pnl = (current_price - pos['avg_price']) * pos['quantity']
            
            positions.append({
                'symbol': symbol,
                'side': 'LONG',
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (pos['avg_price'] * pos['quantity'])) * 100 if pos['avg_price'] > 0 else 0
            })

        for symbol, pos in self.short_positions.items():
            current_price = self.current_prices.get(symbol, pos['entry_price'])
            # Pnl = (entry - current) * quantity
            unrealized_pnl = (pos['entry_price'] - current_price) * pos['quantity']
            
            positions.append({
                'symbol': symbol,
                'side': 'SHORT',
                'quantity': pos['quantity'],
                'avg_price': pos['entry_price'],
                'current_price': current_price,
                'market_value': pos['quantity'] * current_price,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (pos['entry_price'] * pos['quantity'])) * 100 if pos['entry_price'] > 0 else 0
            })
        
        return positions
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for paper execution.
        
        Supports:
        - BUY: Open LONG position or cover SHORT
        - SELL: Close LONG position or open SHORT
        """
        price = self.current_prices.get(order.symbol, 100)
        
        if order.side == OrderSide.BUY:
            # BUY can be: opening LONG or covering SHORT
            if order.symbol in self.short_positions:
                # Covering a SHORT position - no cash needed upfront
                pass
            else:
                # Opening LONG - need cash
                if order.order_type == OrderType.MARKET:
                    required = price * order.quantity * (1 + self.slippage_pct + self.commission_pct)
                    if required > self.cash:
                        logger.warning(f"Insufficient funds: need ${required:.2f}, have ${self.cash:.2f}")
                        return False
        else:  # SELL
            # SELL can be: closing LONG or opening SHORT
            if order.symbol in self.positions:
                # Closing LONG - check we have enough
                if self.positions[order.symbol]['quantity'] < order.quantity:
                    logger.warning(f"Insufficient quantity to sell")
                    return False
            else:
                # Opening SHORT - allowed (paper trading lets us short anything)
                # In real trading, this would need margin check
                pass
        
        # Handle order based on type
        if order.order_type == OrderType.MARKET:
            self._execute_market_order(order)
        else:
            # Limit/Stop orders go to pending
            self.pending_orders[order.order_id] = order
            logger.info(f"Order {order.order_id} pending: {order.order_type.value} @ {order.price or order.stop_price}")
        
        self.order_history.append(order)
        return True
    
    def cancel_order(self, order: Order) -> bool:
        """Cancel a pending order."""
        if order.order_id in self.pending_orders:
            del self.pending_orders[order.order_id]
            order.status = OrderStatus.CANCELLED
            logger.info(f"Cancelled order: {order.order_id}")
            return True
        return False
    
    def get_order_status(self, order_id: str) -> Dict:
        """Get order status."""
        for order in self.order_history:
            if order.order_id == order_id:
                return order.to_dict()
        return {'error': 'Order not found'}
    
    def get_quote(self, symbol: str) -> Dict:
        """Get current quote for a symbol."""
        price = self.current_prices.get(symbol, 0)
        
        if price == 0:
            return {'error': f'No price for {symbol}'}
        
        # Simulate bid/ask spread
        spread = price * 0.001  # 0.1% spread
        
        return {
            'symbol': symbol,
            'bid': price - spread/2,
            'ask': price + spread/2,
            'last': price,
            'timestamp': datetime.now().isoformat()
        }
    
    def _execute_market_order(self, order: Order):
        """
        Execute a market order immediately.
        
        Supports:
        - LONG: BUY to open, SELL to close
        - SHORT: SELL to open, BUY to cover
        """
        base_price = self.current_prices.get(order.symbol)
        
        if base_price is None:
            logger.error(f"No price available for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return
        
        # Apply slippage
        if order.side == OrderSide.BUY:
            fill_price = base_price * (1 + self.slippage_pct)
        else:
            fill_price = base_price * (1 - self.slippage_pct)
        
        # Calculate commission
        trade_value = fill_price * order.quantity
        commission = trade_value * self.commission_pct
        
        # Execute based on side and existing positions
        if order.side == OrderSide.BUY:
            # BUY: either cover SHORT or open/add LONG
            if order.symbol in self.short_positions:
                # COVER SHORT - buy back borrowed shares
                short_pos = self.short_positions[order.symbol]
                cover_qty = min(order.quantity, short_pos['quantity'])
                
                # P&L = (entry - exit) * quantity (profit when price dropped)
                realized_pnl = (short_pos['entry_price'] - fill_price) * cover_qty
                
                # When covering, we use cash to buy back
                self.cash -= (fill_price * cover_qty + commission)
                # But we get back the collateral from the short
                self.cash += short_pos['entry_price'] * cover_qty
                # Net effect = profit/loss from the short
                
                # Record closed short
                self.closed_positions.append({
                    'symbol': order.symbol,
                    'quantity': cover_qty,
                    'entry_price': short_pos['entry_price'],
                    'exit_price': fill_price,
                    'side': 'SHORT',
                    'realized_pnl': realized_pnl,
                    'commission': commission,
                    'closed_at': datetime.now()
                })
                
                short_pos['quantity'] -= cover_qty
                if short_pos['quantity'] <= 0:
                    del self.short_positions[order.symbol]
                
                logger.info(f"🔵 COVERED SHORT: {cover_qty} {order.symbol} @ {fill_price:.2f} | P&L: ${realized_pnl:.2f}")
            else:
                # Open/add to LONG position
                self.cash -= (trade_value + commission)
                
                if order.symbol in self.positions:
                    pos = self.positions[order.symbol]
                    total_qty = pos['quantity'] + order.quantity
                    pos['avg_price'] = (
                        (pos['quantity'] * pos['avg_price'] + order.quantity * fill_price) / 
                        total_qty
                    )
                    pos['quantity'] = total_qty
                else:
                    self.positions[order.symbol] = {
                        'quantity': order.quantity,
                        'avg_price': fill_price
                    }
                logger.info(f"🟢 LONG BUY: {order.quantity} {order.symbol} @ {fill_price:.2f}")
                
        else:  # SELL
            # SELL: either close LONG or open SHORT
            if order.symbol in self.positions:
                # Close LONG position
                pos = self.positions[order.symbol]
                sell_qty = min(order.quantity, pos['quantity'])
                
                realized_pnl = (fill_price - pos['avg_price']) * sell_qty
                self.cash += (fill_price * sell_qty - commission)
                
                self.closed_positions.append({
                    'symbol': order.symbol,
                    'quantity': sell_qty,
                    'entry_price': pos['avg_price'],
                    'exit_price': fill_price,
                    'side': 'LONG',
                    'realized_pnl': realized_pnl,
                    'commission': commission,
                    'closed_at': datetime.now()
                })
                
                pos['quantity'] -= sell_qty
                if pos['quantity'] <= 0:
                    del self.positions[order.symbol]
                
                logger.info(f"🟢 LONG SELL: {sell_qty} {order.symbol} @ {fill_price:.2f} | P&L: ${realized_pnl:.2f}")
            else:
                # Open SHORT position (sell borrowed shares)
                # In paper trading, we simulate borrowing
                # Cash increases by sale proceeds (collateral held)
                self.cash += (trade_value - commission)
                
                if order.symbol in self.short_positions:
                    short_pos = self.short_positions[order.symbol]
                    total_qty = short_pos['quantity'] + order.quantity
                    short_pos['entry_price'] = (
                        (short_pos['quantity'] * short_pos['entry_price'] + order.quantity * fill_price) /
                        total_qty
                    )
                    short_pos['quantity'] = total_qty
                else:
                    self.short_positions[order.symbol] = {
                        'quantity': order.quantity,
                        'entry_price': fill_price
                    }
                logger.info(f"🔴 SHORT SELL: {order.quantity} {order.symbol} @ {fill_price:.2f}")
        
        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        # Record trade
        self.trade_history.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': datetime.now()
        })
        
        logger.info(f"Filled: {order.side.value} {order.quantity} {order.symbol} @ {fill_price:.2f}")
        
        # Callback to order manager
        if self.order_manager:
            self.order_manager.handle_fill(order.order_id, order.quantity, fill_price)
    
    def _check_pending_orders(self):
        """Check and execute pending limit/stop orders."""
        filled_orders = []
        
        for order_id, order in list(self.pending_orders.items()):
            price = self.current_prices.get(order.symbol)
            if price is None:
                continue
            
            should_fill = False
            
            if order.order_type == OrderType.LIMIT:
                if order.side == OrderSide.BUY and price <= order.price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price >= order.price:
                    should_fill = True
                    
            elif order.order_type == OrderType.STOP:
                if order.side == OrderSide.BUY and price >= order.stop_price:
                    should_fill = True
                elif order.side == OrderSide.SELL and price <= order.stop_price:
                    should_fill = True
            
            if should_fill:
                # Convert to market execution at current price
                order.order_type = OrderType.MARKET
                self._execute_market_order(order)
                filled_orders.append(order_id)
        
        # Remove filled orders from pending
        for order_id in filled_orders:
            del self.pending_orders[order_id]
    
    def get_trade_history(self) -> List[Dict]:
        """Get trade history."""
        return self.trade_history
    
    def get_closed_positions(self) -> List[Dict]:
        """Get closed/sold positions with realized P&L."""
        return self.closed_positions
    
    def get_pending_orders(self) -> List[Dict]:
        """Get pending/active orders."""
        return [
            {
                'order_id': order.order_id,
                'symbol': order.symbol,
                'side': order.side.value,
                'order_type': order.order_type.value,
                'quantity': order.quantity,
                'price': order.price,
                'stop_price': order.stop_price,
                'status': order.status.value,
                'created_at': order.created_at
            }
            for order in self.pending_orders.values()
        ]
    
    def get_market_depth(self, symbol: str) -> Dict:
        """Get simulated market depth (bid/ask order book)."""
        price = self.current_prices.get(symbol, 100)
        spread = price * 0.001  # 0.1% spread
        
        # Generate simulated order book
        bids = []
        asks = []
        
        for i in range(5):
            bid_price = price - spread/2 - (i * spread * 0.5)
            ask_price = price + spread/2 + (i * spread * 0.5)
            # Simulated volume (decreasing with distance from price)
            volume = int(random.uniform(100, 1000) * (5 - i) / 5)
            
            bids.append({'price': round(bid_price, 2), 'volume': volume})
            asks.append({'price': round(ask_price, 2), 'volume': volume})
        
        return {
            'symbol': symbol,
            'bids': bids,
            'asks': asks,
            'last_price': price,
            'timestamp': datetime.now().isoformat()
        }
    
    def reset(self):
        """Reset paper trader to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.short_positions.clear()  # Clear shorts too
        self.closed_positions.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        self.trade_history.clear()
        logger.info("Paper trader reset")

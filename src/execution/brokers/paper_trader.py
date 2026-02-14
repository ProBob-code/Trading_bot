"""
Paper Trader
=============

Simulated broker for paper trading and backtesting.
"""

from datetime import datetime
from typing import Dict, List, Optional
import random
import threading
from loguru import logger

from .base_broker import BaseBroker
from ..order_manager import Order, OrderStatus, OrderType, OrderSide


class PaperAccount:
    """Isolated account state for paper trading."""
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, Dict] = {}
        self.short_positions: Dict[str, Dict] = {}
        self.closed_positions: List[Dict] = []
        self.pending_orders: Dict[str, Order] = {}
        self.order_history: List[Order] = []
        self.trade_history: List[Dict] = []

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
        self.lock = threading.Lock()
        
        # Multi-account state
        self.accounts: Dict[int, PaperAccount] = {}
        self.initial_capital_default = initial_capital
        
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
        with self.lock:
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

    def _get_account(self, user_id: Optional[int]) -> PaperAccount:
        """Helper to get/create user account."""
        uid = user_id if user_id is not None else 0
        if uid not in self.accounts:
            self.accounts[uid] = PaperAccount(self.initial_capital_default)
        return self.accounts[uid]

    def get_account_info(self, user_id: Optional[int] = None) -> Dict:
        """Get account information for a specific user."""
        with self.lock:
            acc = self._get_account(user_id)
            
            # Calculate LONG positions value
            long_value = sum(
                pos['quantity'] * self.current_prices.get(symbol, pos['avg_price'])
                for symbol, pos in acc.positions.items()
            )
            
            # Calculate SHORT positions P&L (profit when price drops)
            short_pnl = 0
            for symbol, pos in acc.short_positions.items():
                current_price = self.current_prices.get(symbol, pos['entry_price'])
                # Profit = (entry - current) * quantity
                short_pnl += (pos['entry_price'] - current_price) * pos['quantity']
            
            total_value = acc.cash + long_value + short_pnl
            
            # Buying power = cash minus short margin obligations
            short_margin = sum(
                pos['quantity'] * self.current_prices.get(symbol, pos['entry_price'])
                for symbol, pos in acc.short_positions.items()
            )
            buying_power = acc.cash - short_margin
            
            return {
                'account_id': f'PAPER_USER_{user_id if user_id is not None else 0}',
                'cash': acc.cash,
                'long_value': long_value,
                'short_pnl': short_pnl,
                'total_value': total_value,
                'buying_power': buying_power,
                'initial_capital': acc.initial_capital,
                'pnl': total_value - acc.initial_capital,
                'pnl_pct': ((total_value - acc.initial_capital) / acc.initial_capital) * 100 if acc.initial_capital > 0 else 0
            }

    def get_positions(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get current positions for a specific user."""
        with self.lock:
            acc = self._get_account(user_id)
            positions = []
            
            for symbol, pos in acc.positions.items():
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

            for symbol, pos in acc.short_positions.items():
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
        """
        with self.lock:
            acc = self._get_account(order.user_id)
            price = self.current_prices.get(order.symbol, 100)
            
            if order.side == OrderSide.BUY:
                if order.symbol not in acc.short_positions:
                    if order.order_type == OrderType.MARKET:
                        required = price * order.quantity * (1 + self.slippage_pct + self.commission_pct)
                        if required > acc.cash:
                            logger.warning(f"Insufficient funds user {order.user_id}: need ${required:.2f}, have ${acc.cash:.2f}")
                            return False
            else:  # SELL
                if order.symbol in acc.positions:
                    if acc.positions[order.symbol]['quantity'] < order.quantity:
                        logger.warning(f"Insufficient quantity to sell: user {order.user_id}")
                        return False
            
            # Handle order based on type
            if order.order_type == OrderType.MARKET:
                self._execute_market_order(order)
            else:
                acc.pending_orders[order.order_id] = order
                logger.info(f"Order {order.order_id} pending: {order.order_type.value} @ {order.price or order.stop_price}")
            
            acc.order_history.append(order)
            return True
    
    def cancel_order(self, order: Order) -> bool:
        """Cancel a pending order."""
        with self.lock:
            acc = self._get_account(order.user_id)
            if order.order_id in acc.pending_orders:
                del acc.pending_orders[order.order_id]
                order.status = OrderStatus.CANCELLED
                logger.info(f"Cancelled order: {order.order_id}")
                return True
            return False
    
    def get_order_status(self, order_id: str, user_id: Optional[int] = None) -> Dict:
        """Get order status."""
        acc = self._get_account(user_id)
        for order in acc.order_history:
            if order.order_id == order_id:
                return order.to_dict()
        return {'error': 'Order not found'}
    
    def get_pending_orders(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all pending orders for a specific user."""
        with self.lock:
            acc = self._get_account(user_id)
            return [order.to_dict() for order in acc.pending_orders.values()]
    
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
        """
        acc = self._get_account(order.user_id)
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
            if order.symbol in acc.short_positions:
                # COVER SHORT - buy back borrowed shares
                short_pos = acc.short_positions[order.symbol]
                cover_qty = min(order.quantity, short_pos['quantity'])
                
                # P&L = (entry - exit) * quantity
                realized_pnl = (short_pos['entry_price'] - fill_price) * cover_qty
                
                acc.cash -= (fill_price * cover_qty + commission)
                acc.cash += short_pos['entry_price'] * cover_qty
                
                # Record closed short
                acc.closed_positions.append({
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
                    del acc.short_positions[order.symbol]
                
                logger.info(f"🔵 COVERED SHORT: user {order.user_id} | {cover_qty} {order.symbol} @ {fill_price:.2f}")
            else:
                # Open/add to LONG position
                acc.cash -= (trade_value + commission)
                
                if order.symbol in acc.positions:
                    pos = acc.positions[order.symbol]
                    total_qty = pos['quantity'] + order.quantity
                    pos['avg_price'] = (
                        (pos['quantity'] * pos['avg_price'] + order.quantity * fill_price) / 
                        total_qty
                    )
                    pos['quantity'] = total_qty
                else:
                    acc.positions[order.symbol] = {
                        'quantity': order.quantity,
                        'avg_price': fill_price
                    }
                logger.info(f"🟢 LONG BUY: user {order.user_id} | {order.quantity} {order.symbol} @ {fill_price:.2f}")
                
        else:  # SELL
            # SELL: either close LONG or open SHORT
            if order.symbol in acc.positions:
                # Close LONG position
                pos = acc.positions[order.symbol]
                sell_qty = min(order.quantity, pos['quantity'])
                
                realized_pnl = (fill_price - pos['avg_price']) * sell_qty
                acc.cash += (fill_price * sell_qty - commission)
                
                acc.closed_positions.append({
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
                    del acc.positions[order.symbol]
                
                logger.info(f"🟢 LONG SELL: user {order.user_id} | {sell_qty} {order.symbol} @ {fill_price:.2f}")
            else:
                # Open SHORT position
                acc.cash += (trade_value - commission)
                
                if order.symbol in acc.short_positions:
                    short_pos = acc.short_positions[order.symbol]
                    total_qty = short_pos['quantity'] + order.quantity
                    short_pos['entry_price'] = (
                        (short_pos['quantity'] * short_pos['entry_price'] + order.quantity * fill_price) /
                        total_qty
                    )
                    short_pos['quantity'] = total_qty
                else:
                    acc.short_positions[order.symbol] = {
                        'quantity': order.quantity,
                        'entry_price': fill_price
                    }
                logger.info(f"🔴 SHORT SELL: user {order.user_id} | {order.quantity} {order.symbol} @ {fill_price:.2f}")
        
        # Update order
        order.filled_quantity = order.quantity
        order.filled_price = fill_price
        order.status = OrderStatus.FILLED
        order.filled_at = datetime.now()
        
        # Record trade
        acc.trade_history.append({
            'order_id': order.order_id,
            'symbol': order.symbol,
            'side': order.side.value,
            'quantity': order.quantity,
            'price': fill_price,
            'commission': commission,
            'timestamp': datetime.now()
        })
        
        # Callback to order manager
        if self.order_manager:
            self.order_manager.handle_fill(order.order_id, order.quantity, fill_price)
    
    def _check_pending_orders(self):
        """Check and execute pending limit/stop orders for ALL users."""
        for acc in self.accounts.values():
            filled_orders = []
            for order_id, order in list(acc.pending_orders.items()):
                price = self.current_prices.get(order.symbol)
                if price is None:
                    continue
                
                should_fill = False
                if order.order_type == OrderType.LIMIT:
                    if (order.side == OrderSide.BUY and price <= order.price) or \
                       (order.side == OrderSide.SELL and price >= order.price):
                        should_fill = True
                elif order.order_type == OrderType.STOP:
                    if (order.side == OrderSide.BUY and price >= order.stop_price) or \
                       (order.side == OrderSide.SELL and price <= order.stop_price):
                        should_fill = True
                
                if should_fill:
                    order.order_type = OrderType.MARKET
                    self._execute_market_order(order)
                    filled_orders.append(order_id)
            
            for order_id in filled_orders:
                del acc.pending_orders[order_id]
    
    def get_trade_history(self, user_id: Optional[int] = None) -> List[Dict]:
        return self._get_account(user_id).trade_history
    
    def get_closed_positions(self, user_id: Optional[int] = None) -> List[Dict]:
        return self._get_account(user_id).closed_positions
    
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
            # Simulated volume
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

    def reset(self, user_id: Optional[int] = None):
        """Reset paper trader state for a specific user or all."""
        with self.lock:
            if user_id is not None:
                if user_id in self.accounts:
                    self.accounts[user_id] = PaperAccount(self.initial_capital_default)
                    logger.info(f"Reset paper trader for user {user_id}")
            else:
                self.accounts.clear()
                logger.info("Reset all paper trader accounts")

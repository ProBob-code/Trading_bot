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
        self.positions: Dict[str, Dict] = {}
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
        # Calculate portfolio value
        positions_value = sum(
            pos['quantity'] * self.current_prices.get(symbol, pos['avg_price'])
            for symbol, pos in self.positions.items()
        )
        
        total_value = self.cash + positions_value
        
        return {
            'account_id': 'PAPER_ACCOUNT',
            'cash': self.cash,
            'positions_value': positions_value,
            'total_value': total_value,
            'buying_power': self.cash,
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
                'quantity': pos['quantity'],
                'avg_price': pos['avg_price'],
                'current_price': current_price,
                'market_value': market_value,
                'unrealized_pnl': unrealized_pnl,
                'unrealized_pnl_pct': (unrealized_pnl / (pos['avg_price'] * pos['quantity'])) * 100
            })
        
        return positions
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit an order for paper execution.
        """
        # Validate
        if order.side == OrderSide.BUY:
            if order.order_type == OrderType.MARKET:
                price = self.current_prices.get(order.symbol, 100)
                required = price * order.quantity * (1 + self.slippage_pct + self.commission_pct)
                if required > self.cash:
                    logger.warning(f"Insufficient funds: need ${required:.2f}, have ${self.cash:.2f}")
                    return False
        else:  # SELL
            if order.symbol not in self.positions:
                logger.warning(f"No position to sell: {order.symbol}")
                return False
            if self.positions[order.symbol]['quantity'] < order.quantity:
                logger.warning(f"Insufficient quantity to sell")
                return False
        
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
        """Execute a market order immediately."""
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
        
        # Execute
        if order.side == OrderSide.BUY:
            self.cash -= (trade_value + commission)
            
            if order.symbol in self.positions:
                # Update position
                pos = self.positions[order.symbol]
                total_qty = pos['quantity'] + order.quantity
                pos['avg_price'] = (
                    (pos['quantity'] * pos['avg_price'] + order.quantity * fill_price) / 
                    total_qty
                )
                pos['quantity'] = total_qty
            else:
                # New position
                self.positions[order.symbol] = {
                    'quantity': order.quantity,
                    'avg_price': fill_price
                }
        else:  # SELL
            self.cash += (trade_value - commission)
            
            pos = self.positions[order.symbol]
            pos['quantity'] -= order.quantity
            
            if pos['quantity'] <= 0:
                del self.positions[order.symbol]
        
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
    
    def reset(self):
        """Reset paper trader to initial state."""
        self.cash = self.initial_capital
        self.positions.clear()
        self.pending_orders.clear()
        self.order_history.clear()
        self.trade_history.clear()
        logger.info("Paper trader reset")

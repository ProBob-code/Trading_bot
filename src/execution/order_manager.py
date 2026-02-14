"""
Order Manager
=============

Manages order lifecycle and execution.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Callable
import uuid
from loguru import logger


class OrderType(Enum):
    """Order types."""
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"


class OrderSide(Enum):
    """Order side."""
    BUY = "buy"
    SELL = "sell"


class OrderStatus(Enum):
    """Order status."""
    PENDING = "pending"
    SUBMITTED = "submitted"
    PARTIAL = "partial"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"
    EXPIRED = "expired"


@dataclass
class Order:
    """
    Represents a trading order.
    """
    user_id: int
    symbol: str
    side: OrderSide
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit orders
    stop_price: Optional[float] = None  # For stop orders
    
    # Generated fields
    order_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    status: OrderStatus = OrderStatus.PENDING
    filled_quantity: float = 0
    filled_price: float = 0
    created_at: datetime = field(default_factory=datetime.now)
    filled_at: Optional[datetime] = None
    
    # Bracket orders (OCO)
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    # Metadata
    strategy_id: Optional[str] = None
    signal_id: Optional[str] = None
    notes: str = ""
    
    @property
    def is_filled(self) -> bool:
        return self.status == OrderStatus.FILLED
    
    @property
    def is_active(self) -> bool:
        return self.status in [OrderStatus.PENDING, OrderStatus.SUBMITTED, OrderStatus.PARTIAL]
    
    @property
    def remaining_quantity(self) -> float:
        return self.quantity - self.filled_quantity
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'order_id': self.order_id,
            'symbol': self.symbol,
            'side': self.side.value,
            'order_type': self.order_type.value,
            'quantity': self.quantity,
            'price': self.price,
            'stop_price': self.stop_price,
            'status': self.status.value,
            'filled_quantity': self.filled_quantity,
            'filled_price': self.filled_price,
            'created_at': self.created_at.isoformat(),
            'filled_at': self.filled_at.isoformat() if self.filled_at else None,
            'stop_loss': self.stop_loss,
            'take_profit': self.take_profit
        }


class OrderManager:
    """
    Manages order lifecycle.
    
    Features:
    - Order creation and validation
    - Order routing to broker
    - Order status tracking
    - Fill handling
    - Bracket order management
    """
    
    def __init__(self, broker=None):
        """
        Initialize order manager.
        
        Args:
            broker: Broker instance for order execution
        """
        self.broker = broker
        self.orders: Dict[str, Order] = {}
        self.active_orders: Dict[str, Order] = {}
        self.filled_orders: List[Order] = []
        
        # Callbacks
        self.on_fill: Optional[Callable[[Order], None]] = None
        self.on_cancel: Optional[Callable[[Order], None]] = None
        self.on_reject: Optional[Callable[[Order, str], None]] = None
        
    def set_broker(self, broker):
        """Set the broker for order execution."""
        self.broker = broker
        
    def create_order(
        self,
        user_id: int,
        symbol: str,
        side: str,
        quantity: float,
        order_type: str = "market",
        price: float = None,
        stop_price: float = None,
        stop_loss: float = None,
        take_profit: float = None,
        **kwargs
    ) -> Order:
        """
        Create a new order.
        
        Args:
            user_id: ID of the user
            symbol: Trading symbol
            side: 'buy' or 'sell'
            quantity: Number of units
            order_type: 'market', 'limit', 'stop', 'stop_limit'
            price: Limit price (required for limit orders)
            stop_price: Stop price (required for stop orders)
            stop_loss: Stop loss price for bracket order
            take_profit: Take profit price for bracket order
            
        Returns:
            Created Order object
        """
        order = Order(
            user_id=user_id,
            symbol=symbol,
            side=OrderSide(side.lower()),
            order_type=OrderType(order_type.lower()),
            quantity=quantity,
            price=price,
            stop_price=stop_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            **kwargs
        )
        
        # Validate
        if not self._validate_order(order):
            order.status = OrderStatus.REJECTED
            return order
        
        self.orders[order.order_id] = order
        self.active_orders[order.order_id] = order
        
        logger.info(f"Created order: {order.order_id} {order.side.value} {order.quantity} {order.symbol}")
        return order
    
    def submit_order(self, order: Order) -> bool:
        """
        Submit order to broker.
        
        Returns:
            True if submitted successfully
        """
        if self.broker is None:
            logger.error("No broker configured")
            order.status = OrderStatus.REJECTED
            return False
        
        if order.status != OrderStatus.PENDING:
            logger.warning(f"Order {order.order_id} not in pending status")
            return False
        
        try:
            success = self.broker.submit_order(order)
            
            if success:
                order.status = OrderStatus.SUBMITTED
                logger.info(f"Submitted order: {order.order_id}")
            else:
                order.status = OrderStatus.REJECTED
                logger.warning(f"Order rejected: {order.order_id}")
                
            return success
            
        except Exception as e:
            logger.error(f"Error submitting order: {e}")
            order.status = OrderStatus.REJECTED
            return False
    
    def cancel_order(self, order_id: str) -> bool:
        """
        Cancel an active order.
        
        Returns:
            True if cancelled successfully
        """
        if order_id not in self.active_orders:
            logger.warning(f"Order {order_id} not found or not active")
            return False
        
        order = self.active_orders[order_id]
        
        if self.broker:
            success = self.broker.cancel_order(order)
        else:
            success = True
        
        if success:
            order.status = OrderStatus.CANCELLED
            del self.active_orders[order_id]
            
            if self.on_cancel:
                self.on_cancel(order)
            
            logger.info(f"Cancelled order: {order_id}")
        
        return success
    
    def handle_fill(
        self,
        order_id: str,
        filled_qty: float,
        filled_price: float
    ):
        """
        Handle order fill event.
        
        Called by broker when order is filled.
        """
        if order_id not in self.orders:
            logger.warning(f"Unknown order filled: {order_id}")
            return
        
        order = self.orders[order_id]
        
        # Update fill info (handle partial fills)
        prev_filled = order.filled_quantity
        order.filled_quantity += filled_qty
        order.filled_price = (
            (prev_filled * order.filled_price + filled_qty * filled_price) /
            order.filled_quantity
        )
        
        if order.filled_quantity >= order.quantity:
            order.status = OrderStatus.FILLED
            order.filled_at = datetime.now()
            
            if order_id in self.active_orders:
                del self.active_orders[order_id]
            self.filled_orders.append(order)
            
            logger.info(f"Order filled: {order_id} @ {order.filled_price}")
            
            # Trigger callback
            if self.on_fill:
                self.on_fill(order)
            
            # Create bracket orders if specified
            if order.stop_loss or order.take_profit:
                self._create_bracket_orders(order)
        else:
            order.status = OrderStatus.PARTIAL
            logger.info(f"Order partially filled: {order_id} {order.filled_quantity}/{order.quantity}")
    
    def _validate_order(self, order: Order) -> bool:
        """Validate order parameters."""
        if order.quantity <= 0:
            logger.error("Invalid quantity")
            return False
        
        if order.order_type == OrderType.LIMIT and order.price is None:
            logger.error("Limit order requires price")
            return False
        
        if order.order_type in [OrderType.STOP, OrderType.STOP_LIMIT] and order.stop_price is None:
            logger.error("Stop order requires stop_price")
            return False
        
        return True
    
    def _create_bracket_orders(self, parent_order: Order):
        """Create stop loss and take profit orders for a filled order."""
        exit_side = OrderSide.SELL if parent_order.side == OrderSide.BUY else OrderSide.BUY
        
        if parent_order.stop_loss:
            sl_order = self.create_order(
                symbol=parent_order.symbol,
                side=exit_side.value,
                quantity=parent_order.filled_quantity,
                order_type="stop",
                stop_price=parent_order.stop_loss,
                notes=f"SL for {parent_order.order_id}"
            )
            self.submit_order(sl_order)
        
        if parent_order.take_profit:
            tp_order = self.create_order(
                symbol=parent_order.symbol,
                side=exit_side.value,
                quantity=parent_order.filled_quantity,
                order_type="limit",
                price=parent_order.take_profit,
                notes=f"TP for {parent_order.order_id}"
            )
            self.submit_order(tp_order)
    
    def get_order(self, order_id: str) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)
    
    def get_active_orders(self, symbol: str = None) -> List[Order]:
        """Get all active orders, optionally filtered by symbol."""
        orders = list(self.active_orders.values())
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def get_filled_orders(self, symbol: str = None) -> List[Order]:
        """Get all filled orders, optionally filtered by symbol."""
        orders = self.filled_orders
        if symbol:
            orders = [o for o in orders if o.symbol == symbol]
        return orders
    
    def cancel_all_orders(self, symbol: str = None) -> int:
        """
        Cancel all active orders.
        
        Args:
            symbol: Only cancel orders for this symbol
            
        Returns:
            Number of orders cancelled
        """
        cancelled = 0
        order_ids = list(self.active_orders.keys())
        
        for order_id in order_ids:
            order = self.active_orders[order_id]
            if symbol and order.symbol != symbol:
                continue
            if self.cancel_order(order_id):
                cancelled += 1
        
        return cancelled

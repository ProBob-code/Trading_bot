import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from config import settings

class BrokerInterface(ABC):
    """Abstract interface for all brokers (Binance, Alpaca, etc.)"""
    @abstractmethod
    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        pass

    @abstractmethod
    def get_balance(self) -> float:
        pass

class PaperBroker(BrokerInterface):
    """Simple paper trading broker for simulation."""
    def __init__(self, initial_balance: float = 100000.0):
        self.balance = initial_balance
        self.positions = {}

    def place_order(self, symbol: str, side: str, order_type: str, quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        # Simulated order confirmation
        return {
            "status": "FILLED",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "commission": quantity * price * settings.DEFAULT_TRANSACTION_FEE if price else 0
        }

    def get_balance(self) -> float:
        return self.balance

class ExecutionEngine:
    """
    Orchestrates order execution and interacts with brokers.
    """
    def __init__(self, broker: BrokerInterface):
        self.broker = broker
        self.logger = logging.getLogger("ExecutionEngine")

    def execute_signal(self, symbol: str, signal: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates a signal into an execution order.
        """
        if signal['type'] == 'NONE':
            return {"status": "SKIPPED", "reason": "No signal"}
            
        side = "BUY" if signal['type'] == 'LONG' else "SHORT"
        quantity = signal.get('size', 0)
        price = signal.get('entry_price')

        if quantity <= 0:
            return {"status": "FAILED", "reason": "Invalid quantity"}

        try:
            order_result = self.broker.place_order(
                symbol=symbol,
                side=side,
                order_type="MARKET",
                quantity=quantity,
                price=price
            )
            self.logger.info(f"Executed {side} order for {symbol}: {order_result}")
            return order_result
        except Exception as e:
            self.logger.error(f"Execution failed for {symbol}: {e}")
            return {"status": "FAILED", "error": str(e)}

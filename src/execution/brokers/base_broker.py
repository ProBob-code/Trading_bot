"""
Base Broker Interface
=====================

Abstract base class for all broker implementations.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional
from loguru import logger


class BaseBroker(ABC):
    """
    Abstract base class for broker implementations.
    
    All brokers must implement these methods.
    """
    
    def __init__(self, api_key: str = "", secret_key: str = "", **kwargs):
        """Initialize broker with credentials."""
        self.api_key = api_key
        self.secret_key = secret_key
        self.is_connected = False
        
    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to the broker.
        
        Returns:
            True if connection successful
        """
        pass
    
    @abstractmethod
    def disconnect(self):
        """Disconnect from the broker."""
        pass
    
    @abstractmethod
    def get_account_info(self) -> Dict:
        """
        Get account information.
        
        Returns:
            Account details including balance, buying power, etc.
        """
        pass
    
    @abstractmethod
    def get_positions(self) -> List[Dict]:
        """
        Get current positions.
        
        Returns:
            List of position dictionaries
        """
        pass
    
    @abstractmethod
    def submit_order(self, order) -> bool:
        """
        Submit an order.
        
        Args:
            order: Order object
            
        Returns:
            True if order submitted successfully
        """
        pass
    
    @abstractmethod
    def cancel_order(self, order) -> bool:
        """
        Cancel an order.
        
        Args:
            order: Order object
            
        Returns:
            True if cancellation successful
        """
        pass
    
    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        """
        Get order status.
        
        Args:
            order_id: Order ID
            
        Returns:
            Order status dictionary
        """
        pass
    
    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Quote dictionary with bid, ask, last, etc.
        """
        pass
    
    def get_multiple_quotes(self, symbols: List[str]) -> Dict[str, Dict]:
        """
        Get quotes for multiple symbols.
        
        Default implementation calls get_quote for each.
        Override for more efficient batch requests.
        """
        quotes = {}
        for symbol in symbols:
            try:
                quotes[symbol] = self.get_quote(symbol)
            except Exception as e:
                logger.error(f"Error getting quote for {symbol}: {e}")
        return quotes
    
    def is_market_open(self) -> bool:
        """
        Check if market is open.
        
        Override for specific market hours checking.
        """
        return True

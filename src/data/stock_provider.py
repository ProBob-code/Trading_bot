"""
Stock Data Provider
===================

Live and historical stock data using Yahoo Finance API.
Supports US stocks and international markets.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, Optional
from loguru import logger
import urllib3

# Suppress SSL warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)


class YahooFinanceProvider:
    """
    Stock data provider using Yahoo Finance.
    
    Features:
    - Free API (no key required)
    - Historical daily/intraday data
    - Real-time quotes (delayed 15 min)
    - Works with US, Indian, and global stocks
    """
    
    BASE_URL = "https://query1.finance.yahoo.com/v8/finance/chart"
    
    # Popular stocks
    STOCKS = {
        # US Stocks
        "AAPL": "Apple Inc.",
        "TSLA": "Tesla Inc.",
        "GOOGL": "Alphabet Inc.",
        "MSFT": "Microsoft Corp.",
        "AMZN": "Amazon.com Inc.",
        "NVDA": "NVIDIA Corp.",
        "META": "Meta Platforms",
        # Indian Stocks (NSE)
        "RELIANCE.NS": "Reliance Industries",
        "TCS.NS": "Tata Consultancy",
        "INFY.NS": "Infosys",
        "HDFCBANK.NS": "HDFC Bank",
    }
    
    # Interval mapping (Yahoo format)
    INTERVALS = {
        "1m": "1m",
        "5m": "5m",
        "15m": "15m",
        "1h": "60m",
        "1d": "1d",
    }
    
    def __init__(self):
        """Initialize Yahoo Finance provider."""
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.session.verify = False
    
    def get_current_quote(self, symbol: str) -> Dict:
        """
        Get current stock quote.
        
        Args:
            symbol: Stock ticker (e.g., AAPL, TSLA)
            
        Returns:
            Dict with price info
        """
        try:
            url = f"{self.BASE_URL}/{symbol}"
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            quote = data['chart']['result'][0]
            meta = quote['meta']
            
            return {
                'symbol': symbol,
                'price': meta.get('regularMarketPrice', 0),
                'previous_close': meta.get('previousClose', 0),
                'change': meta.get('regularMarketPrice', 0) - meta.get('previousClose', 0),
                'change_pct': ((meta.get('regularMarketPrice', 0) / meta.get('previousClose', 1)) - 1) * 100,
                'currency': meta.get('currency', 'USD'),
                'exchange': meta.get('exchangeName', ''),
                'market_state': meta.get('marketState', 'CLOSED'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching quote for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}
    
    def get_historical_data(
        self,
        symbol: str,
        interval: str = "1d",
        period: str = "1mo",
        limit: int = 100
    ) -> pd.DataFrame:
        """
        Get historical OHLCV data.
        
        Args:
            symbol: Stock ticker
            interval: Candle interval (1m, 5m, 15m, 1h, 1d)
            period: Time period (1d, 5d, 1mo, 3mo, 1y)
            limit: Max number of candles
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            yf_interval = self.INTERVALS.get(interval, "1d")
            
            # Adjust period based on interval
            if interval in ['1m', '5m']:
                period = "7d"  # Max for 1m/5m
            elif interval == '15m':
                period = "60d"
            elif interval == '1h':
                period = "730d"
            
            url = f"{self.BASE_URL}/{symbol}"
            params = {
                "interval": yf_interval,
                "range": period
            }
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            result = data['chart']['result'][0]
            timestamps = result['timestamp']
            ohlcv = result['indicators']['quote'][0]
            
            df = pd.DataFrame({
                'time': pd.to_datetime(timestamps, unit='s'),
                'open': ohlcv['open'],
                'high': ohlcv['high'],
                'low': ohlcv['low'],
                'close': ohlcv['close'],
                'volume': ohlcv['volume']
            })
            
            # Set index
            df.set_index('time', inplace=True)
            df.dropna(inplace=True)
            
            # Limit results
            if len(df) > limit:
                df = df.tail(limit)
            
            logger.info(f"Fetched {len(df)} candles for {symbol} ({interval})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_market_hours(self, symbol: str) -> Dict:
        """Check if market is open."""
        quote = self.get_current_quote(symbol)
        
        state = quote.get('market_state', 'CLOSED')
        is_open = state in ['REGULAR', 'PRE', 'POST']
        
        return {
            'symbol': symbol,
            'is_open': is_open,
            'state': state,
            'exchange': quote.get('exchange', '')
        }
    
    @staticmethod
    def get_available_stocks() -> Dict[str, str]:
        """Get available stock symbols."""
        return YahooFinanceProvider.STOCKS


class LiveStockFeed:
    """
    Live stock data feed for auto-trading.
    
    Note: Stock data has 15-minute delay on free API.
    """
    
    def __init__(
        self,
        symbol: str = "AAPL",
        interval: str = "1m",
        history_bars: int = 100
    ):
        """Initialize live stock feed."""
        self.symbol = symbol
        self.interval = interval
        self.history_bars = history_bars
        self.provider = YahooFinanceProvider()
        self.df: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
    
    def refresh(self) -> pd.DataFrame:
        """Refresh data."""
        self.df = self.provider.get_historical_data(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.history_bars
        )
        self.last_update = datetime.now()
        return self.df
    
    def get_current_price(self) -> float:
        """Get current price."""
        data = self.provider.get_current_quote(self.symbol)
        return data.get('price', 0)
    
    def is_market_open(self) -> bool:
        """Check if market is open."""
        data = self.provider.get_market_hours(self.symbol)
        return data.get('is_open', False)

"""
Crypto Data Provider
====================

Live cryptocurrency data from Binance API.
Provides 24/7 real-time price data for crypto trading.
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional
import time
from loguru import logger


class BinanceCryptoProvider:
    """
    Cryptocurrency data provider using Binance public API.
    
    Features:
    - Real-time price data (no API key required)
    - Historical klines/candlesticks
    - 24/7 market data (crypto never sleeps!)
    
    Supported pairs: BTCUSDT, ETHUSDT, BNBUSDT, SOLUSDT, XRPUSDT, etc.
    """
    
    BASE_URL = "https://api.binance.com/api/v3"
    
    # Popular crypto pairs
    PAIRS = {
        "BTCUSDT": "Bitcoin/USDT",
        "ETHUSDT": "Ethereum/USDT",
        "BNBUSDT": "BNB/USDT",
        "SOLUSDT": "Solana/USDT",
        "XRPUSDT": "XRP/USDT",
        "ADAUSDT": "Cardano/USDT",
        "DOGEUSDT": "Dogecoin/USDT",
        "MATICUSDT": "Polygon/USDT",
        "DOTUSDT": "Polkadot/USDT",
        "AVAXUSDT": "Avalanche/USDT",
    }
    
    # Interval mappings
    INTERVALS = {
        "1m": "1 minute",
        "5m": "5 minutes",
        "15m": "15 minutes",
        "1h": "1 hour",
        "4h": "4 hours",
        "1d": "1 day",
    }
    
    def __init__(self):
        """Initialize Binance provider."""
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'TradingBot/1.0'
        })
        # Disable SSL verification (workaround for Windows certificate issues)
        self.session.verify = False
        # Suppress InsecureRequestWarning
        import urllib3
        urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    
    def get_current_price(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Get current price for a symbol.
        
        Args:
            symbol: Trading pair (e.g., BTCUSDT)
            
        Returns:
            Dict with price info
        """
        try:
            url = f"{self.BASE_URL}/ticker/price"
            response = self.session.get(url, params={"symbol": symbol.upper()}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price': float(data['price']),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching price for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'error': str(e)}
    
    def get_ticker_24h(self, symbol: str = "BTCUSDT") -> Dict:
        """
        Get 24-hour ticker statistics.
        
        Args:
            symbol: Trading pair
            
        Returns:
            Dict with 24h stats
        """
        try:
            url = f"{self.BASE_URL}/ticker/24hr"
            response = self.session.get(url, params={"symbol": symbol.upper()}, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': data['symbol'],
                'price': float(data['lastPrice']),
                'price_change': float(data['priceChange']),
                'price_change_pct': float(data['priceChangePercent']),
                'high_24h': float(data['highPrice']),
                'low_24h': float(data['lowPrice']),
                'volume_24h': float(data['volume']),
                'quote_volume_24h': float(data['quoteVolume']),
                'open_price': float(data['openPrice']),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error fetching 24h ticker for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_historical_klines(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1h",
        limit: int = 500,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None
    ) -> pd.DataFrame:
        """
        Get historical klines/candlestick data.
        
        Args:
            symbol: Trading pair
            interval: Kline interval (1m, 5m, 15m, 1h, 4h, 1d)
            limit: Number of klines (max 1000)
            start_time: Start datetime
            end_time: End datetime
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            url = f"{self.BASE_URL}/klines"
            
            params = {
                "symbol": symbol.upper(),
                "interval": interval,
                "limit": min(limit, 1000)
            }
            
            if start_time:
                params["startTime"] = int(start_time.timestamp() * 1000)
            if end_time:
                params["endTime"] = int(end_time.timestamp() * 1000)
            
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=[
                'open_time', 'open', 'high', 'low', 'close', 'volume',
                'close_time', 'quote_volume', 'trades', 'taker_buy_base',
                'taker_buy_quote', 'ignore'
            ])
            
            # Convert types
            df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
            df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
            
            for col in ['open', 'high', 'low', 'close', 'volume', 'quote_volume']:
                df[col] = df[col].astype(float)
            
            df['trades'] = df['trades'].astype(int)
            
            # Set index
            df.set_index('open_time', inplace=True)
            df.index.name = None
            
            # Select only required columns
            df = df[['open', 'high', 'low', 'close', 'volume']]
            
            logger.info(f"Fetched {len(df)} klines for {symbol} ({interval})")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching klines for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_order_book(self, symbol: str = "BTCUSDT", limit: int = 20) -> Dict:
        """
        Get order book / market depth.
        
        Args:
            symbol: Trading pair
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000)
            
        Returns:
            Dict with bids and asks
        """
        try:
            url = f"{self.BASE_URL}/depth"
            response = self.session.get(url, params={
                "symbol": symbol.upper(),
                "limit": limit
            }, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return {
                'symbol': symbol,
                'bids': [{'price': float(b[0]), 'volume': float(b[1])} for b in data['bids'][:5]],
                'asks': [{'price': float(a[0]), 'volume': float(a[1])} for a in data['asks'][:5]],
                'timestamp': datetime.now().isoformat()
            }
        except Exception as e:
            logger.error(f"Error fetching order book for {symbol}: {e}")
            return {'symbol': symbol, 'bids': [], 'asks': [], 'error': str(e)}
    
    def get_recent_trades(self, symbol: str = "BTCUSDT", limit: int = 10) -> List[Dict]:
        """
        Get recent trades.
        
        Args:
            symbol: Trading pair
            limit: Number of trades
            
        Returns:
            List of recent trades
        """
        try:
            url = f"{self.BASE_URL}/trades"
            response = self.session.get(url, params={
                "symbol": symbol.upper(),
                "limit": limit
            }, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            return [
                {
                    'id': t['id'],
                    'price': float(t['price']),
                    'qty': float(t['qty']),
                    'time': datetime.fromtimestamp(t['time'] / 1000),
                    'side': 'BUY' if t['isBuyerMaker'] else 'SELL'
                }
                for t in data
            ]
        except Exception as e:
            logger.error(f"Error fetching trades for {symbol}: {e}")
            return []
    
    @staticmethod
    def get_available_pairs() -> Dict[str, str]:
        """Get available crypto pairs."""
        return BinanceCryptoProvider.PAIRS
    
    @staticmethod
    def get_available_intervals() -> Dict[str, str]:
        """Get available intervals."""
        return BinanceCryptoProvider.INTERVALS


class LiveCryptoFeed:
    """
    Live crypto data feed for auto-trading.
    
    Continuously fetches fresh data at specified intervals.
    """
    
    def __init__(
        self,
        symbol: str = "BTCUSDT",
        interval: str = "1m",
        history_bars: int = 100
    ):
        """
        Initialize live feed.
        
        Args:
            symbol: Trading pair
            interval: Kline interval
            history_bars: Number of historical bars to maintain
        """
        self.symbol = symbol
        self.interval = interval
        self.history_bars = history_bars
        self.provider = BinanceCryptoProvider()
        self.df: Optional[pd.DataFrame] = None
        self.last_update: Optional[datetime] = None
    
    def refresh(self) -> pd.DataFrame:
        """
        Refresh data from Binance.
        
        Returns:
            Updated DataFrame with latest data
        """
        self.df = self.provider.get_historical_klines(
            symbol=self.symbol,
            interval=self.interval,
            limit=self.history_bars
        )
        self.last_update = datetime.now()
        return self.df
    
    def get_current_price(self) -> float:
        """Get current price."""
        data = self.provider.get_current_price(self.symbol)
        return data.get('price', 0)
    
    def get_market_depth(self) -> Dict:
        """Get current market depth."""
        return self.provider.get_order_book(self.symbol)
    
    def get_ticker(self) -> Dict:
        """Get 24h ticker stats."""
        return self.provider.get_ticker_24h(self.symbol)

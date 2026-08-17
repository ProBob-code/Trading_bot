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

    # UI ticker → Yahoo ticker. Without this, FX pairs, metals and NSE names
    # 404 on Yahoo and fall through to a static price table, which is how the
    # panel ended up showing a two-year-old USDINR while the chart showed spot.
    YAHOO_SYMBOL_MAP = {
        # ── FX (Yahoo suffixes crosses with "=X") ──
        "EURUSD": "EURUSD=X",
        "GBPUSD": "GBPUSD=X",
        "USDJPY": "USDJPY=X",
        "USDINR": "USDINR=X",
        "AUDUSD": "AUDUSD=X",
        "USDCAD": "USDCAD=X",
        "USDCHF": "USDCHF=X",
        "NZDUSD": "NZDUSD=X",
        "EURGBP": "EURGBP=X",
        "EURJPY": "EURJPY=X",
        "GBPJPY": "GBPJPY=X",
        # ── Commodities (front-month futures) ──
        "XAUUSD": "GC=F",
        "XAGUSD": "SI=F",
        "XCUUSD": "HG=F",
        "XBRUSD": "BZ=F",
        "XTIUSD": "CL=F",
        "XNGUSD": "NG=F",
        # ── NSE equities ──
        "RELIANCE": "RELIANCE.NS",
        "TCS": "TCS.NS",
        "INFY": "INFY.NS",
        "HDFCBANK": "HDFCBANK.NS",
        "TATAMOTORS": "TATAMOTORS.NS",
        "ICICIBANK": "ICICIBANK.NS",
        "SBIN": "SBIN.NS",
        "WIPRO": "WIPRO.NS",
    }

    # Last-resort prices, used ONLY when the live fetch fails, so the UI shows
    # something rather than a zero. Anything served from here is flagged with
    # source='fallback' so callers can tell it apart from a real quote.
    FALLBACK_PRICES = {
        "XAUUSD": 2340.0, "EURUSD": 1.085, "GBPUSD": 1.254, "USDJPY": 155.20,
        "AUDUSD": 0.655, "USDINR": 83.45, "USDCAD": 1.365, "XAGUSD": 28.30,
        "XBRUSD": 83.50, "XCUUSD": 4.55, "XTIUSD": 78.50, "XNGUSD": 2.10,
        "RELIANCE": 2950.0, "TCS": 3850.0, "HDFCBANK": 1520.0, "TATAMOTORS": 980.0,
    }

    # Instruments quoted in Indian rupees rather than US dollars.
    INR_QUOTED = {"RELIANCE", "TCS", "INFY", "HDFCBANK", "TATAMOTORS",
                  "ICICIBANK", "SBIN", "WIPRO"}

    @classmethod
    def to_yahoo_symbol(cls, symbol: str) -> str:
        """Translate a UI ticker into the ticker Yahoo Finance actually serves."""
        return cls.YAHOO_SYMBOL_MAP.get(symbol.strip().upper(), symbol.strip().upper())

    @classmethod
    def quote_currency(cls, symbol: str) -> str:
        """
        The currency a symbol's price is denominated in.

        USDINR is quoted in rupees, USDJPY in yen, NSE stocks in rupees —
        labelling all of them "USD" is what put a "$" in front of a rupee price.
        """
        s = symbol.strip().upper()
        if s in cls.INR_QUOTED:
            return "INR"
        # 6-char FX cross: the trailing three characters are the quote currency.
        if len(s) == 6 and s.isalpha():
            return s[3:]
        return "USD"


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
        symbol = symbol.strip().upper()

        try:
            # FX pairs, metals and NSE names need translating before Yahoo will
            # serve them — see YAHOO_SYMBOL_MAP.
            yahoo_symbol = self.to_yahoo_symbol(symbol)

            url = f"{self.BASE_URL}/{yahoo_symbol}"
            logger.info(f"🌍 Yahoo Finance URL for {symbol}: {url}")
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()

            quote = data['chart']['result'][0]
            meta = quote['meta']

            price = meta.get('regularMarketPrice') or 0
            prev_close = meta.get('previousClose') or meta.get('chartPreviousClose') or 0

            if not price:
                raise ValueError(f"Yahoo returned no price for {yahoo_symbol}")

            return {
                'symbol': symbol,
                'price': price,
                'previous_close': prev_close,
                'change': price - prev_close,
                'change_pct': ((price / prev_close) - 1) * 100 if prev_close else 0.0,
                # Trust Yahoo's currency, but fall back to our own mapping —
                # the price must never be labelled in a currency it isn't in.
                'currency': meta.get('currency') or self.quote_currency(symbol),
                'exchange': meta.get('exchangeName', ''),
                'market_state': meta.get('marketState', 'CLOSED'),
                'source': 'yahoo',
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.warning(f"Live quote failed for {symbol} ({e}) — using fallback price")
            return self._fallback_quote(symbol, str(e))

    def _fallback_quote(self, symbol: str, error: str = '') -> Dict:
        """
        Static-baseline quote, used only when the live fetch fails.

        Marked source='fallback' and flagged in the payload so nothing downstream
        mistakes a placeholder for a real market price.
        """
        base = self.FALLBACK_PRICES.get(symbol)
        if base is None:
            return {'symbol': symbol, 'price': 0, 'source': 'unavailable', 'error': error}

        import random
        price = base + random.uniform(-base * 0.002, base * 0.002)
        return {
            'symbol': symbol,
            'price': price,
            'previous_close': base,
            'change': price - base,
            'change_pct': ((price / base) - 1) * 100,
            'currency': self.quote_currency(symbol),
            'exchange': 'FALLBACK',
            'market_state': 'UNKNOWN',
            'source': 'fallback',
            'stale': True,
            'timestamp': datetime.now()
        }
    
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
            
            # Same translation as get_current_quote — otherwise a forex or NSE
            # bot silently gets an empty frame and never generates a signal.
            url = f"{self.BASE_URL}/{self.to_yahoo_symbol(symbol)}"
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

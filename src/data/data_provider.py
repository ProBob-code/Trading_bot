"""
Data Provider Module
====================

Abstract interface for market data with implementations for various sources.
"""

from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Union
import os
import ssl
import pandas as pd
from loguru import logger

# Disable SSL verification for corporate networks with proxy certificates
# This is needed when behind corporate firewalls with self-signed certs
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
os.environ['PYTHONHTTPSVERIFY'] = '0'
os.environ['CURL_CA_BUNDLE'] = ''
os.environ['REQUESTS_CA_BUNDLE'] = ''

# Patch SSL context for requests
try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    pass
else:
    ssl._create_default_https_context = _create_unverified_https_context

try:
    import yfinance as yf
    # Disable SSL verification in yfinance session
    import requests
    session = requests.Session()
    session.verify = False
    yf.utils.requests = session
except ImportError:
    yf = None


class DataProvider(ABC):
    """
    Abstract base class for market data providers.
    
    All data providers must implement these methods to ensure
    consistent data access across different sources.
    """
    
    @abstractmethod
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """
        Fetch historical OHLCV data.
        
        Args:
            symbol: Ticker symbol (e.g., 'AAPL', 'RELIANCE.NS')
            start_date: Start date for data
            end_date: End date for data
            interval: Data interval (1m, 5m, 15m, 30m, 1h, 1d)
            
        Returns:
            DataFrame with columns: open, high, low, close, volume
            Index should be datetime
        """
        pass
    
    @abstractmethod
    def get_realtime_quote(self, symbol: str) -> Dict:
        """
        Get real-time quote for a symbol.
        
        Args:
            symbol: Ticker symbol
            
        Returns:
            Dictionary with bid, ask, last, volume, etc.
        """
        pass
    
    @abstractmethod
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            start_date: Start date
            end_date: End date
            interval: Data interval
            
        Returns:
            Dictionary mapping symbol to DataFrame
        """
        pass
    
    def _normalize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize column names to lowercase."""
        df.columns = [c.lower() for c in df.columns]
        
        # Ensure required columns exist
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
                
        return df[required]


class YFinanceProvider(DataProvider):
    """
    Data provider using Yahoo Finance (yfinance).
    
    Supports stocks (US, Indian), ETFs, crypto, and forex.
    Free but has rate limits and delayed data.
    """
    
    def __init__(self):
        if yf is None:
            raise ImportError("yfinance is required. Install with: pip install yfinance")
            
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Yahoo Finance."""
        try:
            # Convert interval format
            yf_interval = self._convert_interval(interval)
            
            # Download data
            ticker = yf.Ticker(symbol)
            df = ticker.history(
                start=start_date,
                end=end_date,
                interval=yf_interval,
                auto_adjust=True
            )
            
            if df.empty:
                logger.warning(f"No data returned for {symbol}")
                return pd.DataFrame()
            
            # Normalize and return
            df = self._normalize_columns(df)
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote (delayed for free tier)."""
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            return {
                'symbol': symbol,
                'last': info.get('regularMarketPrice', 0),
                'bid': info.get('bid', 0),
                'ask': info.get('ask', 0),
                'volume': info.get('regularMarketVolume', 0),
                'previous_close': info.get('previousClose', 0),
                'open': info.get('regularMarketOpen', 0),
                'high': info.get('dayHigh', 0),
                'low': info.get('dayLow', 0),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols."""
        result = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date, interval)
            if not df.empty:
                result[symbol] = df
                
        return result
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval format to yfinance format."""
        mapping = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '4h': '4h', '1d': '1d', '1w': '1wk', '1mo': '1mo'
        }
        return mapping.get(interval.lower(), interval)


class AlpacaProvider(DataProvider):
    """
    Data provider using Alpaca Markets API.
    
    Supports US stocks with real-time data (with subscription).
    Free paper trading with delayed data.
    """
    
    def __init__(self, api_key: str, secret_key: str, base_url: str = None):
        try:
            from alpaca_trade_api import REST
            self.api = REST(api_key, secret_key, base_url)
        except ImportError:
            raise ImportError("alpaca-trade-api required. Install with: pip install alpaca-trade-api")
    
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical bars from Alpaca."""
        try:
            timeframe = self._convert_interval(interval)
            
            bars = self.api.get_bars(
                symbol,
                timeframe,
                start=start_date,
                end=end_date
            ).df
            
            if bars.empty:
                return pd.DataFrame()
                
            bars = self._normalize_columns(bars)
            return bars
            
        except Exception as e:
            logger.error(f"Alpaca error for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get latest trade and quote."""
        try:
            trade = self.api.get_latest_trade(symbol)
            quote = self.api.get_latest_quote(symbol)
            
            return {
                'symbol': symbol,
                'last': trade.price,
                'bid': quote.bid_price,
                'ask': quote.ask_price,
                'volume': trade.size,
                'timestamp': trade.timestamp
            }
        except Exception as e:
            logger.error(f"Error getting Alpaca quote: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols efficiently."""
        result = {}
        try:
            timeframe = self._convert_interval(interval)
            
            bars = self.api.get_bars(
                symbols,
                timeframe,
                start=start_date,
                end=end_date
            )
            
            for symbol in symbols:
                if symbol in bars.df.index.get_level_values('symbol').unique():
                    df = bars.df.xs(symbol, level='symbol')
                    result[symbol] = self._normalize_columns(df)
                    
        except Exception as e:
            logger.error(f"Error fetching multiple symbols: {e}")
            
        return result
    
    def _convert_interval(self, interval: str):
        """Convert to Alpaca TimeFrame."""
        from alpaca_trade_api.rest import TimeFrame
        
        mapping = {
            '1m': TimeFrame.Minute,
            '5m': TimeFrame(5, TimeFrame.Minute),
            '15m': TimeFrame(15, TimeFrame.Minute),
            '30m': TimeFrame(30, TimeFrame.Minute),
            '1h': TimeFrame.Hour,
            '1d': TimeFrame.Day,
        }
        return mapping.get(interval.lower(), TimeFrame.Day)


class AlphaVantageProvider(DataProvider):
    """
    Data provider using Alpha Vantage API.
    
    Free tier: 25 requests/day
    Works well with corporate proxies.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize Alpha Vantage provider.
        
        Args:
            api_key: Alpha Vantage API key
        """
        self.api_key = api_key
        self.base_url = "https://www.alphavantage.co/query"
        
        # Use requests with SSL verification disabled for corporate networks
        import requests
        self.session = requests.Session()
        self.session.verify = False
        
    def get_historical_data(
        self,
        symbol: str,
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> pd.DataFrame:
        """Fetch historical data from Alpha Vantage."""
        try:
            # Determine function based on interval
            if interval in ['1m', '5m', '15m', '30m', '1h']:
                function = 'TIME_SERIES_INTRADAY'
                av_interval = self._convert_interval(interval)
                params = {
                    'function': function,
                    'symbol': symbol,
                    'interval': av_interval,
                    'apikey': self.api_key,
                    'outputsize': 'compact',  # Use compact for free tier (100 data points)
                    'datatype': 'json'
                }
            else:
                function = 'TIME_SERIES_DAILY'
                params = {
                    'function': function,
                    'symbol': symbol,
                    'apikey': self.api_key,
                    'outputsize': 'compact',  # Use compact for free tier (100 data points)
                    'datatype': 'json'
                }
            
            logger.info(f"Fetching {symbol} from Alpha Vantage...")
            response = self.session.get(self.base_url, params=params, timeout=30)
            
            if response.status_code != 200:
                logger.error(f"Alpha Vantage error: {response.status_code}")
                return pd.DataFrame()
            
            data = response.json()
            
            # Check for error messages
            if 'Error Message' in data:
                logger.error(f"Alpha Vantage error: {data['Error Message']}")
                return pd.DataFrame()
            
            if 'Note' in data:
                logger.warning(f"Alpha Vantage rate limit: {data['Note']}")
                return pd.DataFrame()
            
            # Parse the time series data
            time_series_key = None
            for key in data.keys():
                if 'Time Series' in key:
                    time_series_key = key
                    break
            
            if time_series_key is None:
                logger.error(f"No time series data found for {symbol}")
                return pd.DataFrame()
            
            time_series = data[time_series_key]
            
            # Convert to DataFrame
            df = pd.DataFrame.from_dict(time_series, orient='index')
            df.index = pd.to_datetime(df.index)
            df.sort_index(inplace=True)
            
            # Rename columns (Alpha Vantage uses numbered prefixes)
            column_mapping = {
                '1. open': 'open',
                '2. high': 'high',
                '3. low': 'low',
                '4. close': 'close',
                '5. volume': 'volume',
                '5. adjusted close': 'close',
                '6. volume': 'volume'
            }
            df.rename(columns=column_mapping, inplace=True)
            
            # Convert to numeric
            for col in ['open', 'high', 'low', 'close', 'volume']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter by date range
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            df = df[(df.index >= start_date) & (df.index <= end_date)]
            
            # Ensure required columns
            required = ['open', 'high', 'low', 'close', 'volume']
            for col in required:
                if col not in df.columns:
                    df[col] = 0
            
            logger.info(f"Fetched {len(df)} rows for {symbol} from Alpha Vantage")
            return df[required]
            
        except Exception as e:
            logger.error(f"Error fetching {symbol} from Alpha Vantage: {e}")
            return pd.DataFrame()
    
    def get_realtime_quote(self, symbol: str) -> Dict:
        """Get real-time quote from Alpha Vantage."""
        try:
            params = {
                'function': 'GLOBAL_QUOTE',
                'symbol': symbol,
                'apikey': self.api_key
            }
            
            response = self.session.get(self.base_url, params=params, timeout=30)
            data = response.json()
            
            if 'Global Quote' not in data:
                return {'symbol': symbol, 'error': 'No quote data'}
            
            quote = data['Global Quote']
            
            return {
                'symbol': symbol,
                'last': float(quote.get('05. price', 0)),
                'open': float(quote.get('02. open', 0)),
                'high': float(quote.get('03. high', 0)),
                'low': float(quote.get('04. low', 0)),
                'volume': int(quote.get('06. volume', 0)),
                'previous_close': float(quote.get('08. previous close', 0)),
                'change': float(quote.get('09. change', 0)),
                'change_pct': quote.get('10. change percent', '0%'),
                'timestamp': datetime.now()
            }
        except Exception as e:
            logger.error(f"Error getting Alpha Vantage quote: {e}")
            return {'symbol': symbol, 'error': str(e)}
    
    def get_multiple_symbols(
        self,
        symbols: List[str],
        start_date: Union[str, datetime],
        end_date: Union[str, datetime],
        interval: str = "1d"
    ) -> Dict[str, pd.DataFrame]:
        """Fetch data for multiple symbols (note: rate limited)."""
        result = {}
        
        for symbol in symbols:
            df = self.get_historical_data(symbol, start_date, end_date, interval)
            if not df.empty:
                result[symbol] = df
        
        return result
    
    def _convert_interval(self, interval: str) -> str:
        """Convert interval to Alpha Vantage format."""
        mapping = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '1h': '60min'
        }
        return mapping.get(interval.lower(), '60min')


def get_data_provider(provider_name: str, **kwargs) -> DataProvider:
    """
    Factory function to get appropriate data provider.
    
    Args:
        provider_name: Name of provider ('yfinance', 'alpaca', 'alphavantage')
        **kwargs: Provider-specific arguments
        
    Returns:
        DataProvider instance
    """
    providers = {
        'yfinance': lambda: YFinanceProvider(),
        'alpaca': lambda: AlpacaProvider(
            kwargs.get('api_key', ''),
            kwargs.get('secret_key', ''),
            kwargs.get('base_url')
        ),
        'alphavantage': lambda: AlphaVantageProvider(
            kwargs.get('api_key', '')
        ),
    }
    
    if provider_name not in providers:
        raise ValueError(f"Unknown provider: {provider_name}. Available: {list(providers.keys())}")
        
    return providers[provider_name]()

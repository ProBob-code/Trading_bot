"""
CoinGeckoProvider - crypto coverage for everything the exchanges have dropped.
=============================================================================

Binance is the primary crypto feed and is better at what it does: exact interval
candles, tick-accurate last price. What it cannot do is tell you anything about a
pair it has delisted - which is exactly how MATICUSDT ended up with a dead chart.

CoinGecko is the answer to that specific gap. It tracks ~17,000 coins including
migrated and retired ones, keeps the history attached across a rename (MATIC's
history lives on POL's coin id), needs no API key, and allows CORS.

It sits between Binance and Yahoo in the cascade: better crypto coverage than
Yahoo, coarser candle granularity than Binance.

Rate limits on the keyless tier are tight (roughly 5-15 calls/minute), so every
lookup here is cached aggressively. This provider is a fallback, not a firehose -
if it is being called often enough to hit a limit, something upstream is wrong.
"""

import threading
import time
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://api.coingecko.com/api/v3"

# Quote assets that mean "priced in dollars".
_STABLE_QUOTES = ("USDT", "BUSD", "USDC", "FDUSD", "TUSD", "USD")

# Ticker symbols are not unique on CoinGecko - dozens of tokens call themselves
# "BTC". Pinning the majors by coin id stops a scam token outranking the real
# asset, whatever the market-cap ordering happens to say today.
SYMBOL_OVERRIDES: Dict[str, str] = {
    "BTC": "bitcoin",
    "ETH": "ethereum",
    "USDT": "tether",
    "BNB": "binancecoin",
    "SOL": "solana",
    "XRP": "ripple",
    "ADA": "cardano",
    "DOGE": "dogecoin",
    "AVAX": "avalanche-2",
    "DOT": "polkadot",
    "LINK": "chainlink",
    "TRX": "tron",
    "MATIC": "polygon-ecosystem-token",   # migrated to POL; history follows the id
    "POL": "polygon-ecosystem-token",
    "LTC": "litecoin",
    "SHIB": "shiba-inu",
    "UNI": "uniswap",
    "ATOM": "cosmos",
    "XLM": "stellar",
    "NEAR": "near",
    "APT": "aptos",
    "ARB": "arbitrum",
    "OP": "optimism",
    "FIL": "filecoin",
    "LUNC": "terra-luna",
    "S": "sonic-3",                        # Fantom migrated to Sonic
    "FTM": "fantom",
}


class CoinGeckoProvider:
    """Keyless crypto quotes and candles, cached hard because the limits are low."""

    MAP_TTL = 21600      # coin list changes slowly; 6h is plenty
    QUOTE_TTL = 30       # a fallback quote does not need to be sub-second
    CANDLE_TTL = 120

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/json',
            'User-Agent': 'GoatBot/2.4 (+trading-terminal)',
        })
        self._lock = threading.Lock()
        self._symbol_map: Dict[str, str] = {}
        self._map_ts = 0.0
        self._quotes: Dict[str, dict] = {}
        self._candles: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Symbol -> coin id
    # ------------------------------------------------------------------
    @staticmethod
    def base_asset(symbol: str) -> str:
        """BTCUSDT -> BTC. Anything already bare is returned unchanged."""
        s = symbol.strip().upper()
        for q in _STABLE_QUOTES:
            if s.endswith(q) and len(s) > len(q):
                return s[: -len(q)]
        return s

    def _refresh_symbol_map(self) -> Dict[str, str]:
        """
        Build ticker -> coin id from the market-cap ranked list.

        Ordering by market cap matters: it is what makes an ambiguous ticker
        resolve to the asset a trader actually meant.
        """
        with self._lock:
            if self._symbol_map and (time.time() - self._map_ts) < self.MAP_TTL:
                return self._symbol_map

        mapping: Dict[str, str] = {}
        try:
            for page in (1, 2):
                resp = self.session.get(
                    f"{BASE_URL}/coins/markets",
                    params={'vs_currency': 'usd', 'order': 'market_cap_desc',
                            'per_page': 250, 'page': page, 'sparkline': 'false'},
                    timeout=10,
                )
                resp.raise_for_status()
                for row in resp.json():
                    sym = str(row.get('symbol', '')).upper()
                    # First occurrence wins - the list is already ranked.
                    if sym and sym not in mapping:
                        mapping[sym] = row['id']
        except Exception as e:
            logger.warning(f"[coingecko] coin list refresh failed: {e}")
            with self._lock:
                return self._symbol_map  # keep whatever we already had

        # Explicit pins always beat the ranking.
        mapping.update(SYMBOL_OVERRIDES)
        with self._lock:
            self._symbol_map = mapping
            self._map_ts = time.time()
        logger.info(f"[coingecko] symbol map refreshed: {len(mapping)} tickers")
        return mapping

    def coin_id(self, symbol: str) -> Optional[str]:
        """Resolve a trading symbol to a CoinGecko coin id."""
        base = self.base_asset(symbol)
        if base in SYMBOL_OVERRIDES:
            return SYMBOL_OVERRIDES[base]
        return self._refresh_symbol_map().get(base)

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------
    def get_current_quote(self, symbol: str) -> Dict:
        """
        Latest price + 24h change, shaped like the other providers so QuoteService
        can consume it without knowing who answered.
        """
        symbol = symbol.strip().upper()
        cached = self._quotes.get(symbol)
        if cached and (time.time() - cached['_ts']) < self.QUOTE_TTL:
            return dict(cached)

        cid = self.coin_id(symbol)
        if not cid:
            return {'symbol': symbol, 'price': 0, 'source': 'coingecko', 'error': 'unknown coin'}

        try:
            resp = self.session.get(
                f"{BASE_URL}/simple/price",
                params={'ids': cid, 'vs_currencies': 'usd', 'include_24hr_change': 'true',
                        'include_24hr_vol': 'true'},
                timeout=10,
            )
            resp.raise_for_status()
            row = resp.json().get(cid) or {}
            price = float(row.get('usd') or 0)
            if price <= 0:
                raise ValueError(f"no usd price for {cid}")

            change_pct = float(row.get('usd_24h_change') or 0)
            # CoinGecko gives the change but not the open; deriving it keeps the
            # payload shape identical to Binance's.
            prev_close = price / (1 + change_pct / 100) if change_pct != -100 else price

            quote = {
                'symbol': symbol,
                'price': price,
                'previous_close': prev_close,
                'change': price - prev_close,
                'change_pct': change_pct,
                # The /simple/price endpoint carries no intraday high/low. Reporting
                # 0 would be read as a real number, so leave them absent-as-zero and
                # let the caller show a dash.
                'high': 0.0,
                'low': 0.0,
                'volume': float(row.get('usd_24h_vol') or 0),
                'currency': 'USD',
                'exchange': 'COINGECKO',
                'market_state': 'REGULAR',
                'source': 'coingecko',
                '_ts': time.time(),
            }
            self._quotes[symbol] = quote
            return dict(quote)
        except Exception as e:
            logger.warning(f"[coingecko] quote failed for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'source': 'coingecko', 'error': str(e)}

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------
    # CoinGecko picks candle granularity from the day range - it is not selectable.
    #   days<=1 -> 30m,  days<=7 -> 4h,  days<=30 -> 4h,  more -> 4d
    # Mapping our interval to the smallest range that covers it gets the finest
    # candles CoinGecko will give for that timeframe.
    _INTERVAL_DAYS = {
        '1m': 1, '5m': 1, '15m': 1, '30m': 1,
        '1h': 7, '4h': 30, '1d': 365,
    }

    def get_historical_data(self, symbol: str, interval: str = '1m', limit: int = 300) -> pd.DataFrame:
        """
        OHLC candles. Returns an empty frame on failure so the caller moves on to
        the next source in the cascade.

        Note the granularity is CoinGecko's choice, not ours - a 1m request comes
        back as 30m candles. That is a deliberate trade: coarse candles on a
        delisted pair beat an empty chart, and the UI labels the source.
        """
        symbol = symbol.strip().upper()
        key = f"{symbol}|{interval}"
        cached = self._candles.get(key)
        if cached and (time.time() - cached['_ts']) < self.CANDLE_TTL:
            return cached['df'].copy()

        cid = self.coin_id(symbol)
        if not cid:
            return pd.DataFrame()

        days = self._INTERVAL_DAYS.get(interval, 1)
        try:
            resp = self.session.get(
                f"{BASE_URL}/coins/{cid}/ohlc",
                params={'vs_currency': 'usd', 'days': days},
                timeout=15,
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                return pd.DataFrame()

            df = pd.DataFrame(rows, columns=['time', 'open', 'high', 'low', 'close'])
            df['time'] = pd.to_datetime(df['time'], unit='ms')
            # The OHLC endpoint carries no volume. Zero is honest here - the native
            # chart simply draws no volume bars rather than inventing them.
            df['volume'] = 0.0
            df.set_index('time', inplace=True)
            df.dropna(inplace=True)
            if len(df) > limit:
                df = df.tail(limit)

            self._candles[key] = {'df': df, '_ts': time.time()}
            logger.info(f"[coingecko] {len(df)} candles for {symbol} ({cid}, {days}d)")
            return df.copy()
        except Exception as e:
            logger.warning(f"[coingecko] candles failed for {symbol}: {e}")
            return pd.DataFrame()


coingecko_provider = CoinGeckoProvider()

"""
QuoteService - the single source of truth for "what does this instrument cost right now".
=========================================================================================

Before this existed, a price reached the screen down three unrelated paths:

  * GET /api/v2/price/<symbol>       -> Binance /ticker/24hr (direct requests call)
  * the price_stream socket thread   -> crypto_provider.get_current_price()
    (/ticker/price) for the price, plus a *separate* /ticker/24hr call for the change
  * whichever of those happened to land last in a given DOM node

Three round-trips at three different moments produce three different numbers, so the
header, Market Watch and the watchlist each showed a different price for the same coin.
This module collapses them into one cached fetch: every caller asks here, and within
the TTL they all get the identical tick - same price, same change, same timestamp.

The TTL is deliberately short (1s). It is not there to save bandwidth; it is there so
that panels rendering within the same second cannot disagree.
"""

import threading
import time
from typing import Dict, Iterable, List, Optional

from loguru import logger

# How long a fetched tick is considered current. Anything asking inside this window
# gets the exact values the previous caller got - that identity is the whole point.
DEFAULT_TTL = 1.0

# Quote currencies that mean "this is a dollar-denominated crypto pair".
_STABLE_QUOTES = ('USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD')


def is_crypto_symbol(symbol: str) -> bool:
    """Crypto pairs are the ones the Binance provider can serve."""
    return symbol.upper().endswith(_STABLE_QUOTES)


class QuoteService:
    """Process-wide cache in front of the market data providers."""

    # How long a streamed tick is trusted without a REST refresh. Comfortably
    # longer than the stream's own cadence, short enough that a silently dead
    # socket falls back to polling within a few seconds.
    LIVE_TTL = 10.0

    def __init__(self, ttl: float = DEFAULT_TTL):
        self.ttl = ttl
        self._cache: Dict[str, dict] = {}
        self._lock = threading.Lock()
        # Per-symbol locks stop N panels asking for BTCUSDT at once from firing N
        # identical upstream requests - the first fetches, the rest wait and reuse.
        self._fetch_locks: Dict[str, threading.Lock] = {}
        self.crypto_provider = None      # Binance - primary crypto
        self.stock_provider = None       # Yahoo - equities/FX/commodities, and last resort
        self.coingecko_provider = None   # crypto coverage Binance has dropped
        self.smartapi_provider = None    # live NSE/BSE, when credentials are present

    def configure(self, crypto_provider=None, stock_provider=None,
                  coingecko_provider=None, smartapi_provider=None) -> None:
        """Inject the shared provider singletons built in api_server."""
        if crypto_provider is not None:
            self.crypto_provider = crypto_provider
        if stock_provider is not None:
            self.stock_provider = stock_provider
        if coingecko_provider is not None:
            self.coingecko_provider = coingecko_provider
        if smartapi_provider is not None:
            self.smartapi_provider = smartapi_provider

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def get(self, symbol: str, max_age: Optional[float] = None) -> dict:
        """
        Return the canonical tick for `symbol`.

        Shape is identical for every asset class so the frontend never has to know
        which provider answered:

            {symbol, price, change_pct, high_24h, low_24h, volume_24h, open,
             currency, source, stale, ts, success}
        """
        symbol = symbol.strip().upper()
        ttl = self.ttl if max_age is None else max_age

        cached = self._cached(symbol, ttl)
        if cached is not None:
            return cached

        lock = self._lock_for(symbol)
        with lock:
            # Someone may have refreshed it while we queued on the lock.
            cached = self._cached(symbol, ttl)
            if cached is not None:
                return cached

            tick = self._fetch(symbol)
            with self._lock:
                self._cache[symbol] = tick
            return dict(tick)

    def get_many(self, symbols: Iterable[str], max_age: Optional[float] = None) -> List[dict]:
        """Batch form of get() - one entry per requested symbol, order preserved."""
        return [self.get(s, max_age=max_age) for s in symbols]

    def ingest(self, tick: dict) -> Optional[dict]:
        """
        Accept a tick pushed by a live stream.

        Returns the stored tick, or None if it was rejected as out of order.
        Ordering uses the exchange's own event time, so a frame delayed in the
        network can never overwrite a newer price that already arrived.
        """
        if not tick:
            return None
        symbol = str(tick.get('symbol', '')).upper()
        try:
            price = float(tick.get('price') or 0)
        except (TypeError, ValueError):
            return None
        if not symbol or price <= 0:
            return None

        ts = float(tick.get('ts') or time.time())
        with self._lock:
            prev = self._cache.get(symbol)
            if prev and ts < prev.get('ts', 0):
                return None
            stored = dict(tick)
            stored['symbol'] = symbol
            stored['price'] = price
            stored['ts'] = ts
            stored['live'] = True
            self._cache[symbol] = stored
        return dict(stored)

    def peek(self, symbol: str) -> Optional[dict]:
        """Last known tick regardless of age, or None. Never triggers a fetch."""
        with self._lock:
            tick = self._cache.get(symbol.strip().upper())
            return dict(tick) if tick else None

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------
    def _cached(self, symbol: str, ttl: float) -> Optional[dict]:
        # ttl <= 0 means "I want a fresh fetch". Without this, Windows' ~16ms clock
        # resolution makes a just-cached tick's age read as exactly 0.0, so a
        # caller asking for max_age=0 would be handed the cache anyway.
        if ttl <= 0:
            return None
        with self._lock:
            tick = self._cache.get(symbol)
            if not tick:
                return None
            age = time.time() - tick['ts']
            # A tick pushed by the WebSocket is fresher than anything a REST call
            # could fetch, so it stays authoritative for LIVE_TTL rather than the
            # poll TTL - otherwise every REST caller would refetch a price the
            # stream had already delivered milliseconds ago.
            if tick.get('live') and age <= self.LIVE_TTL:
                return dict(tick)
            if age <= ttl:
                return dict(tick)
        return None

    def _lock_for(self, symbol: str) -> threading.Lock:
        with self._lock:
            if symbol not in self._fetch_locks:
                self._fetch_locks[symbol] = threading.Lock()
            return self._fetch_locks[symbol]

    def _fetch(self, symbol: str) -> dict:
        """
        Walk the sources for this asset class, best first, until one answers.

        Crypto:  Binance -> CoinGecko -> Yahoo
        Indian:  SmartAPI -> Yahoo
        Other:   Yahoo

        Each step is a real fallback, not a retry: the next source is only asked
        once the previous one has actually failed to produce a price.
        """
        if is_crypto_symbol(symbol):
            tick = self._fetch_crypto(symbol)
            if tick is None:
                # Binance cannot serve it - usually a delisted or migrated pair.
                # CoinGecko keeps those, including history across a rename.
                tick = self._fetch_coingecko(symbol)
            if tick is None:
                tick = self._fetch_stock(symbol)
        else:
            # An Indian instrument gets the live broker feed when it is available;
            # Yahoo's 15-minute delay is the fallback, not the default.
            tick = self._fetch_smartapi(symbol)
            if tick is None:
                tick = self._fetch_stock(symbol)

        if tick is None:
            # Rather than emit a zero - which downstream code reads as a real price
            # and which flashes "$0.00" at the user - reuse the last good tick and
            # mark it stale so the UI can show it greyed.
            stale = self.peek(symbol)
            if stale:
                stale['stale'] = True
                stale['ts'] = time.time()
                return stale
            return {
                'success': False, 'symbol': symbol, 'price': 0.0, 'change_pct': 0.0,
                'high_24h': 0.0, 'low_24h': 0.0, 'volume_24h': 0.0, 'open': 0.0,
                'currency': 'USD', 'source': 'unavailable', 'stale': True,
                'ts': time.time(),
                'error': f'No price data available for {symbol}',
            }
        return tick

    def _fetch_crypto(self, symbol: str) -> Optional[dict]:
        """
        One /ticker/24hr round-trip gives price *and* change together.

        The old socket path took the price from /ticker/price and the change from
        /ticker/24hr; between those two requests the market moves, so the pair it
        published never described a single moment.
        """
        if not self.crypto_provider:
            return None
        try:
            data = self.crypto_provider.get_ticker_24h(symbol)
        except Exception as e:
            logger.warning(f"[quotes] crypto ticker failed for {symbol}: {e}")
            return None

        price = float(data.get('price') or 0)
        if price <= 0:
            logger.warning(f"[quotes] crypto ticker returned no price for {symbol}")
            return None

        return {
            'success': True,
            'symbol': symbol,
            'price': price,
            'change_pct': float(data.get('price_change_pct') or 0),
            'high_24h': float(data.get('high_24h') or 0),
            'low_24h': float(data.get('low_24h') or 0),
            # Quote volume is the dollar figure the UI labels "24h volume".
            'volume_24h': float(data.get('quote_volume_24h') or 0),
            'open': float(data.get('open_price') or 0),
            'currency': 'USD',        # USDT/USDC/BUSD pairs are dollar-quoted
            'source': 'binance',
            'stale': False,
            'ts': time.time(),
        }

    def _fetch_coingecko(self, symbol: str) -> Optional[dict]:
        """Crypto that Binance has delisted or never listed."""
        if not self.coingecko_provider:
            return None
        try:
            q = self.coingecko_provider.get_current_quote(symbol)
        except Exception as e:
            logger.warning(f"[quotes] coingecko failed for {symbol}: {e}")
            return None
        if not q or not q.get('price'):
            return None

        return {
            'success': True,
            'symbol': symbol,
            'price': float(q['price']),
            'change_pct': float(q.get('change_pct') or 0),
            'high_24h': float(q.get('high') or 0),
            'low_24h': float(q.get('low') or 0),
            'volume_24h': float(q.get('volume') or 0),
            'open': float(q.get('previous_close') or 0),
            'currency': 'USD',
            'exchange': 'COINGECKO',
            'source': 'coingecko',
            'stale': False,
            'ts': time.time(),
        }

    def _fetch_smartapi(self, symbol: str) -> Optional[dict]:
        """
        Live NSE/BSE via Angel One.

        Returns None the moment the instrument is not an Indian one, or when no
        credentials are configured - so the non-Indian markets never pay for a
        lookup they cannot use.
        """
        provider = self.smartapi_provider
        if not provider or not getattr(provider, 'configured', False):
            return None
        try:
            if not provider.is_supported(symbol):
                return None
            q = provider.get_current_quote(symbol)
        except Exception as e:
            logger.warning(f"[quotes] smartapi failed for {symbol}: {e}")
            return None
        if not q or not q.get('price'):
            return None

        return {
            'success': True,
            'symbol': symbol,
            'price': float(q['price']),
            'change_pct': float(q.get('change_pct') or 0),
            'high_24h': float(q.get('high') or 0),
            'low_24h': float(q.get('low') or 0),
            'volume_24h': float(q.get('volume') or 0),
            'open': float(q.get('open') or q.get('previous_close') or 0),
            # NSE/BSE/MCX are rupee-quoted; the UI needs this to avoid stamping a
            # dollar sign on a rupee price.
            'currency': q.get('currency', 'INR'),
            'exchange': q.get('exchange', 'NSE'),
            'market_state': q.get('market_state', ''),
            'source': 'smartapi',
            'stale': False,
            'ts': time.time(),
        }

    def _fetch_stock(self, symbol: str) -> Optional[dict]:
        """Equities, FX and commodities all come from the Yahoo provider."""
        if not self.stock_provider:
            return None
        try:
            q = self.stock_provider.get_current_quote(symbol)
        except Exception as e:
            logger.warning(f"[quotes] stock quote failed for {symbol}: {e}")
            return None

        if not q or not q.get('price'):
            return None

        return {
            'success': True,
            'symbol': symbol,
            'price': float(q.get('price') or 0),
            'change_pct': float(q.get('change_pct') or 0),
            'high_24h': float(q.get('high') or 0),
            'low_24h': float(q.get('low') or 0),
            'volume_24h': float(q.get('volume') or 0),
            'open': float(q.get('previous_close') or 0),
            # An instrument is priced in its own quote currency - the UI needs this
            # to avoid stamping "$" on a rupee or yen price.
            'currency': q.get('currency') or 'USD',
            'exchange': q.get('exchange', ''),
            'market_state': q.get('market_state', ''),
            'source': q.get('source', 'yahoo'),
            'stale': bool(q.get('stale')),
            'ts': time.time(),
        }


# Process-wide singleton - every price path in the app reads through this one object.
quote_service = QuoteService()

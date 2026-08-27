"""
SymbolRegistry - decides what a ticker actually IS before anything tries to draw it.
====================================================================================

"This symbol doesn't exist" was TradingView's answer, not ours. It happened because
the frontend guessed an exchange prefix from a hardcoded list (`BINANCE:MATICUSDT`)
and handed it straight to the widget. When Binance retires a pair - MATIC migrated
to POL, LUNA to LUNC, FTM to S - the guess is stale, TradingView refuses it, and the
user gets a dead panel on a coin we can still price perfectly well.

This module answers three questions before a chart is created:

  1. Is this ticker an old name for something else?   -> RENAMES
  2. Does the venue we would chart from really list it? -> live exchangeInfo check
  3. If TradingView cannot serve it, where do WE get candles? -> data_sources

The answer always includes a way to draw a chart. If TradingView cannot be trusted
with the symbol, the caller falls back to GoatBot's own candles, which come from
Binance or - for pairs Binance has dropped - Yahoo's crypto feed. There is no branch
that returns "no data available" while a data source is still untried.
"""

import threading
import time
from typing import Dict, List, Optional

import requests
from loguru import logger

# ---------------------------------------------------------------------------
# Ticker renames
# ---------------------------------------------------------------------------
# Token migrations silently break every chart, watchlist row and open position
# that still uses the old ticker. Mapping them here means one entry fixes the
# price, the chart and the order ticket at once.
RENAMES: Dict[str, str] = {
    'MATIC': 'POL',        # Polygon migrated MATIC -> POL
    'MATICUSDT': 'POLUSDT',
    'LUNA': 'LUNC',        # Terra Classic
    'LUNAUSDT': 'LUNCUSDT',
    'FTM': 'S',            # Fantom -> Sonic
    'FTMUSDT': 'SUSDT',
    'CRO': 'CRO',
    'BAJAJ_AUTO': 'BAJAJ-AUTO',
}

# Human-readable reason, shown once when we quietly redirect a ticker.
RENAME_NOTES: Dict[str, str] = {
    'MATICUSDT': 'MATIC migrated to POL - showing POLUSDT.',
    'LUNAUSDT': 'LUNA is now LUNC - showing LUNCUSDT.',
    'FTMUSDT': 'FTM migrated to S (Sonic) - showing SUSDT.',
}

_STABLE_QUOTES = ('USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD')

# NSE/BSE indices. These are not equities - they have no -EQ form, no Yahoo
# ".NS" suffix, and TradingView files them under their own tickers.
INDIAN_INDICES = {
    'NIFTY':      {'tv': 'NSE:NIFTY',       'yahoo': '^NSEI',     'name': 'Nifty 50'},
    'NIFTY50':    {'tv': 'NSE:NIFTY',       'yahoo': '^NSEI',     'name': 'Nifty 50'},
    'BANKNIFTY':  {'tv': 'NSE:BANKNIFTY',   'yahoo': '^NSEBANK',  'name': 'Nifty Bank'},
    'FINNIFTY':   {'tv': 'NSE:CNXFINANCE',  'yahoo': '^CNXFIN',   'name': 'Nifty Financial Services'},
    'MIDCPNIFTY': {'tv': 'NSE:NIFTYMIDSELECT', 'yahoo': '^NSEMDCP50', 'name': 'Nifty Midcap Select'},
    'SENSEX':     {'tv': 'BSE:SENSEX',      'yahoo': '^BSESN',    'name': 'BSE Sensex'},
    'BANKEX':     {'tv': 'BSE:BANKEX',      'yahoo': '^BSEBANK',  'name': 'BSE Bankex'},
}

# NSE equities the app ships with. The instrument master covers the full universe
# once SmartAPI is configured; this list is what works without credentials.
NSE_EQUITIES = {
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN', 'BHARTIARTL',
    'ITC', 'HINDUNILVR', 'WIPRO', 'TATAMOTORS', 'ADANIENT', 'LT', 'BAJFINANCE',
    'MARUTI', 'AXISBANK', 'TITAN', 'SUNPHARMA', 'ASIANPAINT', 'NESTLEIND',
    'TATASTEEL', 'HCLTECH', 'POWERGRID', 'COALINDIA', 'ONGC', 'NTPC',
    'BAJAJFINSV', 'ADANIPORTS', 'ULTRACEMCO', 'JSWSTEEL', 'TECHM', 'INDUSINDBK',
    'HINDALCO', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'DIVISLAB', 'BPCL', 'GRASIM',
    'APOLLOHOSP', 'HEROMOTOCO', 'TATACONSUM', 'SBILIFE', 'HDFCLIFE', 'BRITANNIA',
    'KOTAKBANK', 'IDEA', 'JIOFIN', 'SUZLON', 'IRFC', 'YESBANK', 'NHPC',
    'TATAPOWER', 'IREDA', 'ZOMATO', 'PAYTM', 'DMART', 'VEDL', 'GAIL', 'PFC',
    'RECLTD', 'IOC', 'BEL', 'HAL', 'SAIL', 'NMDC', 'CANBK', 'PNB', 'BANKBARODA',
}


def is_indian(symbol: str) -> bool:
    """True for NSE/BSE equities and indices the app knows about."""
    s = symbol.strip().upper()
    return s in INDIAN_INDICES or s in NSE_EQUITIES

# TradingView carries every venue below; we only claim one when the venue's own
# API confirms it lists the pair.
_ALT_VENUES = [
    ('BYBIT',   'https://api.bybit.com/v5/market/instruments-info?category=spot'),
    ('OKX',     'https://www.okx.com/api/v5/public/instruments?instType=SPOT'),
]


def is_crypto_pair(symbol: str) -> bool:
    return symbol.upper().endswith(_STABLE_QUOTES)


def canonical(symbol: str) -> str:
    """Apply any rename so the rest of the app only ever sees the live ticker."""
    s = symbol.strip().upper()
    return RENAMES.get(s, s)


def to_yahoo_crypto(symbol: str) -> Optional[str]:
    """
    BTCUSDT -> BTC-USD.

    Yahoo carries crypto that Binance has delisted, which makes it the source of
    last resort for candles on a pair the exchange has dropped.
    """
    s = symbol.strip().upper()
    for q in _STABLE_QUOTES:
        if s.endswith(q):
            base = s[: -len(q)]
            return f"{base}-USD" if base else None
    return None


class SymbolRegistry:
    """Caches each venue's traded-pair list so resolution costs nothing per call."""

    LIST_TTL = 3600         # exchange listings change on the order of days
    NEG_TTL = 300           # don't re-probe a miss on every keystroke

    def __init__(self):
        self._lock = threading.Lock()
        self._listings: Dict[str, dict] = {}     # venue -> {'symbols': set, 'ts': float}
        self._resolved: Dict[str, dict] = {}     # symbol -> {'result': dict, 'ts': float}
        # Set by api_server when the Angel One provider is built. Its presence is
        # what decides whether an Indian symbol gets live data or delayed Yahoo.
        self.smartapi = None

    def configure(self, smartapi_provider=None) -> None:
        if smartapi_provider is not None:
            self.smartapi = smartapi_provider
            self._resolved.clear()   # cached decisions were made without it

    # ------------------------------------------------------------------
    # Venue listings
    # ------------------------------------------------------------------
    def _binance_symbols(self) -> Optional[set]:
        """
        Binance's traded-pair list, or None when we could not find out.

        None is not the same as "not listed". Treating an unreachable exchangeInfo
        as an empty listing would downgrade every crypto chart to the native
        renderer the moment the network hiccuped - so callers must check for None
        and stay optimistic.
        """
        cached = self._listings.get('BINANCE')
        if cached and (time.time() - cached['ts']) < self.LIST_TTL:
            return cached['symbols']
        try:
            resp = requests.get('https://api.binance.com/api/v3/exchangeInfo', timeout=8)
            resp.raise_for_status()
            symbols = {
                s['symbol'] for s in resp.json().get('symbols', [])
                if s.get('status') == 'TRADING'
            }
            if symbols:
                with self._lock:
                    self._listings['BINANCE'] = {'symbols': symbols, 'ts': time.time()}
                logger.info(f"[symbols] Binance listing refreshed: {len(symbols)} pairs")
                return symbols
        except Exception as e:
            logger.warning(f"[symbols] Binance exchangeInfo failed: {e}")
        # A failed refresh must not invalidate what we already knew.
        return cached['symbols'] if cached else None

    def _alt_venue_symbols(self, venue: str, url: str) -> set:
        cached = self._listings.get(venue)
        if cached and (time.time() - cached['ts']) < self.LIST_TTL:
            return cached['symbols']
        try:
            resp = requests.get(url, timeout=8)
            resp.raise_for_status()
            payload = resp.json()
            rows = payload.get('result', {}).get('list') or payload.get('data') or []
            symbols = set()
            for row in rows:
                # Bybit: {'symbol': 'BTCUSDT'}   OKX: {'instId': 'BTC-USDT'}
                sym = row.get('symbol') or row.get('instId', '').replace('-', '')
                if sym:
                    symbols.add(sym.upper())
            if symbols:
                with self._lock:
                    self._listings[venue] = {'symbols': symbols, 'ts': time.time()}
                return symbols
        except Exception as e:
            logger.warning(f"[symbols] {venue} listing failed: {e}")
        return cached['symbols'] if cached else set()

    # ------------------------------------------------------------------
    # TradingView catalogue
    # ------------------------------------------------------------------
    def tv_available(self, tv_symbol: str) -> Optional[bool]:
        """
        Ask TradingView whether it actually carries `EXCHANGE:TICKER`.

        Returns True/False, or None when the lookup itself failed. None matters:
        an unreachable search endpoint is not evidence that a symbol is missing,
        so the caller keeps TradingView rather than downgrading a chart that would
        have worked fine.
        """
        if not tv_symbol or ':' not in tv_symbol:
            return None

        cached = self._listings.get(f"TV::{tv_symbol}")
        if cached and (time.time() - cached['ts']) < self.LIST_TTL:
            return cached['ok']

        exchange, ticker = tv_symbol.split(':', 1)
        try:
            resp = requests.get(
                'https://symbol-search.tradingview.com/symbol_search/',
                params={'text': ticker, 'exchange': exchange, 'lang': 'en', 'type': ''},
                headers={
                    'Referer': 'https://www.tradingview.com/',
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
                },
                timeout=6,
            )
            if resp.status_code != 200:
                return None
            rows = resp.json()
            if not isinstance(rows, list):
                return None
            # The search is fuzzy - RELIANCE matches RELIANCEPP too - so only an
            # exact ticker on the exact exchange counts as "TradingView has it".
            ok = any(
                str(r.get('symbol', '')).replace('<em>', '').replace('</em>', '').upper() == ticker.upper()
                and exchange.upper() in str(r.get('exchange', '')).upper()
                for r in rows
            )
            with self._lock:
                self._listings[f"TV::{tv_symbol}"] = {'ok': ok, 'ts': time.time()}
            return ok
        except Exception as e:
            logger.warning(f"[symbols] TradingView lookup failed for {tv_symbol}: {e}")
            return None

    # ------------------------------------------------------------------
    # Resolution
    # ------------------------------------------------------------------
    def resolve(self, symbol: str, market_hint: Optional[str] = None,
                tv_candidate: Optional[str] = None) -> dict:
        """
        Work out the canonical ticker, the TradingView symbol (if any venue really
        lists it), and the ordered list of places we can get candles ourselves.

        `chart` is never empty: it is 'tradingview' when a venue confirms the pair,
        otherwise 'native', meaning the frontend draws from `data_sources`.
        """
        requested = symbol.strip().upper()
        if not requested:
            return self._unknown(requested)

        key = f"{requested}|{tv_candidate or ''}"
        cached = self._resolved.get(key)
        if cached and (time.time() - cached['ts']) < self.NEG_TTL:
            return dict(cached['result'])

        result = self._resolve_uncached(requested, market_hint, tv_candidate)
        with self._lock:
            self._resolved[key] = {'result': result, 'ts': time.time()}
        return dict(result)

    def _resolve_uncached(self, requested: str, market_hint: Optional[str],
                          tv_candidate: Optional[str]) -> dict:
        target = canonical(requested)
        renamed = target if target != requested else None
        note = RENAME_NOTES.get(requested) if renamed else None

        if is_crypto_pair(target) or (market_hint == 'crypto'):
            return self._resolve_crypto(requested, target, renamed, note)
        if is_indian(target) or market_hint == 'india':
            return self._resolve_indian(requested, target, renamed, note)
        return self._resolve_non_crypto(requested, target, renamed, note, tv_candidate)

    def _resolve_crypto(self, requested, target, renamed, note) -> dict:
        data_sources: List[str] = []
        tv_symbol = None
        exchange = None

        listing = self._binance_symbols()

        if listing is None:
            # We could not reach Binance to check. Assume the pair is fine and let
            # TradingView draw it - the alternative is downgrading every crypto
            # chart on a transient network error.
            tv_symbol, exchange = f"BINANCE:{target}", 'BINANCE'
            data_sources.append('binance')
        elif target in listing:
            tv_symbol, exchange = f"BINANCE:{target}", 'BINANCE'
            data_sources.append('binance')
        else:
            # Binance has dropped it (or never had it). Ask the venues TradingView
            # also carries before giving up on a real exchange chart.
            for venue, url in _ALT_VENUES:
                if target in self._alt_venue_symbols(venue, url):
                    tv_symbol, exchange = f"{venue}:{target}", venue
                    break

        # CoinGecko tracks ~17k coins including migrated and retired ones, and
        # keeps history attached across a rename - which is precisely the case
        # Binance cannot serve. It sits ahead of Yahoo for crypto coverage.
        data_sources.append('coingecko')

        # Yahoo remains the last resort for candles.
        yahoo = to_yahoo_crypto(target)
        if yahoo:
            data_sources.append('yahoo')

        return {
            'requested': requested,
            'symbol': target,
            'renamed_from': requested if renamed else None,
            'note': note,
            'market': 'crypto',
            'region': 'GLOBAL',
            'currency': 'USD',
            'exchange': exchange,
            'tv_symbol': tv_symbol,
            'chart': 'tradingview' if tv_symbol else 'native',
            'data_sources': data_sources,
            'yahoo_symbol': yahoo,
        }

    def _resolve_indian(self, requested, target, renamed, note) -> dict:
        """
        NSE/BSE instruments.

        Indices are the case the old code got wrong twice over: ChartManager would
        chart NSE:NIFTY while the backend had no mapping at all, so the chart drew
        and the price panel stayed empty. Both sides are pinned here.
        """
        idx = INDIAN_INDICES.get(target)
        if idx:
            tv_symbol, yahoo_symbol, market = idx['tv'], idx['yahoo'], 'index'
        else:
            tv_symbol, yahoo_symbol, market = f"NSE:{target}", f"{target}.NS", 'stocks'

        available = self.tv_available(tv_symbol)
        use_tv = available is not False   # unknown != missing

        # SmartAPI first when it is live: Yahoo's NSE data is ~15 minutes delayed,
        # which is fine as a backstop and wrong as a default.
        sources = []
        if self.smartapi and getattr(self.smartapi, 'configured', False):
            sources.append('smartapi')
        sources.append('yahoo')

        return {
            'requested': requested,
            'symbol': target,
            'renamed_from': requested if renamed else None,
            'note': note,
            'market': market,
            'region': 'IN',
            'currency': 'INR',
            'exchange': tv_symbol.split(':')[0] if use_tv else None,
            'tv_symbol': tv_symbol if use_tv else None,
            'chart': 'tradingview' if use_tv else 'native',
            'data_sources': sources,
            'yahoo_symbol': yahoo_symbol,
        }

    def _resolve_non_crypto(self, requested, target, renamed, note, tv_candidate) -> dict:
        """
        Equities, FX and commodities. The frontend owns the exchange-prefix table
        (ChartManager) and passes its guess in as `tv_candidate`; we check that
        guess against TradingView's own catalogue rather than trusting it.

        Yahoo answers for all three asset classes, so a native chart is always
        available even when the prefix guess turns out to be wrong.
        """
        from shared.providers.stock_provider import YahooFinanceProvider as Y

        market = 'forex' if (len(target) == 6 and target.isalpha()) else 'stocks'
        if target.startswith('X') and len(target) == 6:
            market = 'commodities'

        tv_symbol = tv_candidate
        available = self.tv_available(tv_candidate) if tv_candidate else None
        # None means the lookup failed, not that the symbol is missing - keep
        # TradingView rather than downgrading a chart that would have rendered.
        use_tv = tv_symbol is not None and available is not False

        return {
            'requested': requested,
            'symbol': target,
            'renamed_from': requested if renamed else None,
            'note': note,
            'market': market,
            'exchange': tv_symbol.split(':')[0] if (use_tv and ':' in (tv_symbol or '')) else None,
            'tv_symbol': tv_symbol if use_tv else None,
            'chart': 'tradingview' if use_tv else 'native',
            'data_sources': ['yahoo'],
            'yahoo_symbol': Y.to_yahoo_symbol(target),
        }

    def _unknown(self, requested: str) -> dict:
        return {
            'requested': requested, 'symbol': requested, 'renamed_from': None,
            'note': None, 'market': 'unknown', 'exchange': None, 'tv_symbol': None,
            'chart': 'native', 'data_sources': ['yahoo'], 'yahoo_symbol': None,
        }


symbol_registry = SymbolRegistry()

"""
SmartAPIProvider - live Indian market data from Angel One.
==========================================================

This is the first real NSE/BSE feed in the app. Everything Indian previously came
from Yahoo: roughly 15 minutes delayed, eight hardcoded tickers, no indices, no
F&O. SmartAPI gives live quotes, the full instrument universe, and free historical
candles across NSE, BSE, NFO, BFO, MCX and CDS.

Authentication
--------------
SmartAPI is a *broker* API, so it authenticates a person, not a server: API key +
client code + PIN + a TOTP from the authenticator secret. That session lasts a
day and must be refreshed, which is why login is lazy here - the app boots and
serves every other market normally whether or not credentials are present.

Credentials come from the environment and are never logged:

    SMARTAPI_KEY, SMARTAPI_CLIENT_CODE, SMARTAPI_PIN, SMARTAPI_TOTP_SECRET

With any of them missing the provider reports `configured = False` and the
cascade falls through to Yahoo exactly as before. Nothing breaks; the data is
just delayed again.

Note Angel One binds an app to a static IP. A request from an unregistered
address fails authentication no matter how correct the credentials are.
"""

import os
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import requests
from loguru import logger

BASE_URL = "https://apiconnect.angelone.in"

# Angel One publishes the full tradeable universe as one JSON file (~100k rows).
# It is the only way to map a human ticker to the numeric token every other
# endpoint requires.
SCRIP_MASTER_URL = ("https://margincalculator.angelbroking.com/OpenAPI_File/files/"
                    "OpenAPIScripMaster.json")

# Our interval names -> SmartAPI's.
INTERVALS = {
    '1m': 'ONE_MINUTE', '3m': 'THREE_MINUTE', '5m': 'FIVE_MINUTE',
    '10m': 'TEN_MINUTE', '15m': 'FIFTEEN_MINUTE', '30m': 'THIRTY_MINUTE',
    '1h': 'ONE_HOUR', '1d': 'ONE_DAY',
}

# How far back each interval may be requested in one call, per Angel One's limits.
MAX_DAYS = {
    'ONE_MINUTE': 30, 'THREE_MINUTE': 60, 'FIVE_MINUTE': 100, 'TEN_MINUTE': 100,
    'FIFTEEN_MINUTE': 200, 'THIRTY_MINUTE': 200, 'ONE_HOUR': 400, 'ONE_DAY': 2000,
}

# Index tickers as they appear in the scrip master, keyed by what our UI calls them.
INDEX_ALIASES = {
    'NIFTY': 'Nifty 50',
    'NIFTY50': 'Nifty 50',
    'BANKNIFTY': 'Nifty Bank',
    'FINNIFTY': 'Nifty Fin Service',
    'MIDCPNIFTY': 'NIFTY MID SELECT',
    'SENSEX': 'SENSEX',
}


class SmartAPIProvider:
    """Angel One SmartAPI: session handling, symbol lookup, quotes and candles."""

    SESSION_TTL = 20 * 3600     # tokens are day-scoped; refresh well before expiry
    SCRIP_TTL = 20 * 3600       # the instrument master is republished daily
    QUOTE_TTL = 1.0

    def __init__(self):
        self.api_key = os.getenv('SMARTAPI_KEY', '').strip()
        self.client_code = os.getenv('SMARTAPI_CLIENT_CODE', '').strip()
        self.pin = os.getenv('SMARTAPI_PIN', '').strip()
        self.totp_secret = os.getenv('SMARTAPI_TOTP_SECRET', '').strip()

        self.session = requests.Session()
        self._lock = threading.Lock()
        self._auth_lock = threading.Lock()

        self._jwt: Optional[str] = None
        self._feed_token: Optional[str] = None
        self._session_ts = 0.0
        self._auth_failed_until = 0.0    # don't hammer login after a hard rejection

        self._scrip: Dict[str, dict] = {}      # "EXCHANGE:SYMBOL" -> row
        self._scrip_ts = 0.0
        self._quotes: Dict[str, dict] = {}

    @property
    def configured(self) -> bool:
        """True only when every credential needed for a session is present."""
        return all([self.api_key, self.client_code, self.pin, self.totp_secret])

    # ------------------------------------------------------------------
    # Session
    # ------------------------------------------------------------------
    def _headers(self, authed: bool = True) -> dict:
        h = {
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'X-UserType': 'USER',
            'X-SourceID': 'WEB',
            'X-ClientLocalIP': '127.0.0.1',
            'X-ClientPublicIP': '127.0.0.1',
            'X-MACAddress': '00:00:00:00:00:00',
            'X-PrivateKey': self.api_key,
        }
        if authed and self._jwt:
            h['Authorization'] = f'Bearer {self._jwt}'
        return h

    def _ensure_session(self) -> bool:
        """
        Log in if we have no valid session. Returns False rather than raising, so
        a credential problem degrades the app to Yahoo instead of breaking it.
        """
        if not self.configured:
            return False
        if self._jwt and (time.time() - self._session_ts) < self.SESSION_TTL:
            return True
        # After a rejection, back off. Repeatedly failing a broker login is a good
        # way to get an account flagged.
        if time.time() < self._auth_failed_until:
            return False

        with self._auth_lock:
            if self._jwt and (time.time() - self._session_ts) < self.SESSION_TTL:
                return True
            return self._login()

    def _login(self) -> bool:
        try:
            import pyotp
        except ImportError:
            logger.error("[smartapi] pyotp is required for TOTP login (pip install pyotp)")
            self._auth_failed_until = time.time() + 3600
            return False

        try:
            totp = pyotp.TOTP(self.totp_secret).now()
        except Exception as e:
            logger.error(f"[smartapi] could not generate TOTP - check SMARTAPI_TOTP_SECRET: {e}")
            self._auth_failed_until = time.time() + 3600
            return False

        try:
            resp = self.session.post(
                f"{BASE_URL}/rest/auth/angelbroking/user/v1/loginByPassword",
                json={'clientcode': self.client_code, 'password': self.pin, 'totp': totp},
                headers=self._headers(authed=False),
                timeout=15,
            )
            payload = resp.json()
        except Exception as e:
            logger.warning(f"[smartapi] login request failed: {e}")
            self._auth_failed_until = time.time() + 120
            return False

        if not payload.get('status'):
            # Never log the payload wholesale - it can echo identifiers back.
            logger.error(f"[smartapi] login rejected: {payload.get('message', 'unknown reason')}")
            self._auth_failed_until = time.time() + 900
            return False

        data = payload.get('data') or {}
        self._jwt = data.get('jwtToken')
        self._feed_token = data.get('feedToken')
        self._session_ts = time.time()
        self._auth_failed_until = 0.0
        logger.info("[smartapi] session established")
        return bool(self._jwt)

    @property
    def feed_token(self) -> Optional[str]:
        """Token for the tick-by-tick WebSocket feed."""
        return self._feed_token if self._ensure_session() else None

    # ------------------------------------------------------------------
    # Instrument master
    # ------------------------------------------------------------------
    def _load_scrip_master(self) -> Dict[str, dict]:
        """
        Download and index the instrument universe.

        Every quote and candle call needs a numeric `symboltoken`; this file is the
        only mapping from "RELIANCE" to that token. It is large, so it is fetched
        once a day and indexed by "EXCHANGE:SYMBOL".
        """
        with self._lock:
            if self._scrip and (time.time() - self._scrip_ts) < self.SCRIP_TTL:
                return self._scrip

        try:
            resp = requests.get(SCRIP_MASTER_URL, timeout=90)
            resp.raise_for_status()
            rows = resp.json()
        except Exception as e:
            logger.warning(f"[smartapi] scrip master download failed: {e}")
            with self._lock:
                return self._scrip

        index: Dict[str, dict] = {}
        for row in rows:
            exch = str(row.get('exch_seg', '')).upper()
            sym = str(row.get('symbol', '')).upper()
            name = str(row.get('name', '')).upper()
            if not exch or not sym:
                continue
            index[f"{exch}:{sym}"] = row
            # Cash equities carry an "-EQ" suffix in `symbol`; traders type the bare
            # name, so index that too rather than making callers know the suffix.
            if sym.endswith('-EQ') and name:
                index.setdefault(f"{exch}:{name}", row)
            elif exch in ('NSE', 'BSE') and name:
                index.setdefault(f"{exch}:{name}", row)

        with self._lock:
            self._scrip = index
            self._scrip_ts = time.time()
        logger.info(f"[smartapi] scrip master indexed: {len(index)} instruments")
        return index

    def lookup(self, symbol: str, exchange: str = 'NSE') -> Optional[dict]:
        """Find an instrument's scrip row, trying the forms a trader might type."""
        index = self._load_scrip_master()
        if not index:
            return None

        s = symbol.strip().upper()
        exch = exchange.strip().upper()

        # An index is not an equity and has no -EQ form.
        if s in INDEX_ALIASES:
            alias = INDEX_ALIASES[s].upper()
            for ex in (exch, 'NSE', 'BSE'):
                row = index.get(f"{ex}:{alias}")
                if row:
                    return row

        for candidate in (f"{exch}:{s}-EQ", f"{exch}:{s}", f"NSE:{s}-EQ", f"NSE:{s}",
                          f"BSE:{s}"):
            row = index.get(candidate)
            if row:
                return row
        return None

    def is_supported(self, symbol: str, exchange: str = 'NSE') -> bool:
        return self.configured and self.lookup(symbol, exchange) is not None

    # ------------------------------------------------------------------
    # Quotes
    # ------------------------------------------------------------------
    def get_current_quote(self, symbol: str, exchange: str = 'NSE') -> Dict:
        """
        Live LTP with the day's open/high/low/close, shaped like every other
        provider so QuoteService does not care who answered.
        """
        symbol = symbol.strip().upper()
        cached = self._quotes.get(symbol)
        if cached and (time.time() - cached['_ts']) < self.QUOTE_TTL:
            return dict(cached)

        if not self._ensure_session():
            return {'symbol': symbol, 'price': 0, 'source': 'smartapi',
                    'error': 'not configured' if not self.configured else 'no session'}

        row = self.lookup(symbol, exchange)
        if not row:
            return {'symbol': symbol, 'price': 0, 'source': 'smartapi',
                    'error': f'{symbol} not in instrument master'}

        try:
            resp = self.session.post(
                f"{BASE_URL}/rest/secure/angelbroking/order/v1/getLtpData",
                json={'exchange': row['exch_seg'],
                      'tradingsymbol': row['symbol'],
                      'symboltoken': row['token']},
                headers=self._headers(),
                timeout=10,
            )
            payload = resp.json()
        except Exception as e:
            logger.warning(f"[smartapi] quote request failed for {symbol}: {e}")
            return {'symbol': symbol, 'price': 0, 'source': 'smartapi', 'error': str(e)}

        if not payload.get('status'):
            msg = payload.get('message', 'unknown')
            # A dead session shows up as an ordinary error; drop it so the next
            # call re-logs in rather than failing forever.
            if 'token' in str(msg).lower() or 'session' in str(msg).lower():
                self._jwt = None
            logger.warning(f"[smartapi] quote rejected for {symbol}: {msg}")
            return {'symbol': symbol, 'price': 0, 'source': 'smartapi', 'error': str(msg)}

        d = payload.get('data') or {}
        price = float(d.get('ltp') or 0)
        prev_close = float(d.get('close') or 0)

        quote = {
            'symbol': symbol,
            'price': price,
            'previous_close': prev_close,
            'change': price - prev_close if prev_close else 0.0,
            'change_pct': ((price / prev_close) - 1) * 100 if prev_close else 0.0,
            'high': float(d.get('high') or 0),
            'low': float(d.get('low') or 0),
            'open': float(d.get('open') or 0),
            'volume': 0.0,          # LTP endpoint carries no volume
            'currency': 'INR',      # NSE/BSE/MCX are all rupee-quoted
            'exchange': row['exch_seg'],
            'market_state': 'REGULAR',
            'source': 'smartapi',
            '_ts': time.time(),
        }
        self._quotes[symbol] = quote
        return dict(quote)

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------
    def get_historical_data(self, symbol: str, interval: str = '5m',
                            limit: int = 300, exchange: str = 'NSE') -> pd.DataFrame:
        """
        OHLCV candles. Empty frame on any failure, so the caller falls through to
        the next source rather than surfacing an error.
        """
        if not self._ensure_session():
            return pd.DataFrame()

        row = self.lookup(symbol, exchange)
        if not row:
            return pd.DataFrame()

        sm_interval = INTERVALS.get(interval, 'FIVE_MINUTE')

        # Ask for enough calendar days to cover `limit` candles, then clamp to what
        # Angel One allows for that interval. Requesting beyond the cap is an error,
        # not a truncation.
        per_day = {'ONE_MINUTE': 375, 'THREE_MINUTE': 125, 'FIVE_MINUTE': 75,
                   'TEN_MINUTE': 37, 'FIFTEEN_MINUTE': 25, 'THIRTY_MINUTE': 12,
                   'ONE_HOUR': 6, 'ONE_DAY': 1}.get(sm_interval, 75)
        # Weekends and holidays mean calendar days > trading days; the 1.6x pads
        # for that so a 300-candle request does not come back short.
        days = min(int((limit / per_day) * 1.6) + 2, MAX_DAYS.get(sm_interval, 30))

        to_dt = datetime.now()
        from_dt = to_dt - timedelta(days=days)

        try:
            resp = self.session.post(
                f"{BASE_URL}/rest/secure/angelbroking/historical/v1/getCandleData",
                json={'exchange': row['exch_seg'],
                      'symboltoken': row['token'],
                      'interval': sm_interval,
                      'fromdate': from_dt.strftime('%Y-%m-%d %H:%M'),
                      'todate': to_dt.strftime('%Y-%m-%d %H:%M')},
                headers=self._headers(),
                timeout=20,
            )
            payload = resp.json()
        except Exception as e:
            logger.warning(f"[smartapi] candle request failed for {symbol}: {e}")
            return pd.DataFrame()

        if not payload.get('status') or not payload.get('data'):
            logger.warning(f"[smartapi] no candles for {symbol}: {payload.get('message', 'empty')}")
            return pd.DataFrame()

        df = pd.DataFrame(payload['data'],
                          columns=['time', 'open', 'high', 'low', 'close', 'volume'])
        # Timestamps come back with an IST offset; normalise to naive local time so
        # the frame matches what the other providers produce.
        df['time'] = pd.to_datetime(df['time'], format='ISO8601', utc=True).dt.tz_convert(
            'Asia/Kolkata').dt.tz_localize(None)
        df.set_index('time', inplace=True)
        for col in ('open', 'high', 'low', 'close', 'volume'):
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df.dropna(inplace=True)
        if len(df) > limit:
            df = df.tail(limit)

        logger.info(f"[smartapi] {len(df)} candles for {symbol} ({sm_interval})")
        return df

    # ------------------------------------------------------------------
    # Discovery
    # ------------------------------------------------------------------
    def search(self, query: str, exchange: str = 'NSE', limit: int = 20) -> List[dict]:
        """Ticker search over the instrument master, for the UI's market search."""
        index = self._load_scrip_master()
        q = query.strip().upper()
        if not q or not index:
            return []

        exch = exchange.strip().upper()
        hits, seen = [], set()
        for key, row in index.items():
            if not key.startswith(f"{exch}:"):
                continue
            name = str(row.get('name', '')).upper()
            if not name.startswith(q):
                continue
            if name in seen:
                continue
            seen.add(name)
            hits.append({'symbol': name, 'name': row.get('name'),
                         'exchange': row.get('exch_seg'), 'token': row.get('token')})
            if len(hits) >= limit:
                break
        return hits


smartapi_provider = SmartAPIProvider()

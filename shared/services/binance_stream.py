"""
BinanceStream - push, not poll.
===============================

The panels used to trail the chart by seconds, and the reason was architectural
rather than a bug: every price reached the screen by polling. The browser asked
every 5s; the server's own `price_stream` thread walked its symbol list doing one
REST round-trip each, so a single symbol was refreshed every 2-4s. TradingView,
meanwhile, streams. Two different clocks on one screen always looks like a lag,
because it is one.

This opens a single WebSocket to Binance's combined stream and subscribes to two
feeds per symbol:

  <sym>@aggTrade  every trade, in real time -> the price moves the instant it moves
  <sym>@ticker    once a second           -> 24h change, high, low, volume, open

Between ticker frames the change percentage is recomputed from the day's open, so
the number next to the price stays correct at trade cadence rather than jumping
once a second.

Ordering comes from Binance's own event time, not from arrival order, so a frame
delayed in the network can never overwrite a newer one.

Emission is throttled per symbol (see MIN_EMIT_INTERVAL). BTCUSDT can trade
hundreds of times a second and no display needs that; ~10 updates a second is
already smoother than the eye resolves, and it keeps the socket from flooding.
"""

import json
import threading
import time
from typing import Callable, Dict, Iterable, List, Optional, Set

from loguru import logger

WS_URL = "wss://stream.binance.com:9443/stream"

# Fastest we will push any single symbol to the UI. 100ms is visually continuous.
MIN_EMIT_INTERVAL = 0.1

_STABLE_QUOTES = ('USDT', 'BUSD', 'USDC', 'FDUSD', 'TUSD')


def is_streamable(symbol: str) -> bool:
    """Binance streams spot pairs; everything else has to keep polling."""
    return symbol.upper().endswith(_STABLE_QUOTES)


class BinanceStream:
    """One WebSocket, many symbols, dynamically re-subscribed as the UI changes."""

    RECONNECT_BASE = 2
    RECONNECT_MAX = 60

    def __init__(self):
        self._symbols: Set[str] = set()
        self._lock = threading.Lock()
        self._ws = None
        self._thread: Optional[threading.Thread] = None
        self._running = False
        self._connected = False
        self._req_id = 0
        self._backoff = self.RECONNECT_BASE

        self._on_tick: Optional[Callable[[dict], None]] = None
        # Last full picture per symbol, so an aggTrade can carry the 24h stats
        # that only the 1s ticker frame provides.
        self._stats: Dict[str, dict] = {}
        self._last_emit: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def configure(self, on_tick: Callable[[dict], None]) -> None:
        self._on_tick = on_tick

    @property
    def connected(self) -> bool:
        return self._connected

    def start(self, symbols: Iterable[str] = ()) -> None:
        if self._running:
            self.set_symbols(symbols)
            return
        with self._lock:
            self._symbols = {s.upper() for s in symbols if is_streamable(s)}
        self._running = True
        self._thread = threading.Thread(target=self._run, name='binance-stream', daemon=True)
        self._thread.start()
        logger.info(f"[stream] starting with {len(self._symbols)} symbols")

    def stop(self) -> None:
        self._running = False
        try:
            if self._ws:
                self._ws.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Subscription management
    # ------------------------------------------------------------------
    @staticmethod
    def _streams_for(symbol: str) -> List[str]:
        s = symbol.lower()
        return [f"{s}@aggTrade", f"{s}@ticker"]

    def set_symbols(self, symbols: Iterable[str]) -> None:
        """
        Track exactly the symbols the UI is showing.

        Only the difference is sent, so switching market tab does not tear down
        and rebuild the whole subscription.
        """
        wanted = {s.upper() for s in symbols if is_streamable(s)}
        with self._lock:
            added = wanted - self._symbols
            removed = self._symbols - wanted
            self._symbols = wanted

        if not (added or removed):
            return
        if not self._connected:
            return  # the next connect subscribes to the current set anyway

        if added:
            self._send('SUBSCRIBE', [st for s in added for st in self._streams_for(s)])
            logger.info(f"[stream] +{len(added)} symbols")
        if removed:
            self._send('UNSUBSCRIBE', [st for s in removed for st in self._streams_for(s)])
            for s in removed:
                self._stats.pop(s, None)
                self._last_emit.pop(s, None)

    def _send(self, method: str, params: List[str]) -> None:
        if not self._ws or not params:
            return
        self._req_id += 1
        try:
            self._ws.send(json.dumps({'method': method, 'params': params, 'id': self._req_id}))
        except Exception as e:
            logger.warning(f"[stream] {method} failed: {e}")

    # ------------------------------------------------------------------
    # Connection
    # ------------------------------------------------------------------
    def _run(self) -> None:
        try:
            import websocket
        except ImportError:
            logger.error("[stream] websocket-client not installed - live prices will fall back to polling")
            self._running = False
            return

        while self._running:
            try:
                with self._lock:
                    streams = [st for s in self._symbols for st in self._streams_for(s)]
                # Binance rejects a connection with no streams, so idle until the
                # UI asks for something.
                if not streams:
                    time.sleep(2)
                    continue

                url = f"{WS_URL}?streams={'/'.join(streams)}"
                self._ws = websocket.WebSocketApp(
                    url,
                    on_message=self._on_message,
                    on_open=self._on_open,
                    on_close=self._on_close,
                    on_error=self._on_error,
                )
                # ping_interval keeps Binance from dropping us at the 24h mark and
                # surfaces a dead link quickly.
                self._ws.run_forever(ping_interval=180, ping_timeout=10)
            except Exception as e:
                logger.warning(f"[stream] connection error: {e}")

            self._connected = False
            if not self._running:
                break
            # Back off so a Binance outage does not become a reconnect storm.
            time.sleep(self._backoff)
            self._backoff = min(self._backoff * 2, self.RECONNECT_MAX)

    def _on_open(self, ws) -> None:
        self._connected = True
        self._backoff = self.RECONNECT_BASE
        logger.info(f"[stream] connected ({len(self._symbols)} symbols)")

    def _on_close(self, ws, status, msg) -> None:
        self._connected = False
        logger.info("[stream] disconnected")

    def _on_error(self, ws, err) -> None:
        logger.warning(f"[stream] error: {err}")

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------
    def _on_message(self, ws, message: str) -> None:
        try:
            payload = json.loads(message)
        except Exception:
            return

        data = payload.get('data')
        if not isinstance(data, dict):
            return  # subscription acks have no 'data'

        event = data.get('e')
        if event == '24hrTicker':
            tick = self._from_ticker(data)
        elif event == 'aggTrade':
            tick = self._from_trade(data)
        else:
            return

        if tick:
            self._emit(tick)

    def _from_ticker(self, d: dict) -> Optional[dict]:
        """The 1s frame: price plus every 24h statistic, from one event."""
        try:
            symbol = d['s'].upper()
            price = float(d['c'])
            if price <= 0:
                return None
            stats = {
                'change_pct': float(d.get('P') or 0),
                'high_24h': float(d.get('h') or 0),
                'low_24h': float(d.get('l') or 0),
                'volume_24h': float(d.get('q') or 0),   # quote volume, the $ figure
                'open': float(d.get('o') or 0),
            }
            self._stats[symbol] = stats
            return {
                'success': True, 'symbol': symbol, 'price': price,
                **stats, 'currency': 'USD', 'exchange': 'BINANCE',
                'source': 'binance_ws', 'stale': False, 'live': True,
                'ts': float(d.get('E', time.time() * 1000)) / 1000.0,
            }
        except (KeyError, TypeError, ValueError):
            return None

    def _from_trade(self, d: dict) -> Optional[dict]:
        """
        A real trade. Carries only a price, so the 24h stats come from the last
        ticker frame and the change is recomputed against the day's open - that
        keeps the percentage moving with the price instead of stepping once a
        second.
        """
        try:
            symbol = d['s'].upper()
            price = float(d['p'])
            if price <= 0:
                return None
        except (KeyError, TypeError, ValueError):
            return None

        stats = self._stats.get(symbol)
        if not stats:
            # No ticker frame yet: publish the price alone rather than pairing it
            # with statistics we do not have.
            return {
                'success': True, 'symbol': symbol, 'price': price,
                'change_pct': 0.0, 'high_24h': 0.0, 'low_24h': 0.0,
                'volume_24h': 0.0, 'open': 0.0, 'currency': 'USD',
                'exchange': 'BINANCE', 'source': 'binance_ws', 'stale': False,
                'live': True, 'ts': float(d.get('E', time.time() * 1000)) / 1000.0,
            }

        open_px = stats.get('open') or 0
        change_pct = ((price / open_px) - 1) * 100 if open_px else stats['change_pct']
        return {
            'success': True, 'symbol': symbol, 'price': price,
            'change_pct': change_pct,
            'high_24h': max(stats['high_24h'], price),
            'low_24h': min(stats['low_24h'], price) if stats['low_24h'] else price,
            'volume_24h': stats['volume_24h'], 'open': open_px,
            'currency': 'USD', 'exchange': 'BINANCE', 'source': 'binance_ws',
            'stale': False, 'live': True,
            'ts': float(d.get('E', time.time() * 1000)) / 1000.0,
        }

    def _emit(self, tick: dict) -> None:
        symbol = tick['symbol']
        now = time.time()
        last = self._last_emit.get(symbol, 0)
        if (now - last) < MIN_EMIT_INTERVAL:
            return
        self._last_emit[symbol] = now
        try:
            if self._on_tick:
                self._on_tick(tick)
        except Exception as e:
            logger.warning(f"[stream] tick handler failed for {symbol}: {e}")


binance_stream = BinanceStream()

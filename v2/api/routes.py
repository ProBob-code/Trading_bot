"""
V2 API Blueprint — Institutional Trading Engine
=================================================

All V2-specific routes, singletons, and bot execution logic.
Registered as a Flask Blueprint into the main api_server.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from datetime import datetime, date, timezone
import os
import threading
import time
import json
import math

from loguru import logger

# V2 Engine imports
from v2.engine.execution.execution_engine import ExecutionEngine
from v2.engine.execution.paper_trader_v2 import PaperTraderV2
from v2.engine.risk.margin_engine import MarginEngine
from v2.engine.analytics.strategy_analytics import StrategyAnalytics
from v2.engine.analytics.monte_carlo import MonteCarloSimulator
from v2.engine.portfolio.allocator import CapitalAllocator
from v2.engine.portfolio.ranking_engine import StrategyRanker
from v2.engine.intelligence.regime_detector import RegimeDetector
from v2.engine.intelligence.volatility_filter import VolatilityFilter
from v2.engine.bot_manager_v2 import BotManagerV2, bot_manager_v2
from shared.database.db_manager import db_manager
from shared.services.quote_service import quote_service
from shared.services.symbol_registry import symbol_registry
from v2.engine.core.risk_engine import RiskEngineV2
from v2.engine.core.portfolio_engine import PortfolioEngineV2
from v2.engine.core.pipeline import TradingPipelineV2
from shared.logic.strategies.v3_quant_strategies import REGISTRY, atr as compute_atr, compute_smart_entry, compute_atr_position_size
from shared.logic.strategies.public_catalog import (
    public_catalog, public_meta, to_internal as strat_to_internal,
    to_public as strat_to_public, mask_bot_id, unmask_bot_id, interval_for)

# TTL Cache for sessions
_SESSIONS_CACHE = {'data': None, 'timestamp': 0}
SESSIONS_TTL = 5 # seconds

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
v2_bp = Blueprint('v2', __name__)

# ---------------------------------------------------------------------------
# V2 Singletons
# ---------------------------------------------------------------------------
v2_execution_engine = ExecutionEngine(deterministic=False)
v2_paper_trader = PaperTraderV2(
    initial_capital=100000,
    execution_engine=v2_execution_engine
)
v2_margin_engine = MarginEngine()
v2_analytics = None    # Needs db_manager — set in init_v2
v2_monte_carlo = MonteCarloSimulator()
v2_allocator = CapitalAllocator()
v2_ranker = StrategyRanker()
v2_regime_detector = RegimeDetector()
v2_volatility_filter = VolatilityFilter()

# Professional Pipeline Components
v2_risk = RiskEngineV2(db_manager=db_manager)
v2_portfolio = PortfolioEngineV2(v2_paper_trader, db_manager)
v2_pipeline = TradingPipelineV2(
    v2_risk, v2_portfolio, v2_paper_trader, db_manager, 
    bot_manager=bot_manager_v2
)

logger.info("[V2] Institutional engine components loaded")


def _publicise_rows(rows):
    """Swap the internal strategy id for its public code on outgoing rows."""
    out = []
    for r in (rows or []):
        row = dict(r)
        if row.get('strategy'):
            internal = row['strategy']
            row['strategy'] = strat_to_public(internal)
            row['strategy_name'] = public_meta(internal)['name']
        out.append(row)
    return out


def _internal_strategy_filter(value):
    """A public code from a query string -> internal id (None means 'all')."""
    if not value:
        return None
    return strat_to_internal(value, default=None)


def _publicise_bot(bot: dict) -> dict:
    """Strip a bot payload of anything that names the underlying strategy."""
    out = dict(bot or {})
    internal = out.get('strategy')
    out['strategy'] = strat_to_public(internal)
    out['strategy_name'] = public_meta(internal)['name']
    out['bot_id'] = mask_bot_id(out.get('bot_id'))
    # config_hash is derived from the internal id, but is not reversible on its
    # own; the human-readable message is, so it never ships.
    out.pop('message', None)
    return out


# ---------------------------------------------------------------------------
# Module-level references (injected by init_v2)
# ---------------------------------------------------------------------------
socketio = None
strategy_engine = None
db_manager = None
crypto_provider = None
stock_provider = None
coingecko_provider = None
smartapi_provider = None
system_state_fn = None  # get_system_state callable


def init_v2(
    _socketio, _strategy_engine, _db_manager,
    _crypto_provider, _stock_provider, _system_state_fn,
    _coingecko_provider=None, _smartapi_provider=None
):
    """Inject shared dependencies from api_server.py."""
    global socketio, strategy_engine, db_manager
    global crypto_provider, stock_provider, system_state_fn
    global coingecko_provider, smartapi_provider
    global v2_analytics

    socketio = _socketio
    strategy_engine = _strategy_engine
    db_manager = _db_manager
    crypto_provider = _crypto_provider
    stock_provider = _stock_provider
    system_state_fn = _system_state_fn
    # Optional so an older caller still works; the candle cascade simply skips
    # any source whose provider is absent.
    coingecko_provider = _coingecko_provider
    smartapi_provider = _smartapi_provider

    # Analytics needs db_manager
    v2_analytics = StrategyAnalytics(db_manager=db_manager)
    
    # Update Pipeline with injected dependencies
    v2_pipeline.socketio = socketio
    v2_pipeline.db = db_manager
    v2_risk.db = db_manager
    v2_portfolio.db = db_manager

    # Sync strategy profiles
    try:
        db_manager.v2_sync_strategy_profiles(REGISTRY)
        
        # ── Position Reconciliation (Restart Recovery) ──
        # Fetch all unique user_ids with open positions and load them
        conn = db_manager._get_connection()
        try:
            cursor = conn.cursor()
            db_manager._execute(cursor, "SELECT DISTINCT user_id FROM v2_positions")
            uids = [row[0] for row in cursor.fetchall()]
            for uid in uids:
                v2_paper_trader.load_positions(uid, db_manager)
            if uids:
                logger.info(f"🔄 [V2] Restored open positions for {len(uids)} users")
        finally:
            db_manager._safe_close(conn, cursor)
            
    except Exception as e:
        logger.error(f"[V2] Initialisation error (profiles/positions): {e}")

    # Positions restored from a previous run may carry exit levels, and manual
    # positions have no bot loop to police them, so the monitor starts with the
    # app rather than waiting for the next manual bracket order.
    try:
        _ensure_position_monitor()
    except Exception as e:
        logger.warning(f"[V2] Could not start position monitor: {e}")

    try:
        from shared.logic.alerts import discord_reports as _dr
        if _dr.is_configured():
            _ensure_report_scheduler()
        else:
            logger.info("[V2] Discord reports dormant — set DISCORD_BOT_TOKEN "
                        "and DISCORD_GUILD_ID to enable")
    except Exception as e:
        logger.warning(f"[V2] Could not start report scheduler: {e}")

    logger.info("[V2] Blueprint initialised with Modular Pipeline")


# ── Session Management ─────────────────────────────────────────────────────

def v2_start_session():
    """Start a new institutional V2 session."""
    sys_state = system_state_fn()
    if sys_state.get_session_id():
        active_db = db_manager.v2_get_active_session_id()
        if active_db == sys_state.get_session_id():
            return sys_state.get_session_id()

    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sys_state.set_session_id(session_id)
    db_manager.v2_create_session(session_id, sys_state.get_engine_version())
    db_manager.v2_update_session_status(session_id, "ACTIVE")
    logger.info(f"🆕 [V2-API] Unified Session Started: {session_id}")
    return session_id

def v2_stop_session(session_id=None):
    """Stop the current or specified V2 session."""
    sys_state = system_state_fn()
    if not session_id:
        session_id = sys_state.get_session_id() or db_manager.v2_get_active_session_id()
    
    if session_id:
        db_manager.v2_stop_session(session_id)
        if sys_state.get_session_id() == session_id:
            sys_state.set_session_id(None)
        return True
    return False


# ============================================================
# V2 STRATEGIES
# ============================================================

@v2_bp.route('/api/v2/strategies', methods=['GET'])
def get_v2_strategies():
    """
    Available strategies, as the PUBLIC catalog.

    The registry's own ids and descriptions name the underlying technique, so
    they are never serialised. Clients see an opaque code and a behavioural
    summary; see shared/logic/strategies/public_catalog.py.
    """
    return jsonify({
        'success': True,
        'strategies': public_catalog([s.get('id') for s in REGISTRY]),
    })


# ============================================================
# V2 TRADE EXECUTION
# ============================================================

@v2_bp.route('/api/v2/trade', methods=['POST'])
@login_required
def v2_trade():
    """V2 trade execution — backend authoritative fill price."""
    try:
        data = request.json
        symbol = data.get('symbol', '')
        side = data.get('side', 'BUY').upper()
        quantity = float(data.get('quantity', 0))
        leverage = float(data.get('leverage', 1.0))
        strategy = data.get('strategy') or 'manual'
        if strategy != 'manual':
            strategy = strat_to_internal(strategy, default='manual')
        volatility = float(data.get('volatility', 0.02))
        volume = float(data.get('volume', 100_000_000))
        margin_mode = data.get('margin_mode', 'isolated')

        # ── Manual exit conditions (absolute prices, optional) ──
        def _opt_price(key):
            raw = data.get(key)
            try:
                val = float(raw)
            except (TypeError, ValueError):
                return 0.0
            return val if val > 0 else 0.0

        stop_loss_price = _opt_price('stop_loss_price')
        take_profit_price = _opt_price('take_profit_price')
        order_type = str(data.get('order_type', 'market')).lower()
        limit_price = _opt_price('limit_price')

        if quantity <= 0:
            return jsonify({'success': False, 'error': 'Quantity must be > 0'}), 400

        if order_type == 'limit' and limit_price <= 0:
            return jsonify({'success': False,
                            'error': 'A limit order requires a limit price'}), 400

        # Regime filter
        filter_result = v2_volatility_filter.filter(strategy, 'UNKNOWN', leverage)
        if not filter_result['allowed']:
            return jsonify({'success': False, 'error': filter_result['reason']}), 403
        leverage = filter_result['leverage']  # May be reduced

        # Use a mock signal object for manual trades to pass through the pipeline
        class ManualSignal:
            def __init__(self, side):
                self.signal = side
                self.score = 1.0
                self.expected_move_pct = 1.0 # High confidence for manual
        
        signal = ManualSignal(side)
        
        # We wrap the manual request into a pseudo-bot config for the pipeline
        class MockBot:
            def __init__(self, user_id, symbol, strategy, leverage, quantity):
                import types
                self.bot_id = "MANUAL"
                self.config = types.SimpleNamespace(
                    user_id=user_id,
                    symbol=symbol,
                    strategy=strategy,
                    leverage=leverage,
                    max_quantity=quantity,
                    position_size=100 # Manual uses requested quantity directly
                )
        
        mock_bot = MockBot(current_user.id, symbol, strategy, leverage, quantity)
        
        # Execute via Pipeline
        results = v2_paper_trader.execute_trade(
            user_id=current_user.id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage,
            strategy=strategy,
            margin_mode=margin_mode
        )
        
        # Sync and process results using pipeline/portfolio logic
        v2_portfolio.sync_position(current_user.id, symbol, results, strategy, leverage)
        
        # Standardized return
        if results and results[0].get('success'):
            v2_pipeline._handle_trade_results(mock_bot, results, side, quantity, results[0].get('fill_price'))

            # Record the user's exit levels on the open position. V2Position
            # already carries stop_loss/take_profit; nothing was writing them,
            # so a manual bracket had nowhere to live.
            if stop_loss_price or take_profit_price:
                pos = v2_paper_trader.get_position(current_user.id, symbol)
                if pos:
                    pos.stop_loss = stop_loss_price
                    pos.take_profit = take_profit_price
                    logger.info(f"[V2] Manual bracket set on {symbol}: "
                                f"SL={stop_loss_price or '-'} TP={take_profit_price or '-'}")
                    _ensure_position_monitor()

            payload = dict(results[0])
            payload['stop_loss_price'] = stop_loss_price
            payload['take_profit_price'] = take_profit_price
            payload['order_type'] = order_type
            return jsonify(payload)
        
        return jsonify({'success': False, 'error': 'Trade failed', 'results': results})
    except Exception as e:
        logger.error(f"[V2] Trade error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# V2 SAVED STRATEGIES
# ------------------------------------------------------------
# Custom strategies lived only in the browser's localStorage, so
# they vanished on a cache clear or a second device and the
# training side could never see them. They are persisted now.
# ============================================================

# Timeframes the data providers can serve. A custom strategy may only pick
# from these.
ALLOWED_INTERVALS = ('1m', '5m', '15m', '30m', '1h', '4h', '1d')


@v2_bp.route('/api/v2/strategies/custom', methods=['GET'])
@login_required
def v2_list_custom_strategies():
    """Every strategy this user has saved."""
    rows = db_manager.get_user_strategies(current_user.id)
    out = []
    for r in rows:
        try:
            definition = json.loads(r.get('definition_json') or '{}')
        except Exception:
            definition = {}
        out.append({
            'id': r.get('strategy_id'),
            'name': r.get('name'),
            'market': r.get('market'),
            'times_used': r.get('times_used', 0),
            'created_at': str(r.get('created_at') or ''),
            **definition,
        })
    return jsonify({'success': True, 'strategies': out})


@v2_bp.route('/api/v2/strategies/custom', methods=['POST'])
@login_required
def v2_save_custom_strategy():
    """Create or update a saved strategy."""
    data = request.json or {}
    strategy_id = (data.get('id') or '').strip()
    name = (data.get('name') or '').strip()

    if not strategy_id or not name:
        return jsonify({'success': False, 'error': 'id and name are required'}), 400

    # A custom strategy carries its own timeframe, exactly as a catalog one
    # does — validated against the intervals the engine can actually fetch, so
    # a hand-edited request cannot save a strategy that will never get candles.
    interval = str(data.get('interval') or '').strip()
    if interval not in ALLOWED_INTERVALS:
        interval = '15m'

    definition = {
        'indicators': data.get('indicators') or [],
        'buyConditions': data.get('buyConditions') or [],
        'sellConditions': data.get('sellConditions') or [],
        'interval': interval,
    }
    if not definition['indicators']:
        return jsonify({'success': False, 'error': 'Select at least one indicator'}), 400

    ok = db_manager.save_user_strategy(
        current_user.id, strategy_id, name,
        json.dumps(definition), data.get('market'))

    if not ok:
        return jsonify({'success': False, 'error': 'Could not save strategy'}), 500
    return jsonify({'success': True, 'id': strategy_id})


@v2_bp.route('/api/v2/strategies/custom', methods=['DELETE'])
@login_required
def v2_delete_custom_strategy():
    data = request.json or {}
    strategy_id = (data.get('id') or '').strip()
    if not strategy_id:
        return jsonify({'success': False, 'error': 'id required'}), 400
    removed = db_manager.delete_user_strategy(current_user.id, strategy_id)
    return jsonify({'success': True, 'removed': removed})


# ============================================================
# V2 BOT STOCK SCREENER
# ------------------------------------------------------------
# Continuously ranks a universe of instruments by how well each
# one currently satisfies the user's chosen strategy, so the
# opportunity finds the user instead of the other way round.
# ============================================================

SCREENER_UNIVERSE = {
    'stocks': ['AAPL', 'MSFT', 'GOOGL', 'TSLA', 'NVDA', 'AMZN', 'META',
               'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'TATAMOTORS'],
    'crypto': ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
               'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT'],
    'forex': ['EURUSD', 'GBPUSD', 'USDJPY', 'USDINR', 'AUDUSD', 'USDCAD'],
    'commodities': ['XAUUSD', 'XAGUSD', 'XBRUSD', 'XTIUSD', 'XCUUSD', 'XNGUSD'],
}

# Screening every symbol on every request would hammer the upstream feeds,
# so results are cached briefly and shared across callers.
_screener_cache = {}
_screener_lock = threading.Lock()
SCREENER_TTL = 120  # seconds


def _screen_one(symbol: str, market: str, strategy: str, interval: str, sensitivity: str):
    """Run the live strategy over one symbol and describe the result."""
    try:
        if market == 'crypto':
            df = crypto_provider.get_historical_klines(symbol=symbol, interval=interval, limit=200)
            price_data = crypto_provider.get_current_price(symbol)
        else:
            df = stock_provider.get_historical_data(symbol=symbol, interval=interval, limit=200)
            price_data = stock_provider.get_current_quote(symbol)

        if df is None or df.empty or len(df) < 52:
            return {'symbol': symbol, 'market': market, 'status': 'insufficient_data',
                    'signal': 'HOLD', 'score': 0.0}

        price = float((price_data or {}).get('price') or 0)
        signal = strategy_engine.analyze(df, strategy, {}, sensitivity)

        score = float(getattr(signal, 'score', 0) or 0)
        action = str(getattr(signal, 'signal', 'HOLD') or 'HOLD').upper()

        reasons = list(getattr(signal, 'reasons', None) or [])

        return {
            'symbol': symbol,
            'market': market,
            'price': price or float(getattr(signal, 'price', 0) or 0),
            'currency': (price_data or {}).get('currency', 'USD'),
            'signal': action,
            'score': round(score, 4),
            'strength': int(getattr(signal, 'strength', 0) or 0),
            'reasons': reasons[:3],
            'status': 'ok',
        }
    except Exception as e:
        logger.debug(f"[V2-SCREENER] {symbol} failed: {e}")
        return {'symbol': symbol, 'market': market, 'status': 'error',
                'signal': 'HOLD', 'score': 0.0, 'error': str(e)}


@v2_bp.route('/api/v2/screener', methods=['GET'])
@login_required
def v2_screener():
    """
    Rank a market's instruments by how strongly they match a strategy.

    Query: market, strategy, interval, sensitivity, limit, refresh
    """
    market = (request.args.get('market') or 'stocks').lower()
    public_strategy = request.args.get('strategy') or 'GBX-01'
    strategy = strat_to_internal(public_strategy)
    interval = request.args.get('interval') or '15m'
    sensitivity = (request.args.get('sensitivity') or 'conservative').lower()
    limit = max(1, min(50, int(request.args.get('limit', 12))))
    refresh = request.args.get('refresh') == '1'

    universe = SCREENER_UNIVERSE.get(market)
    if not universe:
        return jsonify({'success': False,
                        'error': f'No screener universe for market "{market}"'}), 400

    cache_key = (market, strategy, interval, sensitivity)
    now = time.time()

    with _screener_lock:
        hit = _screener_cache.get(cache_key)
        if hit and not refresh and (now - hit['at']) < SCREENER_TTL:
            return jsonify({'success': True, 'cached': True,
                            'age_seconds': int(now - hit['at']),
                            'market': market, 'strategy': strat_to_public(strategy),
                            'strategy_name': public_meta(strategy)['name'],
                            'interval': interval, 'sensitivity': sensitivity,
                            'results': hit['results'][:limit]})

    if not strategy_engine:
        return jsonify({'success': False, 'error': 'Strategy engine unavailable'}), 503

    results = [_screen_one(sym, market, strategy, interval, sensitivity)
               for sym in universe]

    # Actionable first: BUY/SELL ahead of HOLD, then by conviction.
    def rank(r):
        actionable = 0 if r.get('signal') in ('BUY', 'SELL') else 1
        return (actionable, -abs(float(r.get('score') or 0)))

    results.sort(key=rank)

    with _screener_lock:
        _screener_cache[cache_key] = {'at': now, 'results': results}

    return jsonify({'success': True, 'cached': False, 'market': market,
                    'strategy': strat_to_public(strategy),
                    'strategy_name': public_meta(strategy)['name'],
                    'interval': interval,
                    'sensitivity': sensitivity, 'results': results[:limit]})


# ── V2 Manual Position Monitor ─────────────────────────────────────────────
#
# Bot positions get their TP/SL enforced inside each bot's execution loop.
# A MANUAL trade has no such loop, so its exit levels — and its liquidation
# price — were never checked once the order was placed. This thread keeps
# prices fresh for every symbol holding an open position and closes anything
# that reaches the level the user set.

_position_monitor_thread = None
_position_monitor_lock = threading.Lock()


def _fetch_price_for(symbol: str) -> float:
    """Best-effort live price for any asset class."""
    try:
        if symbol.upper().endswith(('USDT', 'USDC', 'BUSD')) and crypto_provider:
            data = crypto_provider.get_current_price(symbol)
        elif stock_provider:
            data = stock_provider.get_current_quote(symbol)
        else:
            return 0.0
        return float((data or {}).get('price') or 0.0)
    except Exception as e:
        logger.debug(f"[V2-MONITOR] price fetch failed for {symbol}: {e}")
        return 0.0


def v2_position_monitor_loop():
    """Poll open positions and enforce manually-set exit levels."""
    logger.info("🛡️ [V2-MONITOR] Manual position monitor started")
    while True:
        try:
            # Snapshot so we never hold the trader lock across a network call.
            with v2_paper_trader.lock:
                open_positions = [
                    (uid, sym, pos.side, pos.stop_loss, pos.take_profit)
                    for uid, acc in v2_paper_trader.accounts.items()
                    for sym, pos in acc.positions.items()
                    if pos.quantity > 0
                ]

            if not open_positions:
                time.sleep(15)
                continue

            for symbol in {sym for _, sym, _, _, _ in open_positions}:
                price = _fetch_price_for(symbol)
                if price > 0:
                    # Also refreshes unrealized P&L and liquidation checks,
                    # which previously only ran while a bot happened to be up.
                    v2_paper_trader.set_prices({symbol: price})

            for user_id, symbol, side, sl, tp in open_positions:
                if not sl and not tp:
                    continue
                price = v2_paper_trader.current_prices.get(symbol) or 0.0
                if price <= 0:
                    continue

                is_long = str(side).upper() == 'LONG'
                action = None
                if sl and ((is_long and price <= sl) or (not is_long and price >= sl)):
                    action = 'STOP_LOSS'
                elif tp and ((is_long and price >= tp) or (not is_long and price <= tp)):
                    action = 'TAKE_PROFIT'

                if not action:
                    continue

                logger.info(f"{'🛑' if action == 'STOP_LOSS' else '🎯'} [V2-MONITOR] "
                            f"{action} on {symbol} @ {price} (SL={sl or '-'} TP={tp or '-'})")
                try:
                    v2_paper_trader.close_position(user_id, symbol, action=action)
                except Exception as e:
                    logger.error(f"[V2-MONITOR] {action} close failed for {symbol}: {e}")

        except Exception as e:
            logger.error(f"[V2-MONITOR] loop error: {e}")

        time.sleep(10)


def _ensure_position_monitor():
    """Start the monitor once, on first use."""
    global _position_monitor_thread
    with _position_monitor_lock:
        if _position_monitor_thread and _position_monitor_thread.is_alive():
            return
        _position_monitor_thread = threading.Thread(
            target=v2_position_monitor_loop, daemon=True, name='v2-position-monitor')
        _position_monitor_thread.start()


# ── V2 Bot Execution Loop ──────────────────────────────────────────────────

def v2_bot_execution_loop(bot_id):
    """V2 institutional bot execution loop — uses V2 execution engine."""
    bot = bot_manager_v2.bots.get(bot_id)
    if not bot:
        logger.error(f"❌ [V2-BOT-{bot_id}] Not found in manager at startup.")
        return

    config = bot.config
    symbol = config.symbol
    interval = config.interval
    strategy_name = config.strategy

    logger.info(f"🚀 [V2-BOT-{bot_id}] Execution loop starting for {symbol} ({interval}, strategy={strategy_name})")

    last_candle_times = {} # symbol-level tracking
    loop_count = 0
    while not bot.stop_flag.is_set():
        loop_count += 1
        try:
            # Heartbeat
            if loop_count % 5 == 0:
                logger.debug(f"💓 [V2-BOT-{bot_id}] Heartbeat — Loop {loop_count}")

            # System pause
            if system_state_fn().is_paused():
                time.sleep(10)
                continue

            # Check bot still exists
            if bot_id not in bot_manager_v2.bots:
                logger.warning(f"⚠️ [V2-BOT-{bot_id}] Removed from manager. Exiting.")
                break

            # Fetch market data
            if config.market == 'crypto':
                df = crypto_provider.get_historical_klines(symbol=symbol, interval=interval, limit=200)
                price_data = crypto_provider.get_current_price(symbol)
            else:
                df = stock_provider.get_historical_data(symbol=symbol, interval=interval, limit=200)
                price_data = stock_provider.get_current_quote(symbol)

            if df.empty or len(df) < 52:
                logger.warning(f"⚠️ [V2-BOT-{bot_id}] Insufficient data: {len(df)} candles")
                time.sleep(10)
                continue

            current_price = price_data.get('price', 0)
            if current_price <= 0:
                time.sleep(5)
                continue

            # Update V2 paper trader prices EVERY iteration (for live unrealized
            # P&L, liquidation checks, and responsive TP/SL monitoring).
            user_id = config.user_id
            v2_paper_trader.set_prices({symbol: current_price})

            atr_series = compute_atr(df, period=14)
            atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

            # ── TP / SL Enforcement (every ~5s, independent of the candle gate) ──
            # Positions must be protected continuously, not only once per candle.
            if v2_pipeline.check_exit_conditions(bot_id, bot, current_price, atr_value):
                time.sleep(5)
                continue

            # ── New Candle Gate (signal generation runs once per candle) ──
            current_candle_time = df.index[-1]
            last_time = last_candle_times.get(symbol)
            if last_time is not None and current_candle_time <= last_time:
                # Same or old candle, wait for next check
                time.sleep(5)
                continue

            last_candle_times[symbol] = current_candle_time

            # ── Signal Generation ──

            logger.debug(f"🔍 [V2-BOT-{bot_id}] Analyzing {symbol} with {strategy_name}")
            signal = strategy_engine.analyze(
                df, strategy=strategy_name,
                sensitivity=getattr(config, 'sensitivity', 'conservative'))

            bot.stats.last_price = current_price
            bot.stats.last_signal = signal.signal
            bot.stats.signals_generated += 1

            # Emit signal to V2 frontend
            signal_data = {
                'time': datetime.now().isoformat(),
                'signal': signal.signal,
                'strength': getattr(signal, 'strength', 0),
                'score': getattr(signal, 'score', 0),
                'price': current_price,
                'strategy': strat_to_public(strategy_name),
                'strategy_name': public_meta(strategy_name)['name'],
                'reasons': getattr(signal, 'reasons', [])[:3],
                'bot_id': mask_bot_id(bot_id),
                'symbol': symbol,
                'engine': 'v2'
            }
            socketio.emit('auto_trade_signal', signal_data, room=f"user_{user_id}")

            # ── V2 Modular Pipeline Execution ──
            # The pipeline handles risk gating, portfolio sizing, and atomic execution
            v2_pipeline.run_tick(
                bot_id=bot_id,
                bot=bot,
                signal=signal,
                current_price=current_price,
                atr_value=atr_value
            )

            time.sleep(5)

        except Exception as e:
            logger.error(f"❌ [V2-BOT-{bot_id}] CRASH in loop: {e}", exc_info=True)
            time.sleep(10)

    logger.info(f"🛑 [V2-BOT-{bot_id}] Execution loop stopped.")


def v2_execute_bot_trade(bot, signal, current_price, symbol_pos,
                        atr_value=0.0, order_book=None):
    """Execute V2 trade using institutional execution engine with smart filters."""
    config = bot.config
    symbol = config.symbol
    user_id = config.user_id
    strategy = config.strategy
    leverage = config.leverage

    has_position = symbol_pos is not None and symbol_pos.get('quantity', 0) != 0
    pos_side = symbol_pos.get('side', '') if has_position else ''
    signal_score = getattr(signal, 'score', 0)

    logger.info(f"📊 [V2-BOT-{bot.bot_id}] DECISION: Signal={signal.signal}, "
                f"Score={signal_score:.2f}, HasPosition={has_position}, "
                f"PosSide={pos_side}, Price={current_price}")

    # TP/SL checks on existing position
    if has_position:
        pnl_pct = symbol_pos.get('unrealized_pnl_pct', 0)
        tp_pct = config.take_profit
        sl_pct = config.stop_loss

        if pnl_pct >= tp_pct:
            logger.info(f"🎯 [V2-BOT-{bot.bot_id}] TAKE PROFIT: {pnl_pct:.2f}% (Limit: {tp_pct}%)")
            close_side = 'SELL' if pos_side == 'LONG' else 'BUY'
            _v2_place_trade(bot, close_side, abs(symbol_pos['quantity']), current_price, leverage, 'TAKE_PROFIT')
            return
        elif pnl_pct <= -sl_pct:
            logger.info(f"🛑 [V2-BOT-{bot.bot_id}] STOP LOSS: {pnl_pct:.2f}% (Limit: {sl_pct}%)")
            close_side = 'SELL' if pos_side == 'LONG' else 'BUY'
            _v2_place_trade(bot, close_side, abs(symbol_pos['quantity']), current_price, leverage, 'STOP_LOSS')
            return

    # ── Signal Quality Gate ──
    # 1. Score Gate
    if signal_score < 0.6:
        logger.debug(f"⏭️ [V2-BOT-{bot.bot_id}] Signal score {signal_score:.2f} < 0.6. Skipping.")
        return

    # 2. Minimum Edge Filter (Cost-aware)
    # Estimate costs: Spread + Commission + Slippage approximation
    # commission is typically 0.02%-0.1% depending on the provider.
    spread_pct = 0.02 # default 0.02%
    comm_pct = 0.04 # typical institutional/retail mix
    slip_pct = 0.04 # estimated slippage
    total_cost_pct = (spread_pct + comm_pct + slip_pct) / 100


def v2_start_bot_thread(bot_id):
    """Start a V2 bot execution thread."""
    if bot_id in bot_manager_v2.bots:
        bot = bot_manager_v2.bots[bot_id]
        if bot.thread and bot.thread.is_alive():
            logger.info(f"ℹ️ [V2] Bot thread {bot_id} already running.")
            return
        thread = threading.Thread(target=v2_bot_execution_loop, args=(bot_id,), daemon=True)
        bot.thread = thread
        thread.start()
        logger.info(f"🧵 [V2] Started execution thread for: {bot_id}")


def v2_restore_bots_on_startup():
    """Re-spawn V2 bots that were running before a restart/reboot.

    Reads persisted bot configs (status='running') from v2_bot_state, re-registers
    each in the bot manager, and starts its execution thread. Called once at boot
    after the engine globals are initialised.
    """
    try:
        running = db_manager.v2_get_running_bots()
    except Exception as e:
        logger.error(f"[V2-RESTORE] Could not read persisted bots: {e}")
        return

    if not running:
        logger.info("[V2-RESTORE] No V2 bots to restore.")
        return

    restored = 0
    for row in running:
        bot_id = row.get('bot_id')
        try:
            cfg = json.loads(row.get('config_json') or '{}')
            if not cfg.get('symbol'):
                logger.warning(f"[V2-RESTORE] Skip {bot_id}: no config")
                continue
            result = bot_manager_v2.start_bot(
                user_id=row['user_id'],
                symbol=cfg.get('symbol'),
                market=cfg.get('market', 'crypto'),
                strategy=cfg.get('strategy', 'combined'),
                mode=cfg.get('mode', 'paper'),
                interval=cfg.get('interval', '1m'),
                position_size=cfg.get('position_size', 10.0),
                stop_loss=cfg.get('stop_loss', 5.0),
                take_profit=cfg.get('take_profit', 10.0),
                max_quantity=cfg.get('max_quantity', 1.0),
                leverage=cfg.get('leverage', 1.0),
                risk_pct=cfg.get('risk_pct', 2.0),
                sensitivity=cfg.get('sensitivity', 'conservative'),
            )
            if result.get('success'):
                v2_start_bot_thread(result['bot_id'])
                restored += 1
                logger.info(f"[V2-RESTORE] ♻️ Restored {result['bot_id']} "
                            f"({cfg.get('strategy')} {cfg.get('symbol')} {cfg.get('sensitivity')})")
            else:
                logger.warning(f"[V2-RESTORE] Skip {bot_id}: {result.get('error')}")
        except Exception as e:
            logger.error(f"[V2-RESTORE] Failed to restore {bot_id}: {e}")

    logger.info(f"[V2-RESTORE] ✅ Restored {restored} V2 bot(s).")


# ============================================================
# V2 WATCHLIST / FAVORITES
# ============================================================

@v2_bp.route('/api/v2/watchlist', methods=['GET'])
def v2_get_watchlist():
    """Get user's watchlist (favorites)."""
    try:
        user_id = request.args.get('user_id', 1, type=int)
        watchlist = db_manager.get_user_watchlist(user_id)
        # Serialize datetime objects
        for item in watchlist:
            for k, v in item.items():
                if hasattr(v, 'isoformat'):
                    item[k] = v.isoformat()
        return jsonify({'success': True, 'watchlist': watchlist})
    except Exception as e:
        logger.error(f"[V2] Watchlist GET error: {e}")
        return jsonify({'success': False, 'watchlist': [], 'error': str(e)}), 500


@v2_bp.route('/api/v2/watchlist', methods=['POST'])
def v2_add_to_watchlist():
    """Add a symbol to user's watchlist."""
    try:
        data = request.json
        user_id = data.get('user_id', 1)
        symbol = data.get('symbol', '')
        market = data.get('market', 'crypto')
        name = data.get('name', '')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        added = db_manager.add_to_watchlist(user_id, symbol, market, name)
        return jsonify({'success': True, 'added': added, 'symbol': symbol})
    except Exception as e:
        logger.error(f"[V2] Watchlist POST error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/watchlist', methods=['DELETE'])
def v2_remove_from_watchlist():
    """Remove a symbol from user's watchlist."""
    try:
        data = request.json
        user_id = data.get('user_id', 1)
        symbol = data.get('symbol', '')
        
        if not symbol:
            return jsonify({'success': False, 'error': 'Symbol is required'}), 400
        
        removed = db_manager.remove_from_watchlist(user_id, symbol)
        return jsonify({'success': True, 'removed': removed, 'symbol': symbol})
    except Exception as e:
        logger.error(f"[V2] Watchlist DELETE error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# V2 BOT MANAGEMENT ROUTES
# ============================================================

@v2_bp.route('/api/v2/start-bot', methods=['POST'])
@login_required
def v2_start_bot():
    """Start a V2 bot with strategy + config hash."""
    try:
        data = request.json
        strategy = strat_to_internal(data.get('strategy'))
        # The timeframe belongs to the strategy, not the caller. A trend
        # strategy polled every minute is reading noise; a mean-reversion one on
        # a daily candle never sees the stretch it exists to fade. Deriving it
        # here (rather than only hiding the dropdown) means a hand-rolled
        # request cannot put a strategy on a timeframe it was not built for.
        #
        # A custom strategy runs on the generalist engine but keeps the
        # timeframe its author chose in the Forge, so it is looked up rather
        # than inherited from whatever engine executes it.
        interval = interval_for(strategy)
        custom_id = (data.get('custom_strategy_id') or '').strip()
        if custom_id:
            for row in (db_manager.get_user_strategies(current_user.id) or []):
                if row.get('strategy_id') != custom_id:
                    continue
                try:
                    saved = json.loads(row.get('definition_json') or '{}')
                except Exception:
                    saved = {}
                if saved.get('interval') in ALLOWED_INTERVALS:
                    interval = saved['interval']
                break
        result = bot_manager_v2.start_bot(
            user_id=current_user.id,
            symbol=data.get('symbol', ''),
            market=data.get('market', 'crypto'),
            # Clients speak in public codes; the engine speaks internal ids.
            strategy=strategy,
            mode=data.get('mode', 'paper'),
            interval=interval,
            position_size=float(data.get('position_size', 10.0)),
            stop_loss=float(data.get('stop_loss', 5.0)),
            take_profit=float(data.get('take_profit', 10.0)),
            max_quantity=float(data.get('max_quantity', 1.0)),
            leverage=float(data.get('leverage', 1.0)),
            risk_pct=float(data.get('risk_pct', 2.0)),
            sensitivity=str(data.get('sensitivity', 'conservative')).lower(),
        )

        # Start the execution thread if bot was registered successfully
        if result.get('success'):
            # Ensure a session is active
            active_session = db_manager.v2_get_active_session_id()
            if not active_session:
                v2_start_session()
            
            v2_start_bot_thread(result['bot_id'])

        if result.get('bot_id'):
            result = dict(result)
            result['bot_id'] = mask_bot_id(result['bot_id'])
            result.pop('message', None)   # names the strategy in plain text

        return jsonify(result)
    except Exception as e:
        logger.error(f"[V2] Start bot error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/stop-bot', methods=['POST'])
@login_required
def v2_stop_bot():
    """Stop a V2 bot."""
    try:
        data = request.json
        bot_id = unmask_bot_id(data.get('bot_id', ''))
        result = bot_manager_v2.stop_bot(bot_id)
        
        # If no bots are running, stop the session
        active_bots = bot_manager_v2.get_all_bots(user_id=current_user.id)
        running_bots = [b for b in active_bots if b.get('status') == 'running']
        if not running_bots:
            v2_stop_session()
            
        return jsonify(result)
    except Exception as e:
        logger.error(f"[V2] Stop bot error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# V2 DATA ENDPOINTS
# ============================================================

@v2_bp.route('/api/v2/bots', methods=['GET'])
@login_required
def v2_list_bots():
    """List all V2 bots for the current user."""
    bots = bot_manager_v2.get_all_bots(user_id=current_user.id)

    # Enrich with persisted per-bot stats from the ledger so trade counts / realized
    # P&L survive engine restarts (in-memory bot.stats reset on restart).
    ledger_stats = db_manager.v2_get_bot_ledger_stats(current_user.id)
    for b in bots:
        ls = ledger_stats.get(b.get('bot_id'))
        stats = b.setdefault('stats', {})
        if ls:
            # Ledger is authoritative for realized figures; keep the higher of the two
            # so a fresh in-memory count during the current run is never undercounted.
            stats['total_trades'] = max(int(stats.get('total_trades') or 0), ls['trades'])
            stats['realized_pnl'] = ls['realized_pnl']
            stats['total_pnl'] = ls['realized_pnl'] + float(stats.get('unrealized_pnl') or 0)
            for k in ('closed_trades', 'wins', 'losses', 'breakeven', 'win_rate',
                      'loss_rate', 'avg_win', 'avg_loss', 'profit_factor'):
                stats[k] = ls[k]
        else:
            stats.setdefault('closed_trades', 0)

    return jsonify({'success': True, 'bots': [_publicise_bot(b) for b in bots]})


@v2_bp.route('/api/v2/positions', methods=['GET'])
@login_required
def v2_positions():
    """Get V2 positions with leverage and margin data."""
    positions = v2_paper_trader.get_positions(current_user.id)
    return jsonify({'success': True, 'positions': positions})


@v2_bp.route('/api/v2/position/exits', methods=['POST'])
@login_required
def v2_set_position_exits():
    """
    Set, change or clear the exit levels on an OPEN position.

    Exit levels could previously only be chosen when the order was placed;
    there was no way to protect a trade already running. Pass 0 to clear.
    """
    data = request.json or {}
    symbol = (data.get('symbol') or '').strip()
    if not symbol:
        return jsonify({'success': False, 'error': 'symbol required'}), 400

    def _price(key):
        try:
            val = float(data.get(key) or 0)
        except (TypeError, ValueError):
            return -1.0
        return val if val >= 0 else -1.0

    stop_loss = _price('stop_loss_price')
    take_profit = _price('take_profit_price')
    if stop_loss < 0 or take_profit < 0:
        return jsonify({'success': False, 'error': 'Exit levels must be numbers'}), 400

    pos = v2_paper_trader.get_position(current_user.id, symbol)
    if not pos or pos.quantity <= 0:
        return jsonify({'success': False, 'error': f'No open position for {symbol}'}), 404

    entry = float(pos.entry_price or 0)
    is_short = str(pos.side).upper() == 'SHORT'

    # Reject a level that sits on the wrong side of the entry — it would fire
    # on the monitor's very next tick.
    if stop_loss > 0 and entry > 0:
        if not is_short and stop_loss >= entry:
            return jsonify({'success': False,
                            'error': 'Stop loss must be below the entry for a long'}), 400
        if is_short and stop_loss <= entry:
            return jsonify({'success': False,
                            'error': 'Stop loss must be above the entry for a short'}), 400

    if take_profit > 0 and entry > 0:
        if not is_short and take_profit <= entry:
            return jsonify({'success': False,
                            'error': 'Take profit must be above the entry for a long'}), 400
        if is_short and take_profit >= entry:
            return jsonify({'success': False,
                            'error': 'Take profit must be below the entry for a short'}), 400

    pos.stop_loss = stop_loss
    pos.take_profit = take_profit
    logger.info(f"[V2] Exits updated on {symbol}: SL={stop_loss or '-'} TP={take_profit or '-'}")

    _ensure_position_monitor()

    return jsonify({'success': True, 'symbol': symbol,
                    'stop_loss': stop_loss, 'take_profit': take_profit})


# ============================================================
# MARKET CONDITIONS
# ------------------------------------------------------------
# Replaces the old momentum "pulse", which showed a score from
# three if-statements next to a sentence picked at random from a
# hardcoded list. Every number here is measured and explainable.
# ============================================================

_conditions_cache = {}
_conditions_lock = threading.Lock()
CONDITIONS_TTL = 45  # seconds


@v2_bp.route('/api/v2/conditions/<symbol>', methods=['GET'])
@login_required
def v2_conditions(symbol):
    """Is now a good time to trade this instrument, and why not."""
    from v2.engine.intelligence import conditions as conditions_engine

    symbol = (symbol or '').upper()
    market = (request.args.get('market') or '').lower()
    interval = request.args.get('interval') or '15m'

    if not market:
        market = 'crypto' if symbol.endswith(('USDT', 'USDC', 'BUSD')) else 'stocks'

    key = (symbol, market, interval)
    now = time.time()
    with _conditions_lock:
        hit = _conditions_cache.get(key)
        if hit and (now - hit['at']) < CONDITIONS_TTL:
            return jsonify({'success': True, 'cached': True, **hit['data']})

    try:
        if market == 'crypto' and crypto_provider:
            df = crypto_provider.get_historical_klines(symbol=symbol, interval=interval, limit=150)
        elif stock_provider:
            df = stock_provider.get_historical_data(symbol=symbol, interval=interval, limit=150)
        else:
            return jsonify({'success': False, 'error': 'No data provider'}), 503

        data = conditions_engine.read(df, symbol)
    except Exception as e:
        logger.error(f"[V2] Conditions error for {symbol}: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500

    with _conditions_lock:
        _conditions_cache[key] = {'at': now, 'data': data}

    return jsonify({'success': True, 'cached': False, **data})


# ============================================================
# DISCORD DAILY REPORTS
# ------------------------------------------------------------
# Opt-in per user. On approval the bot creates #{username}_report
# in the configured guild; a scheduler posts today's numbers plus
# the running total once a day.
# ============================================================

_report_thread = None
_report_thread_lock = threading.Lock()


def _user_display_name(user_id: int) -> str:
    """Best-effort username for the channel name and report title."""
    try:
        user = db_manager.get_user_by_id(user_id)
        if user:
            return (user.get('username') or user.get('name')
                    or user.get('email', '').split('@')[0] or 'trader-%s' % user_id)
    except Exception as e:
        logger.debug(f"[REPORTS] username lookup failed: {e}")
    return 'trader-%s' % user_id


def _build_user_report(user_id: int):
    """Today's figures and the running total for one user."""
    from shared.logic.alerts import discord_reports as dr

    all_trades = db_manager.v2_get_user_trades(user_id=user_id, limit=5000) or []

    today = datetime.now(timezone.utc).date()
    todays = []
    for t in all_trades:
        ts = t.get('timestamp') or t.get('trade_time')
        if not ts:
            continue
        try:
            when = ts.date() if hasattr(ts, 'date') else datetime.fromisoformat(str(ts)[:19]).date()
        except Exception:
            continue
        if when == today:
            todays.append(t)

    balance, open_count = None, 0
    try:
        info = v2_paper_trader.get_account_info(user_id) or {}
        balance = info.get('equity', info.get('total_value'))
        open_count = len(v2_paper_trader.get_positions(user_id) or [])
    except Exception as e:
        logger.debug(f"[REPORTS] account snapshot failed: {e}")

    return dr.build_embed(
        _user_display_name(user_id),
        dr.summarise(todays),
        dr.summarise(all_trades),
        balance=balance,
        open_positions=open_count,
    )


def send_user_report(user_id: int, force: bool = False):
    """Post one user's report. Returns (ok, message)."""
    from shared.logic.alerts import discord_reports as dr

    if not dr.is_configured():
        return False, ('Discord is not configured on the server '
                       '(DISCORD_BOT_TOKEN / DISCORD_GUILD_ID).')

    sub = db_manager.get_report_subscription(user_id)
    if not sub or not sub.get('enabled'):
        return False, 'Reports are not enabled for this user.'

    today_str = datetime.now(timezone.utc).strftime('%Y-%m-%d')
    if not force and sub.get('last_sent_date') == today_str:
        return False, 'Already sent today.'

    username = _user_display_name(user_id)
    channel_id = dr.ensure_channel(username, sub.get('channel_id'))
    if not channel_id:
        return False, 'Could not create or reach the Discord channel.'

    if channel_id != sub.get('channel_id'):
        db_manager.save_report_subscription(
            user_id, True, channel_id, dr.channel_name_for(username),
            sub.get('last_sent_date'))

    embed = _build_user_report(user_id)
    if not dr.post_report(channel_id, embed):
        return False, 'Discord rejected the message.'

    db_manager.mark_report_sent(user_id, today_str)
    logger.info(f"[REPORTS] daily report delivered for user {user_id}")
    return True, 'Report sent.'


def report_scheduler_loop():
    """Post each subscriber's report once, after the configured hour."""
    from shared.logic.alerts import discord_reports as dr

    hour = int(os.getenv('DISCORD_REPORT_HOUR_UTC', '21'))
    logger.info(f"📮 [REPORTS] scheduler running — daily post at {hour:02d}:00 UTC")

    while True:
        try:
            now = datetime.now(timezone.utc)
            if now.hour >= hour and dr.is_configured():
                today_str = now.strftime('%Y-%m-%d')
                for sub in db_manager.get_report_subscribers():
                    if sub.get('last_sent_date') == today_str:
                        continue
                    try:
                        ok, msg = send_user_report(sub['user_id'])
                        if not ok:
                            logger.warning(f"[REPORTS] user {sub['user_id']}: {msg}")
                    except Exception as e:
                        logger.error(f"[REPORTS] send failed for {sub.get('user_id')}: {e}")
        except Exception as e:
            logger.error(f"[REPORTS] scheduler error: {e}")

        # Checking every 15 minutes is plenty for a once-a-day post.
        time.sleep(900)


def _ensure_report_scheduler():
    global _report_thread
    with _report_thread_lock:
        if _report_thread and _report_thread.is_alive():
            return
        _report_thread = threading.Thread(target=report_scheduler_loop,
                                          daemon=True, name='v2-report-scheduler')
        _report_thread.start()


@v2_bp.route('/api/v2/reports/subscription', methods=['GET'])
@login_required
def v2_get_report_subscription():
    """Current opt-in state for the signed-in user."""
    from shared.logic.alerts import discord_reports as dr

    sub = db_manager.get_report_subscription(current_user.id) or {}
    username = _user_display_name(current_user.id)
    return jsonify({
        'success': True,
        'enabled': bool(sub.get('enabled')),
        'channel_name': sub.get('channel_name') or dr.channel_name_for(username),
        'last_sent_date': sub.get('last_sent_date'),
        'configured': dr.is_configured(),
        'report_hour_utc': int(os.getenv('DISCORD_REPORT_HOUR_UTC', '21')),
    })


@v2_bp.route('/api/v2/reports/subscription', methods=['POST'])
@login_required
def v2_set_report_subscription():
    """
    Approve or revoke Discord reporting.

    Approving creates the user's channel straight away so they can see where
    their reports will land before the first one is due.
    """
    from shared.logic.alerts import discord_reports as dr

    data = request.json or {}
    enabled = bool(data.get('enabled'))
    username = _user_display_name(current_user.id)

    if not enabled:
        db_manager.save_report_subscription(current_user.id, False)
        logger.info(f"[REPORTS] user {current_user.id} opted out")
        return jsonify({'success': True, 'enabled': False})

    if not dr.is_configured():
        return jsonify({
            'success': False,
            'error': 'Discord reporting is not configured on this server yet.',
        }), 503

    existing = db_manager.get_report_subscription(current_user.id) or {}
    channel_id = dr.ensure_channel(username, existing.get('channel_id'))
    if not channel_id:
        return jsonify({
            'success': False,
            'error': 'Could not create your Discord channel. The bot may be '
                     'missing the Manage Channels permission.',
        }), 502

    channel_name = dr.channel_name_for(username)
    db_manager.save_report_subscription(current_user.id, True, channel_id, channel_name)
    _ensure_report_scheduler()

    dr.post_text(channel_id,
                 '**Reports enabled** for `%s`. A summary of the day plus running '
                 'totals will be posted here each day at %02d:00 UTC.'
                 % (username, int(os.getenv('DISCORD_REPORT_HOUR_UTC', '21'))))

    logger.info(f"[REPORTS] user {current_user.id} opted in -> #{channel_name}")
    return jsonify({'success': True, 'enabled': True, 'channel_name': channel_name})


@v2_bp.route('/api/v2/reports/send-now', methods=['POST'])
@login_required
def v2_send_report_now():
    """Post the current report immediately — used to verify the setup."""
    ok, msg = send_user_report(current_user.id, force=True)
    return jsonify({'success': ok, 'message': msg}), (200 if ok else 400)


@v2_bp.route('/api/v2/health', methods=['GET'])
def v2_health():
    """Engine health & active session status with telemetry."""
    db_latency = db_manager.check_db_health()
    # Try to get last trade time for user 1 (primary dashboard owner)
    last_trade = db_manager.v2_get_last_trade_time(1) 
    
    return jsonify({
        'engine_status': 'running',
        'current_session': system_state_fn().get_session_id(),
        'engine_version': system_state_fn().get_engine_version(),
        'active_bots': len(bot_manager_v2.bots),
        'last_trade_time': last_trade,
        'db_latency_ms': round(db_latency, 2),
        'timestamp': datetime.utcnow().isoformat()
    })


@v2_bp.route('/api/v2/current-session', methods=['GET'])
@login_required
def v2_current_session():
    """Get active session ID."""
    return jsonify({
        'success': True,
        'session_id': system_state_fn().get_session_id()
    })


@v2_bp.route('/api/v2/sessions', methods=['GET'])
@login_required
def v2_list_sessions():
    """List all sessions with metadata (start time, trade count, engine version)."""
    global _SESSIONS_CACHE
    now = time.time()
    
    if _SESSIONS_CACHE['data'] and (now - _SESSIONS_CACHE['timestamp']) < SESSIONS_TTL:
        return jsonify({'success': True, 'sessions': _SESSIONS_CACHE['data'], 'cached': True})

    sessions = db_manager.v2_get_sessions()
    # Serialize datetime and add professional display labels
    for s in sessions:
        start_dt = s.get('start_time')
        if start_dt and hasattr(start_dt, 'isoformat'):
            # Formatted Label: Session • Mar 15 • 12:09 PM | Trades: 18 | PnL: +$230
            friendly_date = start_dt.strftime("%b %d • %I:%M %p")
            trades = s.get('total_trades', 0)
            pnl = s.get('total_pnl', 0.0)
            pnl_str = f"{'+$' if pnl >= 0 else '-$'}{abs(pnl):.2f}"
            
            s['display_label'] = f"Session • {friendly_date} | {trades} trades | {pnl_str}"
            s['start_time'] = start_dt.isoformat()
    
    _SESSIONS_CACHE = {'data': sessions, 'timestamp': now}
    return jsonify({'success': True, 'sessions': sessions, 'cached': False})


@v2_bp.route('/api/v2/trades', methods=['GET'])
@login_required
def v2_trade_history():
    """Get V2 trade history with optional strategy, session, and date filters."""
    strategy = _internal_strategy_filter(request.args.get('strategy'))
    session_id = request.args.get('session_id')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    
    # Pagination & Protection
    limit = int(request.args.get('limit', 100))
    offset = int(request.args.get('offset', 0))
    if limit > 5000: limit = 5000

    # Range & Format Validation
    if start_date and end_date:
        try:
            # Handle ISO formats from JS
            sd = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
            ed = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
            if sd > ed:
                return jsonify({'success': False, 'error': 'Start date cannot be after end date'}), 400
            if (ed - sd).days > 365:
                return jsonify({'success': False, 'error': 'Query range limited to 365 days'}), 400
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid date format: {e}'}), 400

    total = db_manager.v2_get_total_trade_count(
        user_id=current_user.id,
        strategy=strategy,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date
    )

    trades = db_manager.v2_get_user_trades(
        user_id=current_user.id,
        strategy=strategy,
        session_id=session_id,
        start_date=start_date,
        end_date=end_date,
        limit=limit,
        offset=offset
    )
    # Institutional: ensure price/quantity/pnl are present for frontend
    # and serialize datetime objects
    for t in trades:
        t['fill_price'] = t.get('price') # Alias for frontend
        t['net_pnl'] = t.get('pnl')     # Alias for frontend
        t['trade_time'] = t.get('timestamp') # Alias for frontend
        for k, v in t.items():
            if hasattr(v, 'isoformat'):
                t[k] = v.isoformat()
                
    return jsonify({
        'success': True, 
        'total': total,
        'limit': limit,
        'offset': offset,
        # The ledger stores internal strategy ids; the client only ever sees codes.
        'trades': _publicise_rows(trades)
    })


@v2_bp.route('/api/v2/stop-all', methods=['POST'])
@login_required
def v2_stop_all_bots():
    """Stop all V2 bots for the current user."""
    try:
        user_bots = [
            bid for bid, bot in bot_manager_v2.bots.items()
            if bot.user_id == current_user.id
        ]
        for bot_id in user_bots:
            bot_manager_v2.stop_bot(bot_id)
        return jsonify({
            'success': True,
            'stopped': len(user_bots),
            'message': f'Stopped {len(user_bots)} V2 bots'
        })
    except Exception as e:
        logger.error(f"[V2] Stop all error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/evolution/status', methods=['GET'])
@login_required
def v2_evolution_status():
    """Current evolved parameters + live metrics per strategy."""
    from v2.engine.evolution.evolution_engine import (
        evolution_engine, DEFAULT_PARAMS, MIN_TRADES, SENSITIVITY_BY_RANK)

    # Evolution is per (strategy, symbol) — the same strategy needs different
    # stops on BTC than on a forex pair.
    #
    # Everything below is derived from THREE queries, not three per pair. Each
    # helper used to load its own trades, state and history, so rendering N
    # pairs opened roughly 7N database connections — fine on an empty account,
    # slow enough to look hung on a real one.
    all_trades = db_manager.v2_get_user_trades(user_id=current_user.id, limit=20000)
    pairs = {(t.get('strategy'), t.get('symbol')) for t in all_trades
             if t.get('strategy') and t.get('symbol')}

    # Closing events per pair, oldest first — the journey needs the order, and
    # the metrics window is a slice off the end of it.
    from v2.engine.evolution.evolution_engine import CLOSING_ACTIONS, _ts_key
    closes_by_pair = {}
    for t in all_trades:
        if t.get('action') in CLOSING_ACTIONS and t.get('strategy') and t.get('symbol'):
            closes_by_pair.setdefault((t['strategy'], t['symbol']), []).append(t)
    for rows in closes_by_pair.values():
        rows.sort(key=_ts_key)

    states_by_pair = {(st.get('strategy'), st.get('symbol')): st
                      for st in (db_manager.v2_get_evolution_state(current_user.id) or [])}

    history_by_strategy = {}
    for h in db_manager.v2_get_evolution_history(current_user.id, None, limit=500):
        history_by_strategy.setdefault(h.get('strategy'), []).append(h)

    # Include running bots too. Without this a bot that hasn't closed a trade
    # yet has no card at all, so there is nothing to evaluate or stop.
    running_pairs = set()
    try:
        for b in bot_manager_v2.get_all_bots(user_id=current_user.id):
            if b.get('strategy') and b.get('symbol'):
                pairs.add((b['strategy'], b['symbol']))
                if b.get('status') == 'running':
                    running_pairs.add((b['strategy'], b['symbol']))
    except Exception as e:
        logger.warning(f"[V2-EVO] could not fold bots into status: {e}")

    out = []
    for s, sym in sorted(pairs):
        closes = closes_by_pair.get((s, sym), [])
        # Metrics read the most recent slice, matching the engine's own query
        # bound; the journey needs the whole run so it can show where it began.
        recent = closes[-200:]
        state = states_by_pair.get((s, sym)) or {}
        m = evolution_engine.compute_metrics(recent)
        params = evolution_engine.live_params(current_user.id, s, sym, state=state or None)
        meter = evolution_engine.readiness(current_user.id, s, sym,
                                           trades=recent, state=state)

        pending = None
        if state.get('pending_json'):
            try:
                pending = json.loads(state['pending_json'])
            except Exception:
                pending = None

        profiles = {}
        if state.get('profiles_json'):
            try:
                profiles = json.loads(state['profiles_json'])
            except Exception:
                profiles = {}

        auto_apply = state.get('auto_apply')

        out.append({
            'strategy': strat_to_public(s),
            'strategy_name': public_meta(s)['name'],
            'symbol': sym,
            'running': (s, sym) in running_pairs,
            'generation': state.get('generation', 0),
            'status': state.get('status', 'active'),
            # Autopilot defaults ON — a pair with no state row yet is still
            # going to learn continuously once it starts closing trades.
            'auto_apply': True if auto_apply is None else bool(auto_apply),
            'params': params,
            'defaults': DEFAULT_PARAMS,
            'sensitivity': SENSITIVITY_BY_RANK[int(params.get('sensitivity_rank', 0))],
            'metrics': m,
            'meter': meter,
            # Where the win rate started vs where it is now — the evidence that
            # the bot actually learned from its mistakes.
            'journey': evolution_engine.win_rate_journey(
                current_user.id, s, sym, state, series=closes,
                history=history_by_strategy.get(s, [])),
            'regime': state.get('regime') or 'unknown',
            'profiles': profiles,          # what it learned per market regime
            'pending': pending,            # the lesson awaiting Proceed
            'trades_needed': max(0, MIN_TRADES - m['closed_trades']),
        })

    # A pair whose bot is not running is DORMANT, not active: evolution only
    # ever evaluates on a closed trade, and a stopped bot closes nothing. Its
    # learned generations are deliberately kept — throwing them away would mean
    # restarting the bot began again from generation zero — but it must not be
    # counted or displayed alongside the strategies actually running, which is
    # how four bots came to show seven live evolutions.
    running = [r for r in out if r['running']]
    idle = [r for r in out if not r['running']]

    return jsonify({'success': True, 'strategies': out,
                    'counts': {'running': len(running), 'idle': len(idle)},
                    'min_trades': MIN_TRADES})


@v2_bp.route('/api/v2/evolution/history', methods=['GET'])
@login_required
def v2_evolution_history():
    """Generation-by-generation audit trail of what changed and why."""
    strategy = _internal_strategy_filter(request.args.get('strategy'))
    limit = int(request.args.get('limit', 100))
    rows = db_manager.v2_get_evolution_history(current_user.id, strategy, limit)
    for r in rows:
        for k in ('params_before', 'params_after', 'changes_json'):
            if r.get(k):
                try:
                    r[k] = json.loads(r[k])
                except Exception:
                    pass
    return jsonify({'success': True, 'history': _publicise_rows(rows)})


@v2_bp.route('/api/v2/evolution/approve', methods=['POST'])
@login_required
def v2_evolution_approve():
    """PROCEED — apply the reviewed lesson to live trading."""
    from v2.engine.evolution.evolution_engine import evolution_engine, SENSITIVITY_BY_RANK

    data = request.json or {}
    # Clients send the public code; the engine keys on the internal id.
    strategy = strat_to_internal(data.get('strategy'), default=None) \
        if data.get('strategy') else None
    symbol = data.get('symbol', 'ALL')
    if not strategy:
        return jsonify({'success': False, 'error': 'strategy required'}), 400

    result = evolution_engine.approve(current_user.id, strategy, symbol)
    if not result.get('success'):
        return jsonify(result), 400

    # Push the approved params onto every running bot using this strategy.
    params = result.get('params') or {}
    applied_to = []
    for bot in list(bot_manager_v2.bots.values()):
        if bot.user_id != current_user.id or bot.config.strategy != strategy:
            continue
        if symbol and symbol != 'ALL' and bot.config.symbol != symbol:
            continue
        if 'take_profit' in params:
            bot.config.take_profit = float(params['take_profit'])
        if 'stop_loss' in params:
            bot.config.stop_loss = float(params['stop_loss'])
        if 'sensitivity_rank' in params:
            rank = int(max(0, min(2, params['sensitivity_rank'])))
            bot.config.sensitivity = SENSITIVITY_BY_RANK[rank]
        # Applying a lesson never stops the bot. De-risking keeps it trading at
        # the smallest exposure its bounds allow, because a stopped bot closes
        # no trades and so can never learn its way back to profit.
        applied_to.append(bot.bot_id)

    logger.info(f"🧬 [V2-EVO] Approved {strategy}/{symbol} gen {result.get('generation')} — "
                f"applied to {len(applied_to)} running bot(s)")
    result['applied_to'] = applied_to
    return jsonify(result)


@v2_bp.route('/api/v2/evolution/dismiss', methods=['POST'])
@login_required
def v2_evolution_dismiss():
    """Discard a pending lesson without applying it."""
    from v2.engine.evolution.evolution_engine import evolution_engine
    data = request.json or {}
    # Clients send the public code; the engine keys on the internal id.
    strategy = strat_to_internal(data.get('strategy'), default=None) \
        if data.get('strategy') else None
    symbol = data.get('symbol', 'ALL')
    if not strategy:
        return jsonify({'success': False, 'error': 'strategy required'}), 400
    return jsonify(evolution_engine.dismiss(current_user.id, strategy, symbol))


@v2_bp.route('/api/v2/evolution/candidates', methods=['GET'])
@login_required
def v2_evolution_candidates():
    """
    Everything the user could evaluate — what the Evaluate picker lists.

    Union of (a) currently running bots and (b) pairs that already have trade
    history. A freshly started bot has no closed trades yet, so history alone
    would hide it from the picker; that is why only the one pair with trades
    ever appeared before.
    """
    from v2.engine.evolution.evolution_engine import evolution_engine

    pairs = {}

    def add(strategy, symbol, running):
        if not strategy or not symbol:
            return
        key = (strategy, symbol)
        entry = pairs.setdefault(key, {'strategy': strategy, 'symbol': symbol,
                                       'running': False})
        entry['running'] = entry['running'] or running

    try:
        for b in bot_manager_v2.get_all_bots(user_id=current_user.id):
            add(b.get('strategy'), b.get('symbol'), b.get('status') == 'running')
    except Exception as e:
        logger.warning(f"[V2-EVO] could not list bots for candidates: {e}")

    try:
        for t in db_manager.v2_get_user_trades(user_id=current_user.id, limit=5000):
            add(t.get('strategy'), t.get('symbol'), False)
    except Exception as e:
        logger.warning(f"[V2-EVO] could not list trades for candidates: {e}")

    out = []
    for (strategy, symbol), entry in sorted(pairs.items()):
        state = db_manager.v2_get_evolution_state(current_user.id, strategy, symbol) or {}
        meter = evolution_engine.readiness(current_user.id, strategy, symbol)
        out.append({
            **entry,
            'strategy': strat_to_public(strategy),
            'strategy_name': public_meta(strategy)['name'],
            'generation': state.get('generation', 0),
            'status': state.get('status', 'active'),
            'auto_apply': True if state.get('auto_apply') is None
                          else bool(state.get('auto_apply')),
            'has_pending': bool(state.get('pending_json')),
            'meter': meter,
        })

    return jsonify({'success': True, 'candidates': out})


@v2_bp.route('/api/v2/evolution/status-set', methods=['POST'])
@login_required
def v2_evolution_set_status():
    """Stop ('paused') or resume ('active') evaluation for one pair."""
    from v2.engine.evolution.evolution_engine import evolution_engine
    data = request.json or {}
    # Clients send the public code; the engine keys on the internal id.
    strategy = strat_to_internal(data.get('strategy'), default=None) \
        if data.get('strategy') else None
    symbol = data.get('symbol', 'ALL')
    status = str(data.get('status', 'paused')).lower()
    if not strategy:
        return jsonify({'success': False, 'error': 'strategy required'}), 400
    try:
        return jsonify(evolution_engine.set_status(current_user.id, strategy, symbol, status))
    except Exception as e:
        logger.error(f"[V2-EVO] set status failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/evolution/autopilot', methods=['POST'])
@login_required
def v2_evolution_set_autopilot():
    """Turn continuous self-application of lessons on or off for one pair.

    Learning runs either way — this only decides whether the engine applies a
    lesson the moment the evidence supports it, or waits for Proceed.
    """
    from v2.engine.evolution.evolution_engine import evolution_engine
    data = request.json or {}
    strategy = strat_to_internal(data.get('strategy'), default=None) \
        if data.get('strategy') else None
    symbol = data.get('symbol', 'ALL')
    enabled = bool(data.get('enabled', True))
    if not strategy:
        return jsonify({'success': False, 'error': 'strategy required'}), 400
    try:
        return jsonify(evolution_engine.set_auto_apply(
            current_user.id, strategy, symbol, enabled))
    except Exception as e:
        logger.error(f"[V2-EVO] autopilot toggle failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/evolution/journey', methods=['GET'])
@login_required
def v2_evolution_journey():
    """Win-rate journey for one pair, or for every pair the user trades."""
    from v2.engine.evolution.evolution_engine import evolution_engine

    strategy = _internal_strategy_filter(request.args.get('strategy'))
    symbol = request.args.get('symbol') or 'ALL'

    if strategy:
        state = db_manager.v2_get_evolution_state(current_user.id, strategy, symbol) or {}
        return jsonify({'success': True, 'strategy': strat_to_public(strategy),
                        'symbol': symbol,
                        'journey': evolution_engine.win_rate_journey(
                            current_user.id, strategy, symbol, state)})

    pairs = {(t.get('strategy'), t.get('symbol')) for t in
             db_manager.v2_get_user_trades(user_id=current_user.id, limit=5000)
             if t.get('strategy') and t.get('symbol')}
    out = []
    for st, sy in sorted(pairs):
        state = db_manager.v2_get_evolution_state(current_user.id, st, sy) or {}
        out.append({'strategy': strat_to_public(st),
                    'strategy_name': public_meta(st)['name'],
                    'symbol': sy,
                    'generation': state.get('generation', 0),
                    'journey': evolution_engine.win_rate_journey(
                        current_user.id, st, sy, state)})
    return jsonify({'success': True, 'journeys': out})


@v2_bp.route('/api/v2/evolution/run', methods=['POST'])
@login_required
def v2_evolution_run():
    """Manually trigger an evolution pass (normally automatic on each close)."""
    from v2.engine.evolution.evolution_engine import evolution_engine
    data = request.json or {}
    # Clients send the public code; the engine keys on the internal id.
    strategy = strat_to_internal(data.get('strategy'), default=None) \
        if data.get('strategy') else None
    force = bool(data.get('force', False))
    try:
        if strategy:
            results = [evolution_engine.evolve(current_user.id, strategy,
                                               data.get('symbol', 'ALL'), force=force)]
        else:
            # Bulk run covers running bots as well as pairs with history, so a
            # newly started bot is included instead of silently skipped.
            pairs = {(t.get('strategy'), t.get('symbol')) for t in
                     db_manager.v2_get_user_trades(user_id=current_user.id, limit=5000)
                     if t.get('strategy') and t.get('symbol')}
            try:
                pairs |= {(b.get('strategy'), b.get('symbol'))
                          for b in bot_manager_v2.get_all_bots(user_id=current_user.id)
                          if b.get('strategy') and b.get('symbol')}
            except Exception as e:
                logger.warning(f"[V2-EVO] could not fold bots into run: {e}")
            results = [evolution_engine.evolve(current_user.id, st, sy, force=force)
                       for st, sy in sorted(pairs)]
        return jsonify({'success': True, 'results': results})
    except Exception as e:
        logger.error(f"[V2-EVO] manual run failed: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


@v2_bp.route('/api/v2/account', methods=['GET'])
@login_required
def v2_account():
    """Get V2 account info with margin tracking and trade count from DB."""
    info = v2_paper_trader.get_account_info(current_user.id)
    
    # Fetch total trade count from DB (persisted) instead of just in-memory simulation history
    total_trades = db_manager.v2_get_total_trade_count(user_id=current_user.id)
    
    # Frontend expects 'total_value' and 'buying_power' — alias from V2 fields
    info.setdefault('total_value', info.get('equity', 100000))
    info.setdefault('buying_power', info.get('available_margin', 100000))
    info.setdefault('pnl', info.get('total_pnl', 0))
    info['total_trades'] = total_trades
    
    return jsonify({'success': True, **info})


@v2_bp.route('/api/v2/reports/strategy-benchmark', methods=['GET'])
@login_required
def v2_strategy_benchmark():
    """Get per-strategy metrics report with live fallback and session filtering."""
    strategy = _internal_strategy_filter(request.args.get('strategy'))
    session_id = request.args.get('session_id')

    # If session filtering is requested, we MUST compute live metrics
    # as cached metrics in v2_strategy_metrics are all-time aggregates.
    if session_id:
        try:
            metrics = _compute_live_metrics(user_id=current_user.id, strategy_filter=strategy, session_id=session_id)
            return jsonify({'success': True, 'metrics': _publicise_rows(metrics)})
        except Exception as e:
            logger.error(f"[V2] Session metrics error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    metrics = db_manager.v2_get_strategy_metrics(current_user.id, strategy)

    # ── Fallback: compute live metrics from v2_trade_ledger if metrics are empty/zero ──
    has_real_data = False
    if metrics:
        for m in metrics:
            if m.get('total_trades', 0) > 0:
                has_real_data = True
                break

    if not has_real_data:
        try:
            live = _compute_live_metrics(user_id=current_user.id, strategy_filter=strategy)
            if live:
                metrics = live
        except Exception as e:
            logger.error(f"[V2] Live metrics fallback error: {e}")

    return jsonify({'success': True, 'metrics': _publicise_rows(metrics)})


def _compute_live_metrics(user_id, strategy_filter=None, session_id=None):
    """
    Compute strategy metrics live from the v2_trade_ledger table.
    Used as fallback when v2_strategy_metrics is empty/zero.
    Delegates to StrategyAnalytics.compute_metrics() for accurate institutional metrics.
    """
    from v2.engine.analytics.strategy_analytics import StrategyAnalytics
    analytics = StrategyAnalytics()

    trades = db_manager.v2_get_user_trades(
        user_id=user_id, 
        strategy=strategy_filter, 
        session_id=session_id,
        limit=10000
    )
    if not trades:
        return []

    # Group by strategy
    from collections import defaultdict
    grouped = defaultdict(list)
    for t in trades:
        strat = t.get('strategy') or 'unknown'
        grouped[strat].append(t)

    results = []
    # Institutional Action mapping for metrics
    from shared.logic.trade_actions import CLOSING_ACTIONS

    for strat, strat_trades in grouped.items():
        close_trades = [t for t in strat_trades if t.get('action') in CLOSING_ACTIONS]
        if not close_trades:
            metrics = analytics._empty_metrics()
        else:
            metrics = analytics.compute_metrics(close_trades)
        metrics['strategy'] = strat
        results.append(metrics)

    return results


@v2_bp.route('/api/v2/reports/strategy-ranking', methods=['GET'])
@login_required
def v2_strategy_ranking():
    """Get composite-scored strategy ranking with optional session filtering."""
    session_id = request.args.get('session_id')
    
    if session_id:
        # Compute ranking metrics for specific session
        all_metrics = _compute_live_metrics(user_id=current_user.id, session_id=session_id)
    else:
        all_metrics = db_manager.v2_get_strategy_metrics(current_user.id)
        
    ranked = v2_ranker.rank(all_metrics)
    return jsonify({'success': True, 'ranking': ranked})


@v2_bp.route('/api/v2/portfolio/allocation', methods=['GET'])
@login_required
def v2_portfolio_allocation():
    """Get capital allocation across strategies."""
    metrics = db_manager.v2_get_strategy_metrics(current_user.id)
    profiles_raw = db_manager.v2_get_strategy_profile()
    profiles = {p['strategy']: p for p in profiles_raw} if profiles_raw else {}
    account = v2_paper_trader.get_account_info(current_user.id)
    total_capital = account.get('equity', 100000)

    allocations = v2_allocator.allocate(metrics, total_capital, profiles)
    summary = v2_allocator.get_allocation_summary(allocations, total_capital)
    return jsonify({'success': True, **summary})


@v2_bp.route('/api/v2/strategy-profiles', methods=['GET'])
@login_required
def v2_get_strategy_profiles():
    """List V2 strategy profiles."""
    profiles = db_manager.v2_get_strategy_profile()
    return jsonify({'success': True, 'profiles': profiles})


@v2_bp.route('/api/v2/strategy-profiles/<strategy>', methods=['PUT'])
@login_required
def v2_update_strategy_profile(strategy):
    """Update a V2 strategy profile."""
    try:
        data = request.json
        db_manager.v2_upsert_strategy_profile(strategy, data)
        return jsonify({'success': True, 'message': f'Profile {strategy} updated'})
    except Exception as e:
        logger.error(f"[V2] Strategy profile update error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


# ============================================================
# V2 LIVE PRICE PROXY
# ============================================================

@v2_bp.route('/api/v2/price/<symbol>', methods=['GET'])
def v2_price_proxy(symbol):
    """
    Live price for one symbol, served from the shared quote cache.

    This route used to run its own Binance request, independent of the one the
    socket streamer ran, so the header and Market Watch could show two prices for
    the same coin seconds apart. Both now read `quote_service`, which fetches once
    per TTL window and hands every caller the identical tick.
    """
    return jsonify(quote_service.get(symbol.upper()))


@v2_bp.route('/api/v2/prices', methods=['GET'])
def v2_prices_batch():
    """
    Live prices for many symbols in one request: /api/v2/prices?symbols=BTCUSDT,ETHUSDT

    The frontend renders the header, Market Watch and the watchlist from a single
    response, so every panel on the page is painted from the same snapshot rather
    than from N separate requests that each landed at a different instant.
    """
    raw = request.args.get('symbols', '')
    symbols = [s.strip().upper() for s in raw.split(',') if s.strip()]

    if not symbols:
        return jsonify({'success': False, 'error': 'No symbols requested', 'prices': {}}), 400
    if len(symbols) > 50:
        symbols = symbols[:50]

    ticks = quote_service.get_many(symbols)
    return jsonify({
        'success': True,
        'prices': {t['symbol']: t for t in ticks},
        'ts': time.time(),
    })


def clean_nan(val):
    """NaN/Inf -> JSON null. A NaN in a candle silently kills the whole chart."""
    try:
        if val is None or math.isnan(float(val)) or math.isinf(float(val)):
            return None
    except (TypeError, ValueError):
        return None
    return float(val)


@v2_bp.route('/api/v2/symbol/<symbol>', methods=['GET'])
def v2_resolve_symbol(symbol):
    """
    Resolve a ticker before anything tries to chart it.

    The frontend used to guess an exchange prefix from a hardcoded list and hand
    `BINANCE:MATICUSDT` to TradingView, which refused it because Binance had
    migrated the pair to POL - and the user got "This symbol doesn't exist" on a
    coin we could still price. This asks the exchange what it actually lists, and
    always names a way to draw the chart.
    """
    market_hint = request.args.get('market')
    # `tv` is ChartManager's exchange-prefix guess. We check it against TradingView's
    # own catalogue instead of trusting it - that check is what turns a dead panel
    # into an automatic fallback.
    tv_candidate = request.args.get('tv')
    return jsonify({
        'success': True,
        **symbol_registry.resolve(symbol, market_hint, tv_candidate)
    })


# Yahoo serves intraday crypto/equity candles at these granularities only; asking
# for anything else returns an empty frame, which is what a blank chart looks like.
_YAHOO_INTERVALS = {'1m', '5m', '15m', '30m', '1h', '1d'}


@v2_bp.route('/api/v2/candles/<symbol>', methods=['GET'])
def v2_candles(symbol):
    """
    OHLCV for any instrument, from whichever source answers first.

    This is what GoatBot draws when TradingView cannot serve a symbol. It walks the
    sources the registry named - Binance, then Yahoo - and only reports failure
    once every one of them has actually been tried and come back empty.
    """
    interval = request.args.get('interval', '1m')
    try:
        limit = max(10, min(int(request.args.get('limit', 300)), 1000))
    except ValueError:
        limit = 300

    info = symbol_registry.resolve(symbol, request.args.get('market'))
    target = info['symbol']
    attempted = []

    for source in info['data_sources']:
        try:
            if source == 'binance' and crypto_provider:
                attempted.append('binance')
                df = crypto_provider.get_historical_klines(
                    symbol=target, interval=interval, limit=limit)
            elif source == 'smartapi' and smartapi_provider:
                # Live NSE/BSE candles. Only reached for Indian instruments, and
                # only when credentials are configured.
                attempted.append('smartapi')
                df = smartapi_provider.get_historical_data(
                    symbol=target, interval=interval, limit=limit)
            elif source == 'coingecko' and coingecko_provider:
                # Crypto Binance has dropped. Granularity is CoinGecko's choice,
                # not ours - coarse candles beat an empty chart.
                attempted.append('coingecko')
                df = coingecko_provider.get_historical_data(
                    symbol=target, interval=interval, limit=limit)
            elif source == 'yahoo' and stock_provider:
                attempted.append('yahoo')
                yf_interval = interval if interval in _YAHOO_INTERVALS else '1d'
                # An index has its own Yahoo ticker (^NSEI), which is neither the
                # bare name nor the ".NS" equity form.
                yahoo_target = info.get('yahoo_symbol') or target
                df = stock_provider.get_historical_data(
                    symbol=yahoo_target, interval=yf_interval, limit=limit)
            else:
                continue
        except Exception as e:
            logger.warning(f"[V2] candles: {source} failed for {target}: {e}")
            continue

        if df is None or df.empty:
            continue

        candles = [{
            'time': int(idx.timestamp()),
            'open': clean_nan(row['open']),
            'high': clean_nan(row['high']),
            'low': clean_nan(row['low']),
            'close': clean_nan(row['close']),
            'volume': clean_nan(row.get('volume', 0)),
        } for idx, row in df.iterrows()]

        return jsonify({
            'success': True,
            'symbol': target,
            'requested': info['requested'],
            'renamed_from': info['renamed_from'],
            'note': info['note'],
            'interval': interval,
            'source': source,
            'currency': info.get('currency') or ('USD' if info['market'] == 'crypto' else None),
            'candles': candles,
        })

    logger.warning(f"[V2] candles: no source served {target} (tried {attempted})")
    return jsonify({
        'success': False,
        'symbol': target,
        'requested': info['requested'],
        'interval': interval,
        'attempted': attempted,
        'candles': [],
        'error': f'No candle data for {target} from any source',
    })


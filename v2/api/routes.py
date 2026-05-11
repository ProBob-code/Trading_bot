"""
V2 API Blueprint — Institutional Trading Engine
=================================================

All V2-specific routes, singletons, and bot execution logic.
Registered as a Flask Blueprint into the main api_server.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from datetime import datetime, date
import threading
import time

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
from v2.engine.core.risk_engine import RiskEngineV2
from v2.engine.core.portfolio_engine import PortfolioEngineV2
from v2.engine.core.pipeline import TradingPipelineV2
from shared.logic.strategies.v2_strategies import REGISTRY, atr as compute_atr, compute_smart_entry, compute_atr_position_size

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

# ---------------------------------------------------------------------------
# Module-level references (injected by init_v2)
# ---------------------------------------------------------------------------
socketio = None
strategy_engine = None
db_manager = None
crypto_provider = None
stock_provider = None
system_state_fn = None  # get_system_state callable


def init_v2(
    _socketio, _strategy_engine, _db_manager,
    _crypto_provider, _stock_provider, _system_state_fn
):
    """Inject shared dependencies from api_server.py."""
    global socketio, strategy_engine, db_manager
    global crypto_provider, stock_provider, system_state_fn
    global v2_analytics

    socketio = _socketio
    strategy_engine = _strategy_engine
    db_manager = _db_manager
    crypto_provider = _crypto_provider
    stock_provider = _stock_provider
    system_state_fn = _system_state_fn

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
    """Return list of available V2 strategies."""
    # Return registry without logic functions
    strategies = []
    for s in REGISTRY:
        s_copy = s.copy()
        s_copy.pop('logic', None)
        strategies.append(s_copy)
    return jsonify({'success': True, 'strategies': strategies})


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
        strategy = data.get('strategy', 'manual')
        volatility = float(data.get('volatility', 0.02))
        volume = float(data.get('volume', 100_000_000))
        margin_mode = data.get('margin_mode', 'isolated')

        if quantity <= 0:
            return jsonify({'success': False, 'error': 'Quantity must be > 0'}), 400

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
            return jsonify(results[0])
        
        return jsonify({'success': False, 'error': 'Trade failed', 'results': results})
    except Exception as e:
        logger.error(f"[V2] Trade error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500


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

            # ── New Candle Gate ──
            current_candle_time = df.index[-1]
            last_time = last_candle_times.get(symbol)
            if last_time is not None and current_candle_time <= last_time:
                # Same or old candle, wait for next check
                time.sleep(5)
                continue
                
            last_candle_times[symbol] = current_candle_time

            current_price = price_data.get('price', 0)
            if current_price <= 0:
                time.sleep(5)
                continue

            # Update V2 paper trader prices (for unrealized P&L and liquidation checks)
            user_id = config.user_id
            v2_paper_trader.set_prices({symbol: current_price})

            # ── Signal Generation ──
            atr_series = compute_atr(df, period=14)
            atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0
            
            logger.debug(f"🔍 [V2-BOT-{bot_id}] Analyzing {symbol} with {strategy_name}")
            signal = strategy_engine.analyze(df, strategy=strategy_name)

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
                'strategy': strategy_name,
                'reasons': getattr(signal, 'reasons', [])[:3],
                'bot_id': bot_id,
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
        result = bot_manager_v2.start_bot(
            user_id=current_user.id,
            symbol=data.get('symbol', ''),
            market=data.get('market', 'crypto'),
            strategy=data.get('strategy', 'combined'),
            mode=data.get('mode', 'paper'),
            interval=data.get('interval', '1m'),
            position_size=float(data.get('position_size', 10.0)),
            stop_loss=float(data.get('stop_loss', 5.0)),
            take_profit=float(data.get('take_profit', 10.0)),
            max_quantity=float(data.get('max_quantity', 1.0)),
            leverage=float(data.get('leverage', 1.0)),
            risk_pct=float(data.get('risk_pct', 2.0)),
        )

        # Start the execution thread if bot was registered successfully
        if result.get('success'):
            # Ensure a session is active
            active_session = db_manager.v2_get_active_session_id()
            if not active_session:
                v2_start_session()
            
            v2_start_bot_thread(result['bot_id'])

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
        bot_id = data.get('bot_id', '')
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
    return jsonify({'success': True, 'bots': bots})


@v2_bp.route('/api/v2/positions', methods=['GET'])
@login_required
def v2_positions():
    """Get V2 positions with leverage and margin data."""
    positions = v2_paper_trader.get_positions(current_user.id)
    return jsonify({'success': True, 'positions': positions})


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
    strategy = request.args.get('strategy')
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
        'trades': trades
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
    strategy = request.args.get('strategy')
    session_id = request.args.get('session_id')

    # If session filtering is requested, we MUST compute live metrics
    # as cached metrics in v2_strategy_metrics are all-time aggregates.
    if session_id:
        try:
            metrics = _compute_live_metrics(user_id=current_user.id, strategy_filter=strategy, session_id=session_id)
            return jsonify({'success': True, 'metrics': metrics})
        except Exception as e:
            logger.error(f"[V2] Session metrics error: {e}")
            return jsonify({'success': False, 'error': str(e)}), 500

    metrics = db_manager.v2_get_strategy_metrics(current_user.id, strategy)

    # ── Fallback: compute live metrics from v2_trades if metrics are empty/zero ──
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

    return jsonify({'success': True, 'metrics': metrics})


def _compute_live_metrics(user_id, strategy_filter=None, session_id=None):
    """
    Compute strategy metrics live from the v2_trades table.
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
    CLOSING_ACTIONS = ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')

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
    Fetch live price data for a symbol.
    - Crypto (USDT pairs): proxies Binance /api/v3/ticker/24hr
    - Stocks/Forex/Commodities: returns cached data from providers if available
    """
    import requests as http_requests

    symbol = symbol.upper()

    # Crypto: Binance API (no key needed for public ticker)
    if symbol.endswith('USDT') or symbol.endswith('BUSD') or symbol.endswith('USDC'):
        try:
            resp = http_requests.get(
                f'https://api.binance.com/api/v3/ticker/24hr',
                params={'symbol': symbol},
                timeout=5
            )
            if resp.status_code == 200:
                data = resp.json()
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'price': float(data.get('lastPrice', 0)),
                    'change_pct': float(data.get('priceChangePercent', 0)),
                    'high_24h': float(data.get('highPrice', 0)),
                    'low_24h': float(data.get('lowPrice', 0)),
                    'volume_24h': float(data.get('quoteVolume', 0)),
                    'open': float(data.get('openPrice', 0)),
                    'source': 'binance'
                })
            else:
                logger.warning(f"[V2] Binance API returned {resp.status_code} for {symbol}")
        except Exception as e:
            logger.error(f"[V2] Binance price fetch error for {symbol}: {e}")

    # Non-crypto: try our internal providers
    try:
        if crypto_provider and symbol.endswith('USDT'):
            price_data = crypto_provider.get_current_price(symbol)
            if price_data:
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'price': price_data.get('price', 0),
                    'change_pct': price_data.get('change_pct', 0),
                    'high_24h': price_data.get('high_24h', 0),
                    'low_24h': price_data.get('low_24h', 0),
                    'volume_24h': price_data.get('volume_24h', 0),
                    'source': 'internal_crypto'
                })

        if stock_provider:
            quote = stock_provider.get_current_quote(symbol)
            if quote and quote.get('price'):
                return jsonify({
                    'success': True,
                    'symbol': symbol,
                    'price': quote.get('price', 0),
                    'change_pct': quote.get('change_pct', 0),
                    'high_24h': quote.get('high', 0),
                    'low_24h': quote.get('low', 0),
                    'volume_24h': quote.get('volume', 0),
                    'source': 'internal_stock'
                })
    except Exception as e:
        logger.warning(f"[V2] Internal price provider error for {symbol}: {e}")

    # Fallback: no data available
    return jsonify({
        'success': False,
        'symbol': symbol,
        'price': 0,
        'change_pct': 0,
        'error': f'No price data available for {symbol}'
    })

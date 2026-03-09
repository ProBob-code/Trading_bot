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
from src.v2.execution.execution_engine import ExecutionEngine
from src.v2.execution.paper_trader_v2 import PaperTraderV2
from src.v2.risk.margin_engine import MarginEngine
from src.v2.analytics.strategy_analytics import StrategyAnalytics
from src.v2.analytics.monte_carlo import MonteCarloSimulator
from src.v2.portfolio.allocator import CapitalAllocator
from src.v2.portfolio.ranking_engine import StrategyRanker
from src.v2.intelligence.regime_detector import RegimeDetector
from src.v2.intelligence.volatility_filter import VolatilityFilter
from src.v2.bot_manager_v2 import BotManagerV2, bot_manager_v2
from src.strategies.v2_strategies import REGISTRY, atr as compute_atr, compute_smart_entry, compute_atr_position_size

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

    # Sync strategy profiles
    try:
        db_manager.v2_sync_strategy_profiles(REGISTRY)
    except Exception as e:
        logger.error(f"[V2] Failed to sync strategy profiles: {e}")

    logger.info("[V2] Blueprint initialised")


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

        result = v2_paper_trader.execute_trade(
            user_id=current_user.id,
            symbol=symbol,
            side=side,
            quantity=quantity,
            leverage=leverage,
            volatility=volatility,
            volume=volume,
            strategy=strategy,
            margin_mode=margin_mode,
        )

        if result.get('success'):
            # Persist to V2 trade ledger
            trade_record = {
                **result,
                'user_id': current_user.id,
                'trade_type': result.get('type', 'UNKNOWN'),
                'date': date.today().isoformat(),
            }
            try:
                db_manager.v2_save_trade(trade_record)
            except Exception as e:
                logger.error(f"[V2] Failed to save trade to DB: {e}")

            # Persist position state AFTER trade insert (correct ordering)
            try:
                if result.get('type') == 'CLOSE':
                    db_manager.v2_delete_position(current_user.id, symbol)
                else:
                    v2_paper_trader.save_positions(current_user.id, db_manager)
            except Exception as e:
                logger.error(f"[V2] Failed to persist position: {e}")

            # If this was a close, update strategy analytics
            if result.get('type') == 'CLOSE' and strategy != 'manual':
                acc_info = v2_paper_trader.get_account_info(current_user.id)
                v2_analytics.update_after_trade(
                    user_id=current_user.id,
                    strategy=strategy,
                    pnl=result.get('realized_pnl', 0),
                    account_value=acc_info.get('equity', 100000),
                )

        return jsonify(result)
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

            # Update V2 paper trader prices (for unrealized P&L and liquidation checks)
            user_id = config.user_id
            v2_paper_trader.set_prices({symbol: current_price})

            # Get V2 positions for this symbol
            v2_positions = v2_paper_trader.get_positions(user_id)
            symbol_pos = next((p for p in v2_positions if p['symbol'] == symbol), None)

            # Update bot stats
            bot.stats.unrealized_pnl = symbol_pos.get('unrealized_pnl', 0) if symbol_pos else 0
            bot.stats.total_pnl = bot.stats.realized_pnl + bot.stats.unrealized_pnl

            # ── Order Book Intelligence ──
            order_book = None
            if config.market == 'crypto':
                try:
                    order_book = crypto_provider.get_order_book(symbol)
                except Exception as ob_err:
                    logger.debug(f"[V2-BOT-{bot_id}] Order book fetch failed: {ob_err}")

            # ── Compute ATR for smart execution ──
            atr_series = compute_atr(df, period=14)
            atr_value = float(atr_series.iloc[-1]) if not atr_series.empty else 0.0

            # Analyze with strategy engine (passes order_book via kwargs)
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

            # Execute trade logic (with ATR + order book context)
            v2_execute_bot_trade(bot, signal, current_price, symbol_pos,
                                atr_value=atr_value, order_book=order_book)

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
    # Skip weak signals (score below 0.6) for new entries
    if signal.signal in ('BUY', 'SELL') and not has_position:
        if signal_score < 0.6:
            logger.debug(f"⏸️ [V2-BOT-{bot.bot_id}] Signal too weak ({signal_score:.2f} < 0.6) — skipping")
            return

    # Signal-based trading
    if signal.signal == 'BUY':
        if has_position and pos_side == 'SHORT':
            # Close short first
            logger.info(f"🔄 [V2-BOT-{bot.bot_id}] BUY signal → closing SHORT")
            _v2_place_trade(bot, 'BUY', abs(symbol_pos['quantity']), current_price, leverage, 'CLOSE_SHORT')
        elif not has_position or pos_side != 'LONG':
            # ── ATR-based Position Sizing ──
            account = v2_paper_trader.get_account_info(user_id)
            equity = account.get('equity', account.get('available_margin', 0))

            if atr_value > 0:
                # Smart sizing: risk 1% per trade, stop at 2×ATR
                atr_qty = compute_atr_position_size(equity, atr_value, risk_pct=0.01)
                # Cap to config max and margin-based max
                margin_qty = (equity * (config.position_size / 100)) / current_price if current_price > 0 else 0
                quantity = min(atr_qty, margin_qty, config.max_quantity)
            else:
                # Fallback to margin-based sizing
                buying_power = account.get('available_margin', 0)
                trade_value = buying_power * (config.position_size / 100)
                quantity = min(trade_value / current_price, config.max_quantity) if current_price > 0 else 0

            if quantity > 0:
                logger.info(f"🚀 [V2-BOT-{bot.bot_id}] Opening LONG {quantity:.6f} {symbol} @ {current_price} "
                            f"(Equity=${equity:.2f}, ATR={atr_value:.2f})")
                _v2_place_trade(bot, 'BUY', quantity, current_price, leverage, 'OPEN_LONG')
            else:
                logger.warning(f"⚠️ [V2-BOT-{bot.bot_id}] BUY skipped — zero quantity "
                               f"(Equity=${equity:.2f}, price={current_price})")
        else:
            logger.debug(f"⏸️ [V2-BOT-{bot.bot_id}] BUY skipped — already LONG {symbol}")

    elif signal.signal == 'SELL':
        if has_position and pos_side == 'LONG':
            # Close long first
            logger.info(f"🔄 [V2-BOT-{bot.bot_id}] SELL signal → closing LONG")
            _v2_place_trade(bot, 'SELL', abs(symbol_pos['quantity']), current_price, leverage, 'CLOSE_LONG')
        elif not has_position or pos_side != 'SHORT':
            # ── ATR-based Position Sizing ──
            account = v2_paper_trader.get_account_info(user_id)
            equity = account.get('equity', account.get('available_margin', 0))

            if atr_value > 0:
                atr_qty = compute_atr_position_size(equity, atr_value, risk_pct=0.01)
                margin_qty = (equity * (config.position_size / 100)) / current_price if current_price > 0 else 0
                quantity = min(atr_qty, margin_qty, config.max_quantity)
            else:
                buying_power = account.get('available_margin', 0)
                trade_value = buying_power * (config.position_size / 100)
                quantity = min(trade_value / current_price, config.max_quantity) if current_price > 0 else 0

            if quantity > 0:
                logger.info(f"🚀 [V2-BOT-{bot.bot_id}] Opening SHORT {quantity:.6f} {symbol} @ {current_price} "
                            f"(Equity=${equity:.2f}, ATR={atr_value:.2f})")
                _v2_place_trade(bot, 'SELL', quantity, current_price, leverage, 'OPEN_SHORT')
            else:
                logger.warning(f"⚠️ [V2-BOT-{bot.bot_id}] SELL skipped — zero quantity "
                               f"(Equity=${equity:.2f}, price={current_price})")
        else:
            logger.debug(f"⏸️ [V2-BOT-{bot.bot_id}] SELL skipped — already SHORT {symbol}")


def _v2_place_trade(bot, side, quantity, current_price, leverage, trade_type):
    """Execute a V2 trade through the institutional execution engine and persist."""
    config = bot.config
    user_id = config.user_id
    strategy = config.strategy
    symbol = config.symbol

    result = v2_paper_trader.execute_trade(
        user_id=user_id,
        symbol=symbol,
        side=side,
        quantity=quantity,
        leverage=leverage,
        strategy=strategy,
    )

    if result.get('success'):
        # Update bot stats
        # Use net_pnl to ensure commissions are deducted from bot card display
        pnl = result.get('net_pnl', result.get('realized_pnl', 0)) or 0
        bot_manager_v2.increment_trades(bot.bot_id, side, pnl)

        # Persist to v2_trades table
        trade_record = {
            **result,
            'user_id': user_id,
            'trade_type': result.get('type', trade_type),
            'date': date.today().isoformat(),
        }
        try:
            db_manager.v2_save_trade(trade_record)
        except Exception as e:
            logger.error(f"[V2-BOT-{bot.bot_id}] Failed to save trade: {e}")

        # Persist position state AFTER trade insert (correct ordering)
        try:
            if result.get('type') == 'CLOSE':
                db_manager.v2_delete_position(user_id, symbol)
            else:
                v2_paper_trader.save_positions(user_id, db_manager)
        except Exception as e:
            logger.error(f"[V2-BOT-{bot.bot_id}] Failed to persist position: {e}")

        # Update strategy analytics on close trades
        if result.get('type') == 'CLOSE' and strategy != 'manual':
            acc_info = v2_paper_trader.get_account_info(user_id)
            v2_analytics.update_after_trade(
                user_id=user_id,
                strategy=strategy,
                pnl=pnl,
                account_value=acc_info.get('equity', 100000),
            )

        # Emit V2 trade event
        socketio.emit('v2_trade_executed', {
            'type': 'trade',
            'side': side,
            'symbol': symbol,
            'quantity': quantity,
            'price': result.get('fill_price', current_price),
            'pnl': pnl,
            'strategy': strategy,
            'bot_id': bot.bot_id,
            'timestamp': datetime.now().isoformat(),
            'engine': 'v2'
        }, room=f"user_{user_id}")

        logger.info(f"✅ [V2-BOT-{bot.bot_id}] Trade executed: {side} {quantity:.6f} {symbol} @ {result.get('fill_price', current_price):.2f} (P&L: {pnl:.2f})")
    else:
        logger.warning(f"⚠️ [V2-BOT-{bot.bot_id}] Trade failed: {result.get('error', 'Unknown')}")


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


@v2_bp.route('/api/v2/trades', methods=['GET'])
@login_required
def v2_trade_history():
    """Get V2 trade history with optional filters."""
    strategy = request.args.get('strategy')
    limit = int(request.args.get('limit', 200))
    trades = db_manager.v2_get_user_trades(
        user_id=current_user.id,
        strategy=strategy,
        limit=limit
    )
    # Serialize datetime objects
    for t in trades:
        for k, v in t.items():
            if hasattr(v, 'isoformat'):
                t[k] = v.isoformat()
    return jsonify({'success': True, 'trades': trades})


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
    """Get V2 account info with margin tracking."""
    info = v2_paper_trader.get_account_info(current_user.id)
    # Frontend expects 'total_value' and 'buying_power' — alias from V2 fields
    info.setdefault('total_value', info.get('equity', 100000))
    info.setdefault('buying_power', info.get('available_margin', 100000))
    info.setdefault('pnl', info.get('total_pnl', 0))
    return jsonify({'success': True, **info})


@v2_bp.route('/api/v2/reports/strategy-benchmark', methods=['GET'])
@login_required
def v2_strategy_benchmark():
    """Get per-strategy metrics report with live fallback."""
    strategy = request.args.get('strategy')
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
            live = _compute_live_metrics(current_user.id, strategy)
            if live:
                metrics = live
        except Exception as e:
            logger.error(f"[V2] Live metrics fallback error: {e}")

    return jsonify({'success': True, 'metrics': metrics})


def _compute_live_metrics(user_id, strategy_filter=None):
    """
    Compute strategy metrics live from the v2_trades table.
    Used as fallback when v2_strategy_metrics is empty/zero.
    Delegates to StrategyAnalytics.compute_metrics() for accurate institutional metrics.
    """
    from src.v2.analytics.strategy_analytics import StrategyAnalytics
    analytics = StrategyAnalytics()

    trades = db_manager.v2_get_user_trades(user_id=user_id, strategy=strategy_filter, limit=10000)
    if not trades:
        return []

    # Group by strategy
    from collections import defaultdict
    grouped = defaultdict(list)
    for t in trades:
        strat = t.get('strategy') or 'unknown'
        grouped[strat].append(t)

    results = []
    for strat, strat_trades in grouped.items():
        close_trades = [t for t in strat_trades if t.get('trade_type') == 'CLOSE']
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
    """Get composite-scored strategy ranking."""
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

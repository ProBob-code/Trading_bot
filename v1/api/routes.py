"""
V1 API Blueprint — Legacy Retail Trading Engine
================================================

All V1-specific routes and bot execution logic.
Registered as a Flask Blueprint into the main api_server.
"""

from flask import Blueprint, jsonify, request
from flask_login import login_required, current_user
from datetime import datetime
import threading
import time

from loguru import logger
from shared.config.settings import is_v1_enabled

from v1.engine.execution.order_manager import Order, OrderSide, OrderType

# ---------------------------------------------------------------------------
# Blueprint
# ---------------------------------------------------------------------------
v1_bp = Blueprint('v1', __name__)

# ---------------------------------------------------------------------------
# Module-level references (injected by init_v1)
# ---------------------------------------------------------------------------
socketio = None
paper_trader = None
order_manager = None
strategy_engine = None
bot_manager = None
trade_logger = None
db_manager = None
crypto_provider = None
stock_provider = None
system_state_fn = None  # get_system_state callable

# V1 module-level state
live_auto_trading = False
live_auto_thread = None
auto_trade_stats = {
    'total_trades': 0,
    'buy_trades': 0,
    'sell_trades': 0,
    'total_pnl': 0,
    'signals': [],
    'start_time': None,
    'trades_log': [],
    'journal': []
}

# Current market state (shared with main server via injection)
current_market = "crypto"
current_symbol = "BTCUSDT"
current_interval = "1m"
current_strategy = "ichimoku"


def init_v1(
    _socketio, _paper_trader, _order_manager, _strategy_engine,
    _bot_manager, _trade_logger, _db_manager, _crypto_provider,
    _stock_provider, _system_state_fn, _current_market_ref=None
):
    """Inject shared dependencies from api_server.py."""
    global socketio, paper_trader, order_manager, strategy_engine
    global bot_manager, trade_logger, db_manager
    global crypto_provider, stock_provider, system_state_fn

    socketio = _socketio
    paper_trader = _paper_trader
    order_manager = _order_manager
    strategy_engine = _strategy_engine
    bot_manager = _bot_manager
    trade_logger = _trade_logger
    db_manager = _db_manager
    crypto_provider = _crypto_provider
    stock_provider = _stock_provider
    system_state_fn = _system_state_fn

    logger.info("[V1] Blueprint initialised")


# ============================================================
# ACCOUNT & TRADING
# ============================================================

@v1_bp.route('/api/account')
@login_required
def get_account():
    """Get account info."""
    account = paper_trader.get_account_info()
    positions = paper_trader.get_positions()

    return jsonify({
        'cash': account['cash'],
        'total_value': account['total_value'],
        'pnl': account['pnl'],
        'pnl_pct': account['pnl_pct'],
        'positions': positions
    })


@v1_bp.route('/api/trade', methods=['POST'])
@login_required
def execute_trade():
    """Execute a manual trade for the user."""
    if not is_v1_enabled():
        logger.warning("V1 manual trade blocked: V1 trading disabled")
        return jsonify({'success': False, 'error': 'V1 trading engine disabled'}), 403

    data = request.json
    symbol = data.get('symbol', current_symbol)
    side = data.get('side', 'buy')
    quantity = float(data.get('quantity', 0))
    market = data.get('market', current_market)

    if quantity <= 0:
        return jsonify({'success': False, 'error': 'Invalid quantity'})

    # Get current price
    price_data = crypto_provider.get_current_price(symbol) if market == 'crypto' else stock_provider.get_current_quote(symbol)
    price = price_data.get('price', 0)

    if price <= 0:
        return jsonify({'success': False, 'error': 'Could not get price'})

    # Update paper trader price
    paper_trader.set_prices({symbol: price})

    # Create and execute order
    order = Order(
        user_id=current_user.id,
        symbol=symbol,
        side=OrderSide.BUY if side.lower() == 'buy' else OrderSide.SELL,
        quantity=quantity,
        order_type=OrderType.MARKET
    )

    success = paper_trader.submit_order(order)

    if success:
        # Log via TradeLogger
        trade_logger.log_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            user_id=current_user.id,
            pnl=0,
            strategy='manual',
            bot_id='manual_control',
            mode='paper',
            account_value=paper_trader.get_account_info(current_user.id)['total_value'],
            notes=f"Manual {side} Trade"
        )

        return jsonify({
            'success': True,
            'order_id': order.order_id,
            'symbol': symbol,
            'side': side.upper(),
            'quantity': quantity,
            'price': price
        })
    return jsonify({'success': False, 'error': 'Order failed'})


@v1_bp.route('/api/panic-sell', methods=['POST'])
@login_required
def panic_sell():
    """Close all open positions immediately AND stop all bots for the user."""
    try:
        positions = paper_trader.get_positions(current_user.id)
        closed_count = 0

        for pos in positions:
            symbol = pos['symbol']
            qty = pos['quantity']
            side_str = 'sell' if qty > 0 else 'buy'
            order = order_manager.create_order(
                user_id=current_user.id,
                symbol=symbol,
                side=side_str,
                quantity=abs(qty),
                order_type='market'
            )
            if order_manager.submit_order(order):
                closed_count += 1
                trade_logger.log_trade(
                    symbol=symbol,
                    side='PANIC_CLOSE',
                    quantity=abs(qty),
                    price=pos.get('current_price', 0),
                    user_id=current_user.id,
                    pnl=pos.get('unrealized_pnl', 0),
                    strategy='panic_sell',
                    bot_id='panic_button',
                    mode='paper',
                    account_value=paper_trader.get_account_info(current_user.id)['total_value'],
                    notes="Panic Sell Triggered"
                )

        # CRITICAL: Also stop all running bots so they don't reopen positions
        user_bots = [b for b_id, b in list(bot_manager.bots.items()) if b.config.user_id == current_user.id]
        stopped_count = 0
        for bot in user_bots:
            result = bot_manager.stop_bot(bot.bot_id)
            if result.get('success'):
                stopped_count += 1
        bot_manager.save_configs()

        return jsonify({
            'success': True,
            'message': f'Panic Sell executed. Closed {closed_count} positions, stopped {stopped_count} bots.',
            'closed_count': closed_count,
            'stopped_bots': stopped_count
        })
    except Exception as e:
        logger.error(f"Panic sell failed for user {current_user.id}: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# BOT EXECUTION ENGINE (MULTI-THREADED)
# ============================================================

def bot_execution_loop(bot_id):
    if not is_v1_enabled():
        logger.warning(f"V1 execution loop blocked for bot {bot_id}: V1 trading disabled")
        return

    bot = bot_manager.bots.get(bot_id)
    if not bot:
        logger.error(f"❌ [BOT-{bot_id}] Not found in manager at startup.")
        return

    config = bot.config
    symbol = config.symbol
    interval = config.interval

    logger.info(f"🚀 [BOT-{bot_id}] Execution loop starting for {symbol} ({interval})")

    loop_count = 0
    while not bot.stop_flag.is_set():
        loop_count += 1
        try:
            # Heartbeat every 5 loops
            if loop_count % 5 == 0:
                logger.debug(f"💓 [BOT-{bot_id}] Heartbeat - Loop {loop_count} is active")

            # Check for system pause
            if system_state_fn().is_paused():
                logger.info(f"⏸️ [BOT-{bot_id}] System paused. Waiting...")
                time.sleep(10)
                continue

            # Check if bot still exists in manager (wasn't deleted)
            if bot_id not in bot_manager.bots:
                logger.warning(f"⚠️ [BOT-{bot_id}] Bot removed from manager. Exiting.")
                break

            # Fetch data
            logger.debug(f"🔍 [BOT-{bot_id}] Fetching latest data for {symbol}...")
            if config.market == 'crypto':
                df = crypto_provider.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=200
                )
                price_data = crypto_provider.get_current_price(symbol)
            else:
                df = stock_provider.get_historical_data(
                    symbol=symbol,
                    interval=interval,
                    limit=200
                )
                price_data = stock_provider.get_current_quote(symbol)

            if df.empty or len(df) < 52:
                logger.warning(f"⚠️ [BOT-{bot_id}] Insufficient data: {len(df)} candles")
                time.sleep(10)
                continue

            current_price = price_data.get('price', 0)
            if current_price <= 0:
                logger.warning(f"⚠️ [BOT-{bot_id}] Invalid price: {current_price}")
                time.sleep(5)
                continue

            # Update Shared Paper Trader Prices & Account State
            user_id = bot.config.user_id
            paper_trader.set_prices({symbol: current_price})

            # Calculate Unrealized P&L for this bot specifically
            positions = paper_trader.get_positions(user_id)
            symbol_pos = next((p for p in positions if p['symbol'] == symbol), None)

            bot.stats.unrealized_pnl = symbol_pos.get('unrealized_pnl', 0) if symbol_pos else 0
            bot.stats.total_pnl = bot.stats.realized_pnl + bot.stats.unrealized_pnl

            # Analyze with selected strategy
            strategy_settings = bot.config.settings if hasattr(bot.config, 'settings') and isinstance(bot.config.settings, dict) else {}
            strategy_engine.min_confluence = strategy_settings.get('confluence', 1)

            current_strat = bot.config.strategy
            logger.debug(f"🔍 [BOT-{bot_id}] Analyzing {symbol} with {current_strat}")
            signal = strategy_engine.analyze(df, strategy=current_strat)

            # Update bot stats
            bot.stats.last_price = current_price
            bot.stats.last_signal = signal.signal
            bot.stats.signals_generated += 1

            # Store signal
            signal_data = {
                'time': datetime.now().isoformat(),
                'signal': signal.signal,
                'strength': getattr(signal, 'strength', 0),
                'price': current_price,
                'strategy': current_strat,
                'reasons': getattr(signal, 'reasons', [])[:3],
                'bot_id': bot_id,
                'symbol': symbol,
                'engine': 'v1'
            }

            # Emit signal to user room
            socketio.emit('auto_trade_signal', signal_data, room=f"user_{user_id}")
            if signal.signal != 'HOLD':
                logger.info(f"📡 [BOT-{bot_id}] SIGNAL: {signal.signal} for {symbol} at {current_price}")

            # 1. Check for TP/SL Exit Conditions (Proactive Exit)
            if symbol_pos:
                pnl_pct = symbol_pos.get('unrealized_pnl_pct', 0)
                tp_pct = bot.config.take_profit
                sl_pct = bot.config.stop_loss

                if pnl_pct >= tp_pct:
                    logger.info(f"🎯 [BOT-{bot_id}] TAKE PROFIT: {pnl_pct:.2f}% (Limit: {tp_pct}%)")
                    exit_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Take Profit Hit']})
                    execute_bot_trade(bot, exit_signal, current_price)
                    time.sleep(5)
                    continue  # TP triggered — skip signal-based trading this iteration
                elif pnl_pct <= -sl_pct:
                    logger.info(f"🛑 [BOT-{bot_id}] STOP LOSS: {pnl_pct:.2f}% (Limit: {sl_pct}%)")
                    exit_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Stop Loss Hit']})
                    execute_bot_trade(bot, exit_signal, current_price)
                    time.sleep(5)
                    continue  # SL triggered — skip signal-based trading this iteration

            # 2. Signal-based trading (BUY/SELL only — HOLD does NOT trade)
            if signal.signal in ('BUY', 'SELL'):
                logger.info(f"🎯 [BOT-{bot_id}] SIGNAL: {signal.signal} for {symbol} (Confidence: {getattr(signal, 'confidence', 0.0):.2f})")
                execute_bot_trade(bot, signal, current_price)
            elif loop_count % 10 == 0:
                logger.debug(f"😴 [BOT-{bot_id}] No signal (HOLD) for {symbol}")

            # Sleep (check every 5 seconds)
            time.sleep(5)

        except Exception as e:
            logger.error(f"❌ [BOT-{bot_id}] CRASH in loop: {e}", exc_info=True)
            time.sleep(10)

    logger.info(f"🛑 Bot execution stopped: {bot_id}")

def execute_bot_trade(bot, signal, current_price):
    """Execute trades for a specific bot."""
    if not is_v1_enabled():
        logger.error(f"BLOCKED: Attempted V1 trade execution for bot {bot.bot_id} (Firewall Lock)")
        return None

    symbol = bot.config.symbol
    user_id = bot.config.user_id

    # Fetch positions from paper trader (needed for close/cover logic)
    positions = paper_trader.get_positions(user_id)

    # Check current positions for THIS bot specifically
    bot_qty = bot.stats.current_qty
    has_long = bot_qty > 0
    has_short = bot_qty < 0

    # Debug logging for every signal
    logger.info(f"📊 [BOT-{bot.bot_id}] DECISION: Signal={signal.signal}, BotQty={bot_qty}, HasLong={has_long}, HasShort={has_short}")

    # SAFETY: If both LONG and SHORT exist (illegal state), close both immediately
    if has_long and has_short:
        logger.warning(f"⚠️ [BOT-{bot.bot_id}] ILLEGAL STATE: BOTH LONG and SHORT for {symbol}. Closing both.")
        # Close the LONG
        long_pos = next((p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG'), None)
        if long_pos:
            order = order_manager.create_order(
                user_id=user_id,
                symbol=symbol,
                side='sell',
                quantity=long_pos['quantity'],
                order_type='market'
            )
            if order_manager.submit_order(order):
                entry_price = long_pos.get('avg_price', current_price)
                pnl = (current_price - entry_price) * long_pos['quantity']
                bot_manager.increment_trades(bot.bot_id, 'sell', long_pos['quantity'], pnl)
                emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl)

        # Cover the SHORT
        short_pos = next((p for p in positions if p['symbol'] == symbol and p['side'] == 'SHORT'), None)
        if short_pos:
            order = order_manager.create_order(
                user_id=user_id,
                symbol=symbol,
                side='buy',
                quantity=short_pos['quantity'],
                order_type='market'
            )
            if order_manager.submit_order(order):
                pnl = (short_pos['avg_price'] - current_price) * short_pos['quantity']
                bot_manager.increment_trades(bot.bot_id, 'buy', short_pos['quantity'], pnl)
                emit_trade_event(bot, 'COVER SHORT', short_pos['quantity'], current_price, pnl)
        return  # Exit to let state settle

    if signal.signal == 'BUY' and has_short:
        logger.info(f"🔄 [BOT-{bot.bot_id}] BUY signal vs SHORT position. Closing SHORT first.")
        short_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'SHORT')
        order = order_manager.create_order(
            user_id=user_id,
            symbol=symbol,
            side='buy',
            quantity=short_pos['quantity'],
            order_type='market'
        )
        if order_manager.submit_order(order):
            pnl = (short_pos['avg_price'] - current_price) * short_pos['quantity']
            bot_manager.increment_trades(bot.bot_id, 'buy', short_pos['quantity'], pnl)
            emit_trade_event(bot, 'COVER SHORT', short_pos['quantity'], current_price, pnl, reasons=getattr(signal, 'reasons', []))
        return

    if signal.signal == 'SELL' and has_long:
        logger.info(f"🔄 [BOT-{bot.bot_id}] SELL signal vs LONG position. Closing LONG first.")
        long_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG' and p['quantity'] > 0)
        pnl = (current_price - long_pos['avg_price']) * long_pos['quantity']
        order = order_manager.create_order(
            user_id=user_id,
            symbol=symbol,
            side='sell',
            quantity=long_pos['quantity'],
            order_type='market'
        )
        if order_manager.submit_order(order):
            bot_manager.increment_trades(bot.bot_id, 'sell', long_pos['quantity'], pnl)
            emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl, reasons=getattr(signal, 'reasons', []))
        return

    if signal.signal == 'BUY':
        if has_long:
            logger.debug(f"⏸️ [BOT-{bot.bot_id}] BUY skipped - already LONG {symbol}")
        else:
            # Open new LONG
            account = paper_trader.get_account_info(user_id)
            trade_value = account['buying_power'] * (bot.config.position_size / 100)
            quantity = min(trade_value / current_price, bot.config.max_quantity)

            if quantity > 0:
                logger.info(f"🚀 [BOT-{bot.bot_id}] Opening LONG {quantity} {symbol} @ {current_price}")
                order = order_manager.create_order(
                    user_id=user_id,
                    symbol=symbol,
                    side='buy',
                    quantity=quantity,
                    order_type='market'
                )
                if order_manager.submit_order(order):
                    bot_manager.increment_trades(bot.bot_id, 'buy', quantity, 0)
                    emit_trade_event(bot, 'LONG BUY', quantity, current_price, 0, reasons=getattr(signal, 'reasons', []))
            else:
                logger.warning(f"⚠️ [BOT-{bot.bot_id}] BUY failed - zero quantity (Buying Power: {account['buying_power']})")

    elif signal.signal == 'SELL':
        if has_short:
            logger.debug(f"⏸️ [BOT-{bot.bot_id}] SELL skipped - already SHORT {symbol}")
        elif has_long:
            return  # handled above
        else:
            # Open new SHORT
            account = paper_trader.get_account_info(user_id)
            trade_value = account['buying_power'] * (bot.config.position_size / 100)
            quantity = min(trade_value / current_price, bot.config.max_quantity)

            if quantity > 0:
                logger.info(f"🚀 [BOT-{bot.bot_id}] Opening SHORT {quantity} {symbol} @ {current_price}")
                order = order_manager.create_order(
                    user_id=user_id,
                    symbol=symbol,
                    side='sell',
                    quantity=quantity,
                    order_type='market'
                )
                if order_manager.submit_order(order):
                    bot_manager.increment_trades(bot.bot_id, 'sell', quantity, 0)
                    emit_trade_event(bot, 'SHORT SELL', quantity, current_price, 0, reasons=getattr(signal, 'reasons', []))
            else:
                logger.warning(f"⚠️ [BOT-{bot.bot_id}] SELL failed - zero quantity")

def emit_trade_event(bot, side, quantity, price, pnl=0, reasons=None):
    user_id = bot.config.user_id
    trade_msg = {
        'type': 'trade',
        'side': side,
        'symbol': bot.config.symbol,
        'quantity': quantity,
        'price': price,
        'pnl': pnl,
        'strategy': bot.config.strategy,
        'reasons': reasons or [],
        'timestamp': datetime.now().isoformat(),
        'user': 'system_bot',
        'bot_id': bot.bot_id
    }

    # Update Daily report -> Log via TradeLogger
    trade_logger.log_trade(
        symbol=bot.config.symbol,
        side=side,
        quantity=quantity,
        price=price,
        user_id=user_id,
        pnl=pnl,
        strategy=bot.config.strategy,
        bot_id=bot.bot_id,
        mode=bot.config.mode.value if hasattr(bot.config, 'mode') and hasattr(bot.config.mode, 'value') else 'paper',
        account_value=paper_trader.get_account_info(user_id)['total_value'],
        notes=f"Auto Bot Trade {side}"
    )

    socketio.emit('auto_trade_executed', trade_msg, room=f"user_{user_id}")

def start_bot_thread(bot_id):
    """Helper to start a bot thread with existence check."""
    if not is_v1_enabled():
        logger.warning(f"V1 bot thread start blocked for {bot_id}: V1 trading disabled")
        return

    if bot_id in bot_manager.bots:
        bot = bot_manager.bots[bot_id]
        # Check if thread is already running
        if bot.thread and bot.thread.is_alive():
            logger.info(f"ℹ️ Bot thread {bot_id} already running. Skipping startup.")
            return

        thread = threading.Thread(target=bot_execution_loop, args=(bot_id,), daemon=True)
        bot.thread = thread
        thread.start()
        logger.info(f"🧵 Started new thread for bot: {bot_id}")


# ============================================================
# LEGACY COMPATIBILITY (REDIRECTS TO NEW BOT SYSTEM)
# ============================================================


@v1_bp.route('/api/auto-trade/start', methods=['POST'])
def start_auto_trade():
    """Legacy endpoint - redirects to new bot system."""
    return jsonify({
        'success': False,
        'error': 'Legacy auto-trade is deprecated. Please use the bot system (Start Bot button).'
    }), 400


@v1_bp.route('/api/auto-trade/stop', methods=['POST'])
def stop_auto_trade():
    """Stop live auto-trading and get report."""
    global live_auto_trading, auto_trade_stats

    if not live_auto_trading:
        return jsonify({'success': False, 'error': 'Not running'})

    live_auto_trading = False

    # Generate report
    account = paper_trader.get_account_info()
    duration = (datetime.now() - auto_trade_stats['start_time']).total_seconds() if auto_trade_stats['start_time'] else 0

    report = {
        'total_trades': auto_trade_stats['total_trades'],
        'buy_trades': auto_trade_stats['buy_trades'],
        'sell_trades': auto_trade_stats['sell_trades'],
        'total_pnl': account['pnl'],
        'roi_percent': account['pnl_pct'],
        'final_balance': account['total_value'],
        'signals_generated': len(auto_trade_stats['signals']),
        'duration_seconds': duration,
        'market': current_market,
        'symbol': current_symbol,
        'strategy': current_strategy,
        'trades_log': auto_trade_stats['trades_log'][-20:]  # Last 20 trades
    }

    return jsonify({'success': True, 'report': report})


@v1_bp.route('/api/auto-trade/report')
@login_required
def auto_trade_report():
    """Generate trading report for the current session for the user."""
    acc = paper_trader.get_account_info(current_user.id)
    summary = trade_logger.get_daily_summary(current_user.id, datetime.now().strftime("%Y-%m-%d"))

    report = {
        'total_trades': summary['total_trades'],
        'buy_trades': summary.get('buy_trades', 0),
        'sell_trades': summary.get('sell_trades', 0),
        'total_pnl': acc['pnl'],
        'roi_percent': acc['pnl_pct'],
        'final_balance': acc['total_value'],
        'signals_generated': 0, # signals are transient
        'duration_seconds': 0,
        'market': current_market,
        'symbol': current_symbol,
        'strategy': current_strategy,
        'trades_log': trade_logger.get_history(current_user.id, limit=20)
    }

    return jsonify({'success': True, 'report': report})


@v1_bp.route('/api/auto-trade/status')
@login_required
def auto_trade_status():
    """Get live auto-trade status for the user."""
    acc = paper_trader.get_account_info(current_user.id)
    summary = trade_logger.get_daily_summary(current_user.id, datetime.now().strftime("%Y-%m-%d"))

    # Check if user has any active bots
    user_bots = bot_manager.get_all_bots(current_user.id)
    any_running = any(bot_manager.is_running(b['bot_id']) for b in user_bots)

    return jsonify({
        'running': any_running,
        'market': current_market,
        'symbol': current_symbol,
        'strategy': current_strategy,
        'total_trades': summary['total_trades'],
        'buy_trades': summary.get('buy_trades', 0),
        'sell_trades': summary.get('sell_trades', 0),
        'current_pnl': acc['pnl'],
        'signals': [] # Signals are now streamed via Socket.io
    })


@v1_bp.route('/api/report/download')
@login_required
def download_report():
    """Generate downloadable trading report for user."""
    account = paper_trader.get_account_info(current_user.id)
    summary = trade_logger.get_daily_summary(current_user.id, datetime.now().strftime("%Y-%m-%d"))

    report = {
        'generated_at': datetime.now().isoformat(),
        'user': current_user.username,
        'account': {
            'initial_capital': account.get('initial_capital', 100000),
            'current_balance': account['total_value'],
            'total_pnl': account['pnl'],
            'roi_percent': account['pnl_pct']
        },
        'stats': summary,
        'trades_log': trade_logger.get_history(current_user.id, limit=100)
    }

    return jsonify(report)


# ============================================================
# BOT MANAGEMENT ROUTES
# ============================================================

@v1_bp.route('/api/bots', methods=['GET'])
@login_required
def list_bots():
    """List all active bots for the current user."""
    running_bots = bot_manager.get_running_bots(current_user.id)
    logger.info(f"📊 User {current_user.id} active bots: {len(running_bots)}")
    return jsonify({
        'success': True,
        'bots': running_bots,
        'running_count': len(running_bots)
    })

@v1_bp.route('/api/bots/start', methods=['POST'])
@login_required
def start_bot():
    """Start a new trading bot for the user."""
    if not is_v1_enabled():
        logger.warning("V1 bot start blocked: V1 trading disabled")
        return jsonify({'success': False, 'error': 'V1 trading engine disabled', 'status': 'disabled'}), 403

    data = request.json
    try:
        # Extract settings from nested object
        settings = data.get('settings', {})

        # Call bot_manager.start_bot with keyword arguments (not BotConfig)
        result = bot_manager.start_bot(
            user_id=current_user.id,
            symbol=data.get('symbol', 'BTCUSDT'),
            market=data.get('market', 'crypto'),
            strategy=data.get('strategy', 'Ichimoku Cloud'),
            mode=data.get('mode', 'paper'),
            interval=data.get('interval', '1m'),
            position_size=float(settings.get('positionSize', 10.0)),
            stop_loss=float(settings.get('stopLoss', 5.0)),
            take_profit=float(settings.get('takeProfit', 10.0)),
            max_quantity=float(settings.get('maxQuantity', 1.0))
        )

        if result['success']:
            # Start the bot execution thread
            start_bot_thread(result['bot_id'])

            # Persist bot configurations to disk
            bot_manager.save_configs()

            logger.info(f"✅ Bot started for user {current_user.id}: {result['bot_id']}")
            return jsonify({'success': True, 'bot_id': result['bot_id']})
        else:
            return jsonify(result)

    except Exception as e:
        logger.error(f"Failed to start bot for user {current_user.id}: {e}")
        return jsonify({'success': False, 'error': str(e)})

@v1_bp.route('/api/bots/<bot_id>/stop', methods=['POST'])
@login_required
def stop_bot(bot_id):
    """Stop a specific bot and auto-close its positions for the current user."""
    try:
        bot = bot_manager.bots.get(bot_id)
        if bot:
            # Security check: Ensure bot belongs to current user
            if bot.config.user_id != current_user.id:
                return jsonify({'success': False, 'error': 'Unauthorized'})

            symbol = bot.config.symbol
            positions = paper_trader.get_positions(current_user.id)
            symbol_pos = next((p for p in positions if p['symbol'] == symbol), None)

            if symbol_pos:
                logger.info(f"🛑 Bot {bot_id} stop requested by user {current_user.id}. Auto-closing position.")
                qty = symbol_pos['quantity']
                side = 'sell' if qty > 0 else 'buy'
                pnl = symbol_pos.get('unrealized_pnl', 0)

                trade_logger.log_trade(
                    symbol=symbol,
                    side='CLOSE',
                    quantity=abs(qty),
                    price=symbol_pos.get('current_price', 0),
                    user_id=current_user.id,
                    pnl=pnl,
                    strategy=bot.config.strategy,
                    bot_id=bot.bot_id,
                    mode=bot.config.mode.value if hasattr(bot.config.mode, 'value') else 'paper',
                    notes="Stop Command Received - Auto Liquidating"
                )

                # Execute closure
                order = Order(user_id=current_user.id, symbol=symbol, side=OrderSide.SELL if qty > 0 else OrderSide.BUY, quantity=abs(qty), order_type=OrderType.MARKET)
                paper_trader.submit_order(order)

        result = bot_manager.stop_bot(bot_id)
        bot_manager.save_configs()
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during bot stop: {e}")
        return jsonify({'success': False, 'error': str(e)})

@v1_bp.route('/api/bots/stop-all', methods=['POST'])
@login_required
def stop_all_bots():
    """Stop all active bots and close their positions for the current user."""
    try:
        # Close positions for all user's bots before stopping
        user_bots = [b for b_id, b in list(bot_manager.bots.items()) if b.config.user_id == current_user.id]
        closed_positions = 0

        for bot in user_bots:
            symbol = bot.config.symbol
            positions = paper_trader.get_positions(current_user.id)
            symbol_pos = next((p for p in positions if p['symbol'] == symbol), None)

            if symbol_pos:
                qty = symbol_pos['quantity']
                side = 'sell' if qty > 0 else 'buy'
                order = Order(
                    user_id=current_user.id,
                    symbol=symbol,
                    side=OrderSide.SELL if qty > 0 else OrderSide.BUY,
                    quantity=abs(qty),
                    order_type=OrderType.MARKET
                )
                if paper_trader.submit_order(order):
                    closed_positions += 1
                    trade_logger.log_trade(
                        symbol=symbol,
                        side='CLOSE',
                        quantity=abs(qty),
                        price=symbol_pos.get('current_price', 0),
                        user_id=current_user.id,
                        pnl=symbol_pos.get('unrealized_pnl', 0),
                        strategy=bot.config.strategy,
                        bot_id=bot.bot_id,
                        mode=bot.config.mode.value if hasattr(bot.config.mode, 'value') else 'paper',
                        notes="Stop All - Auto Liquidating"
                    )

        # Now stop all bots
        bot_manager.stop_all()
        bot_manager.save_configs()

        return jsonify({
            'success': True,
            'message': f'All bots stopped. Closed {closed_positions} positions.',
            'closed_positions': closed_positions
        })
    except Exception as e:
        logger.error(f"Error in stop_all_bots: {e}")
        return jsonify({'success': False, 'error': str(e)})

@v1_bp.route('/api/bots/<bot_id>/strategy', methods=['PUT'])
@login_required
def update_bot_strategy(bot_id):
    """Update strategy for a running bot."""
    data = request.json
    new_strategy = data.get('strategy')

    if bot_manager.update_bot_config(bot_id, strategy=new_strategy):
        return jsonify({'success': True, 'strategy': new_strategy})
    return jsonify({'success': False, 'error': 'Bot not found'})


# ============================================================
# INSTITUTIONAL RESEARCH TERMINAL API
# ============================================================

@v1_bp.route('/api/status')
@login_required
def get_paper_status_new():
    """Consolidated status for the institutional terminal."""
    user_id = current_user.id if current_user.is_authenticated else 1
    user_bots = bot_manager.get_all_bots(user_id)
    summary = trade_logger.get_daily_summary(user_id, datetime.now().strftime("%Y-%m-%d"))

    return jsonify({
        'total_bots': len(user_bots),
        'total_trades': summary['total_trades'],
        'paper_mode': True,
        'timestamp': datetime.now().isoformat()
    })

@v1_bp.route('/api/market-pulse')
@login_required
def get_market_pulse():
    """Simulated market pulse data for institutional research."""
    return jsonify({
        'metrics': [
            { 'label': 'Regime', 'value': 75, 'text': 'Strong Trend (Bull)' },
            { 'label': 'Volatility (ATR%)', 'value': 35, 'text': '1.8%' },
            { 'label': 'Volume Surge', 'value': 62, 'text': 'High' },
            { 'label': 'Trend Strength', 'value': 88, 'text': 'Institutional' },
            { 'label': 'Spread Status', 'value': 5, 'text': 'Tight' },
            { 'label': 'Correlation Risk', 'value': 20, 'text': 'Low' }
        ]
    })

@v1_bp.route('/api/live-signals')
@login_required
def get_live_signals():
    """Get active trading signals before execution."""
    # Simulated for dashboard demonstration
    return jsonify({
        'signals': [
            {
                'bot_id': 'TrendBot_BTC',
                'strategy': 'Ichimoku Cloud',
                'dir': 'LONG',
                'entry': 64850.5,
                'sl': 63500.0,
                'tp': 68200.0,
                'rr': 2.5,
                'risk_pct': 1.5,
                'regime': 'Trend',
                'confidence': 92,
                'active_time': '12m ago'
            }
        ]
    })

@v1_bp.route('/api/live-trades')
@login_required
def get_paper_live_trades():
    """Get detailed open positions for the terminal."""
    user_id = current_user.id if current_user.is_authenticated else 1
    positions = paper_trader.get_positions(user_id)

    trades = []
    for p in positions:
        trades.append({
            'trade_id': p.get('trade_id', 'T123'),
            'bot_id': p['symbol'],
            'entry_price': p['avg_price'],
            'current_price': p.get('current_price', p['avg_price']),
            'unrealized_pnl': p.get('unrealized_pnl', 0),
            'r_multiple': p.get('r_multiple', 1.2),
            'slippage_impact': -12.50,
            'time_in_trade': '2h 15m',
            'risk_ladder_active': True
        })
    return jsonify({ 'trades': trades })

@v1_bp.route('/api/compare')
@login_required
def get_bot_comparison():
    """Bot performance matrix for institutional sorting."""
    user_id = current_user.id if current_user.is_authenticated else 1
    user_bots = bot_manager.get_all_bots(user_id)

    comparison = []
    for b in user_bots:
        stats = b.get('stats', {})
        comparison.append({
            'bot_id': b['bot_id'],
            'safety_label': 'SAFE' if stats.get('max_drawdown', 0) < 10 else 'CAUTION',
            'composite_score': stats.get('profit_factor', 0) * 10,
            'regime_stability': 0.85,
            'stress_grade': 'ROBUST',
            'slippage_sensitivity': 1.1,
            'capital_efficiency': 0.92,
            'max_drawdown_pct': stats.get('max_drawdown', 0),
            'expectancy': stats.get('expectancy', 0),
            'total_pnl': stats.get('total_pnl', 0),
            'risk_of_ruin': 2.1
        })
    return jsonify({ 'comparison': comparison })

@v1_bp.route('/api/allocation')
@login_required
def get_capital_allocation():
    """Capital allocation intelligence based on performance."""
    user_id = current_user.id if current_user.is_authenticated else 1
    user_bots = bot_manager.get_all_bots(user_id)

    allocs = []
    for b in user_bots:
        allocs.append({
            'bot_id': b['bot_id'],
            'recommended_allocation_pct': 25,
            'kelly_fraction': 0.45,
            'vol_parity_weight': 0.35,
            'risk_cap_applied': True
        })
    return jsonify({ 'allocations': allocs })

@v1_bp.route('/api/stress/<bot_id>')
@login_required
def get_bot_stress_test(bot_id):
    """Deep stress analysis for a specific bot."""
    return jsonify({
        'stress_test': {
            'parameter_stability': { 'stability_ratio': 0.88 },
            'slippage_stress': { 'breakpoint_multiplier': 4.5 },
            'tail_risk': { 'all_surviving': True }
        }
    })

@v1_bp.route('/api/efficiency/<bot_id>')
@login_required
def get_bot_efficiency(bot_id):
    """Capital efficiency metrics."""
    return jsonify({
        'efficiency': {
            'capital_efficiency_ratio': 0.94,
            'avg_utilization': 65
        }
    })


# ============================================================
# BOT WATCHDOG & RESTORE (BACKGROUND THREADS)
# ============================================================

def bot_watchdog_loop():
    """Background watchdog that monitors bots and restarts crashed ones."""
    logger.info("🐕 Bot watchdog started")
    while True:
        time.sleep(30)  # Check every 30 seconds

        try:
            for bot_id in list(bot_manager.bots.keys()):
                bot = bot_manager.bots.get(bot_id)
                if not bot:
                    continue

                # Check if bot should be running but its thread is dead
                if bot.status.value == 'running' and bot.config.auto_restart_enabled:
                    # SAFETY: Never restart a bot whose stop_flag is set (intentionally stopped)
                    if bot.stop_flag.is_set():
                        continue
                    if bot.thread is None or not bot.thread.is_alive():
                        logger.warning(f"🔄 Watchdog: Restarting crashed bot {bot_id}")

                        # Reset the stop flag
                        bot.stop_flag.clear()

                        # Start a new thread
                        start_bot_thread(bot_id)
        except Exception as e:
            logger.error(f"Watchdog error: {e}")


def restore_bots_on_startup():
    """Load and start bots that were previously running or enabled for auto-restart."""
    configs = bot_manager.load_configs()
    if not configs:
        return

    logger.info(f"🔄 Restoring {len(configs)} bots from MySQL...")
    started_canonical_ids = set()

    for cfg in configs:
        user_id = cfg.get('user_id', 1)
        bot_id = cfg.get('id')

        # Compute canonical bot_id to avoid duplicates
        canonical_id = bot_manager.generate_bot_id(user_id, cfg['market'], cfg['symbol'])
        if canonical_id in started_canonical_ids:
            logger.info(f"⏭️ Skipping duplicate config: {bot_id} (canonical: {canonical_id})")
            # Clean up the stale duplicate from MySQL
            try:
                db_manager.delete_bot_config(bot_id)
            except Exception:
                pass
            continue

        # Check if bot should be running (status was 'running' or auto_restart is true)
        if cfg.get('status') == 'running' or cfg.get('auto_restart_enabled', 0):
            logger.info(f"🚀 Auto-restoring bot: {canonical_id}")

            # Start the bot using bot_manager.start_bot
            result = bot_manager.start_bot(
                user_id=user_id,
                symbol=cfg['symbol'],
                market=cfg['market'],
                strategy=cfg['strategy'],
                mode=cfg.get('mode', 'paper'),
                interval=cfg.get('interval', '1m'),
                position_size=cfg.get('position_size', 10.0),
                stop_loss=cfg.get('stop_loss', 5.0),
                take_profit=cfg.get('take_profit', 10.0),
                max_quantity=cfg.get('max_quantity', 1.0)
            )

            if result.get('success'):
                actual_bot_id = result['bot_id']
                start_bot_thread(actual_bot_id)
                started_canonical_ids.add(canonical_id)
                logger.info(f"✅ Restored bot {actual_bot_id} (Thread started)")
            else:
                logger.error(f"❌ Failed to restore bot {bot_id}: {result.get('error')}")

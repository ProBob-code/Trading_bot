try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    print("❌ ERROR: 'python-dotenv' module not found.")
    print("💡 TIP: You should run this server using the provided virtual environment.")
    print("   Run: ./start_server.sh  (or: venv/bin/python3 api_server.py)")
    import sys
    sys.exit(1)

"""
GoatBotTrade API Server
=======================
V2 routes live in api_v2.py. Both are registered as Flask Blueprints.
"""

from flask import Flask, jsonify, request, send_from_directory, redirect, url_for
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import random
import string
from shared.database.db_manager import db_manager
from shared.services.trade_logger import get_trade_logger
from shared.services.system_state import get_system_state  # <--- SystemState import
from v2.engine.bot_manager_v2 import bot_manager_v2
from shared.config.settings import is_v1_enabled

import sys
import os
import threading
import time
import json
import logging # Added logging import
from datetime import datetime, timedelta, timezone
import atexit # Added atexit import
from pathlib import Path
import sys
import pandas as pd
import numpy as np
from loguru import logger

# Configure logging for maximum visibility
logger.remove()
logger.add(
    sys.stdout, 
    level="DEBUG", 
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add("trading_bot.log", rotation="10 MB", level="DEBUG")

# MySQL Logging Handler
def mysql_log_handler(message):
    """Log important messages to MySQL for frontend display."""
    try:
        if message.record['level'].no >= 20:  # INFO and above
            msg_text = message.record['message']
            level = message.record['level'].name
            # Only log trading-related messages
            if any(keyword in msg_text for keyword in ['BOT', 'Trade', 'SIGNAL', 'PROFIT', 'LOSS', 'PANIC', 'STOP', 'START', 'LONG', 'SHORT', 'CLOSE', 'COVER']):
                db_manager.log_event(
                    level=level,
                    message=msg_text[:500],
                    source=message.record['name']
                )
    except Exception:
        pass  # Never fail on logging

logger.add(mysql_log_handler, level="INFO")

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from shared.providers.crypto_provider import BinanceCryptoProvider
from shared.providers.stock_provider import YahooFinanceProvider
from v1.engine.execution.brokers.paper_trader import PaperTrader
from v1.engine.execution.order_manager import OrderManager, Order, OrderSide, OrderType
from shared.logic.strategies.strategy_engine import StrategyEngine, get_strategy_engine
from v1.engine.core.bot_manager import get_bot_manager, BotManager, BotStats

# Initialize Flask
app = Flask(__name__, static_folder='web', static_url_path='/static')
app.config['SECRET_KEY'] = 'god-bot-trade-secret-2026'  # PRO-CODER: Use environment variable in production
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# --- Flask-Login Setup ---
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login_page'

class User(UserMixin):
    def __init__(self, user_info):
        self.id = user_info['id']
        self.username = user_info.get('username', 'Anonymous')
        self.mobile = user_info.get('mobile')
        self.is_verified = user_info.get('is_verified', 0)

@login_manager.user_loader
def load_user(user_id):
    # In a real app, query database
    # For now, we'll implement the db query
    user_data = db_manager.get_user_by_id(int(user_id))
    return User(user_data) if user_data else None

def generate_otp():
    return ''.join(random.choices(string.digits, k=6))

def clean_nan(val):
    """Convert NaN/Inf to None (JSON null) for browser compatibility."""
    try:
        if pd.isna(val) or not np.isfinite(val):
            return None
    except:
        pass
    return val

# Initialize trading components
INITIAL_CAPITAL = 100000  # $100k paper trading
paper_trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
order_manager = OrderManager(paper_trader)
paper_trader.set_order_manager(order_manager)

# Data providers
crypto_provider = BinanceCryptoProvider()
stock_provider = YahooFinanceProvider()

# Strategy engine
strategy_engine = get_strategy_engine(min_confluence=3)

# State
current_market = "crypto"
current_symbol = "BTCUSDT"
current_interval = "1m"
current_strategy = "ichimoku"
is_streaming = False
stream_thread = None

# Auto-trading state
live_auto_trading = False
live_auto_thread = None
auto_trade_settings = {
    'confluence': 3,
    'position_size': 10,
    'check_interval': 5,
    'stop_loss': 5,
    'take_profit': 10
}
# Initialize Logger
trade_logger = get_trade_logger()
system_state = get_system_state()  # <--- Initialize System State

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

# Bot manager (singleton)
bot_manager = get_bot_manager()


# ============================================================
# REGISTER V1 & V2 BLUEPRINTS
# ============================================================

from v1.api.routes import v1_bp, init_v1, restore_bots_on_startup, bot_watchdog_loop
from v2.api.routes import v2_bp, init_v2, v2_paper_trader

init_v1(
    _socketio=socketio,
    _paper_trader=paper_trader,
    _order_manager=order_manager,
    _strategy_engine=strategy_engine,
    _bot_manager=bot_manager,
    _trade_logger=trade_logger,
    _db_manager=db_manager,
    _crypto_provider=crypto_provider,
    _stock_provider=stock_provider,
    _system_state_fn=get_system_state,
)

init_v2(
    _socketio=socketio,
    _strategy_engine=strategy_engine,
    _db_manager=db_manager,
    _crypto_provider=crypto_provider,
    _stock_provider=stock_provider,
    _system_state_fn=get_system_state,
)

# V2 SESSION INITIALIZATION (Lazy Loading Moved to BotManagerV2)
# Recovery: Mark any stale sessions as CRASHED
db_manager.v2_mark_crashed_sessions()

# Note: Position recovery is handled automatically in api_v2.init_v2()

def v2_shutdown_handler():
    """Cleanup current V2 session on server stop."""
    sid = system_state.get_session_id()
    if sid:
        logger.info(f"🛑 [V2] Marking session {sid} as STOPPED...")
        db_manager.v2_update_session_status(sid, "STOPPED")

atexit.register(v2_shutdown_handler)
logger.info(f"🚀 V2 Institutional Engine Live (Engine: {system_state.get_engine_version()})")

app.register_blueprint(v1_bp)
app.register_blueprint(v2_bp)


# ============================================================
# STATIC FILES
# ============================================================

@app.route('/api/stats')
def get_server_stats():
    """Simple health check endpoint for the frontend."""
    return jsonify({
        'success': True,
        'status': 'Online',
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/status')
def get_system_status():
    """Get global system pause/play status."""
    return jsonify({
        'paused': system_state.is_paused(),
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/system/pause', methods=['POST'])
def pause_system():
    """Pause all trading activity."""
    system_state.set_paused(True)
    socketio.emit('system_status', {'paused': True})
    return jsonify({'success': True, 'paused': True})

@app.route('/api/system/resume', methods=['POST'])
def resume_system():
    """Resume trading activity."""
    system_state.set_paused(False)
    socketio.emit('system_status', {'paused': False})
    return jsonify({'success': True, 'paused': False})

@app.route('/api/auto-trade/status')
@login_required
def get_auto_trade_status():
    """Get current auto-trading status and trade counts for the user."""
    user_bots = bot_manager.get_all_bots(current_user.id)
    running_bots = [b for b in user_bots if bot_manager.is_running(b['bot_id'])]

    # Get summary from trade_logger
    summary = trade_logger.get_daily_summary(current_user.id, datetime.now().strftime("%Y-%m-%d"))

    return jsonify({
        'total_trades': summary['total_trades'],
        'buy_trades': summary['wins'] + summary['losses'], # simplification
        'sell_trades': 0, # trade_logger doesn't track this yet
        'total_pnl': summary['total_pnl'],
        'start_time': datetime.now().isoformat(), # Mock
        'active_bots': len(running_bots)
    })

# ============================================================
# VERSION ROUTING — V1 (Legacy) & V2 (Institutional)
# ============================================================

@app.route('/godbot_login')
@app.route('/goatbot_login')
@app.route('/v1/godbot_login')
@app.route('/v1/goatbot_login')
@app.route('/v2/godbot_login')
@app.route('/v2/goatbot_login')
def login_page():
    """Serve the login page (V1)."""
    return send_from_directory('v1/web', 'godbot_login.html')

@app.route('/')
@app.route('/godbot_home')
@app.route('/goatbot_home')
@app.route('/paper_dashboard')
@app.route('/v2/home')
@app.route('/v2/godbot_home')
@app.route('/v2/goatbot_home')
@app.route('/godbot_home_v2')
@app.route('/goatbot_home_v2')
def paper_dashboard():
    """Serve V2 institutional terminal home."""
    return send_from_directory('v2/web', 'godbot_home.html')

@app.route('/v1/home')
@app.route('/v1/godbot_home')
@app.route('/v1/goatbot_home')
def index():
    """Serve the main frontend (V1 default)."""
    return send_from_directory('v1/web', 'godbot_home.html')

@app.route('/v1/<path:filename>')
def serve_v1(filename):
    """Serve V1 legacy frontend files."""
    return send_from_directory('v1/web', filename)

@app.route('/v2/<path:filename>')
def serve_v2(filename):
    """Serve V2 institutional terminal files."""
    return send_from_directory('v2/web', filename)

@app.route('/common/<path:filename>')
def serve_common(filename):
    """Serve shared frontend files."""
    return send_from_directory('shared/web_common', filename)

@app.route('/v1')
@app.route('/v1/')
def v1_root():
    """Redirect to V1 home."""
    return redirect(url_for('index'))

@app.route('/v2')
@app.route('/v2/')
def v2_root():
    """Redirect to V2 home."""
    return redirect(url_for('paper_dashboard'))


@app.route('/live-settings.html')
def live_settings():
    """Serve the live trading settings page (V1)."""
    return send_from_directory('v1/web', 'live-settings.html')

@app.route('/report.html')
def report_page():
    """Serve the trading report page (V1)."""
    return send_from_directory('v1/web', 'report.html')

@app.route('/v2_report.html')
@app.route('/v2/report')
def v2_report_page():
    """Serve the V2 institutional strategy report page."""
    return send_from_directory('v2/web', 'report.html')



# ============================================================
# POSITIONS API
# ============================================================

REPORTS_DIR = Path(__file__).parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)

def update_daily_report(user, symbol, side, qty, price, pnl=0):
    """Legacy function replaced by trade_logger.log_trade - kept briefly for safety but now redirects."""
    # This is now a no-op or a redirect if any old code calls it
    pass

@app.route('/api/reports')
@login_required
def get_reports_legacy():
    """Get all daily reports with date range support."""
    # Accept date filters
    date_filter = request.args.get('date')
    start_date = request.args.get('start_date', date_filter)
    end_date = request.args.get('end_date', date_filter)

    if not start_date:
        start_date = datetime.now().strftime("%Y-%m-%d")
    if not end_date:
        end_date = datetime.now().strftime("%Y-%m-%d")

    reports = []
    try:
        # Generate date range list
        start_dt = datetime.strptime(start_date, "%Y-%m-%d")
        end_dt = datetime.strptime(end_date, "%Y-%m-%d")

        # Limit range to 30 days for safety
        num_days = (end_dt - start_dt).days
        if num_days > 30:
            start_dt = end_dt - timedelta(days=30)

        current_dt = start_dt
        while current_dt <= end_dt:
            day_str = current_dt.strftime("%Y-%m-%d")
            summary = trade_logger.get_daily_summary(current_user.id, day_str)

            if summary['total_trades'] > 0:
                win_rate = (summary['wins'] / summary['total_trades'] * 100)
                avg_profit = (summary['total_pnl'] / summary['total_trades'])

                reports.append({
                    'date': day_str,
                    'user': current_user.username,
                    'total_trades': summary['total_trades'],
                    'win_loss': f"{summary['wins']}/{summary['losses']}",
                    'win_rate': f"{win_rate:.1f}%",
                    'total_pnl': round(summary['total_pnl'], 2),
                    'avg_profit': round(avg_profit, 2)
                })
            current_dt += timedelta(days=1)

    except Exception as e:
        logger.error(f"Error fetching filtered reports: {e}")
        # Fallback to today on error
        today = datetime.now().strftime("%Y-%m-%d")
        try:
            summary = trade_logger.get_daily_summary(current_user.id, today)
            if summary['total_trades'] > 0:
                reports.append({
                    'date': today,
                    'user': current_user.username,
                    'total_trades': summary['total_trades'],
                    'win_loss': f"{summary['wins']}/{summary['losses']}",
                    'win_rate': f"{(summary['wins']/summary['total_trades']*100):.1f}%",
                    'total_pnl': round(summary['total_pnl'], 2),
                    'avg_profit': round(summary['total_pnl']/summary['total_trades'], 2)
                })
        except: pass

    return jsonify(reports)

@app.route('/api/reports/trades')
@login_required
def get_all_trades():
    """Get all trades with filters for the current user."""
    # Accept 'date' as alias for start_date+end_date (used by report.html)
    date_filter = request.args.get('date')
    start_date = request.args.get('start_date', date_filter)
    end_date = request.args.get('end_date', date_filter)
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 100))

    trades = trade_logger.get_history(user_id=current_user.id, start_date=start_date, end_date=end_date, symbol=symbol, limit=limit)
    return jsonify(trades)

@app.route('/api/reports/summary')
@login_required
def get_daily_summary():
    """Get summarized daily stats for the current user."""
    date_str = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    summary = trade_logger.get_daily_summary(current_user.id, date_str)

    # Add authoritative P&L from paper_trader (equity based)
    account = paper_trader.get_account_info(current_user.id)
    summary['account_pnl'] = account['pnl']
    summary['account_pnl_pct'] = account['pnl_pct']

    return jsonify(summary)


@app.route('/api/reports/strategy-stats')
@login_required
def get_strategy_stats():
    """Get performance stats grouped by strategy for the user."""
    date_filter = request.args.get('date')
    trades = trade_logger.get_history(user_id=current_user.id, start_date=date_filter, end_date=date_filter, limit=1000)

    stats = {}
    for t in trades:
        strat = t.get('strategy', 'Unknown')
        if strat not in stats:
            stats[strat] = {
                'strategy': strat,
                'total_trades': 0,
                'closing_trades': 0,
                'wins': 0,
                'losses': 0,
                'pnl': 0.0,
                'volume': 0.0
            }

        s = stats[strat]
        s['total_trades'] += 1
        pnl = float(t.get('pnl', 0))
        s['volume'] += float(t.get('quantity', 0)) * float(t.get('price', 0))

        if abs(pnl) > 1e-9:
            s['closing_trades'] += 1
            s['pnl'] += pnl
            if pnl > 0: s['wins'] += 1
            elif pnl < 0: s['losses'] += 1

    # Calculate win rates
    for s in stats.values():
        if s['closing_trades'] > 0:
            s['win_rate'] = (s['wins'] / s['closing_trades']) * 100
        else:
            s['win_rate'] = 0

    return jsonify(list(stats.values()))

@app.route('/api/positions')
@login_required
def get_positions():
    """Get all positions and trade history for the user."""
    positions = paper_trader.get_positions(current_user.id)

    # Build open positions with current prices
    open_positions = []
    for pos in positions:
        symbol = pos['symbol']
        current_price = pos.get('current_price', pos['avg_price'])

        open_positions.append({
            'symbol': symbol,
            'side': pos['side'],
            'qty': abs(pos['quantity']),
            'avg_price': pos['avg_price'],
            'current_price': current_price,
            'net_pnl': pos.get('unrealized_pnl', 0),
            'open_interest': 0
        })

    # Filter closed positions: Only show if the bot that opened/closed them is STOPPED
    all_closed = paper_trader.get_closed_positions(current_user.id)
    filtered_closed = []

    for pos in all_closed:
        symbol = pos['symbol']
        user_bot_id = f"user_{current_user.id}_{current_market}_{symbol}".lower()

        if not bot_manager.is_running(user_bot_id):
            filtered_closed.append({
                'symbol': pos['symbol'],
                'side': pos['side'],
                'qty': pos['quantity'],
                'entry': pos['entry_price'],
                'exit': pos['exit_price'],
                'pnl': pos['realized_pnl'],
                'closed_at': pos['closed_at'].isoformat() if hasattr(pos['closed_at'], 'isoformat') else pos['closed_at']
            })

    # Get trade history with filters
    date_filter = request.args.get('date')
    start_date = request.args.get('start_date', date_filter)
    end_date = request.args.get('end_date', date_filter)

    trade_history = trade_logger.get_history(
        user_id=current_user.id,
        start_date=start_date,
        end_date=end_date,
        limit=50
    )

    return jsonify({
        'success': True,
        'open_positions': open_positions,
        'closed_positions': filtered_closed,
        'trade_history': trade_history,
        'journal': [],
        'pending_orders': paper_trader.get_pending_orders(current_user.id)
    })


@app.route('/api/balance', methods=['POST'])
@login_required
def update_balance():
    """Update paper trading balance for user."""
    data = request.json
    new_balance = data.get('cash') or data.get('balance', 100000)

    try:
        acc = paper_trader._get_account(current_user.id)
        acc.cash = float(new_balance)
        acc.initial_capital = float(new_balance)
        logger.info(f"💰 Balance reset for user {current_user.id} to: ${new_balance:,.2f}")

        return jsonify({
            'success': True,
            'balance': acc.cash,
            'message': f'Balance updated to ${new_balance:,.2f}'
        })
    except Exception as e:
        logger.error(f"Error updating balance: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/paper/reset', methods=['POST'])
@login_required
def reset_paper_trading():
    """Reset paper trading state for the user (V1 + V2)."""
    try:
        user_id = int(current_user.id)
        logger.warning(f"🚨 FORCE RESET: Paper Trading (V1+V2) triggered for user {user_id}")

        # 1. Reset V1 Broker state
        paper_trader.reset(user_id)

        # 2. Reset V1 Bot Manager stats for all bots of this user
        for bot in bot_manager.bots.values():
            if bot.user_id == user_id:
                bot.stats = BotStats()
                logger.debug(f"🧹 Reset V1 bot stats (memory): {bot.bot_id}")

        # 3. Clear V1 trade logs from DB
        try:
            db_manager.clear_user_trades(user_id)
        except Exception as e:
            logger.warning(f"Could not clear V1 trade logs: {e}")

        # 4. Clear V1 TradeLogger in-memory state (CRITICAL FOR COUNTER)
        trade_logger.reset_user(user_id)

        # 5. Reset V2 paper trader state
        v2_paper_trader.reset(user_id)

        # 6. Reset V2 Bot Manager stats
        from v2.engine.bot_manager_v2 import V2BotStats
        for bot in bot_manager_v2.bots.values():
            if bot.user_id == user_id:
                bot.stats = V2BotStats()
                logger.debug(f"🧹 Reset V2 bot stats (memory): {bot.bot_id}")

        # 7. Clear V2 trade logs + metrics from DB
        try:
            db_manager.v2_clear_user_trades(user_id)
        except Exception as e:
            logger.warning(f"Could not clear V2 trade logs: {e}")

        logger.info(f"🔥 SUCCESS: Paper Trading Reset (V1+V2) for user {user_id}")
        return jsonify({
            'success': True, 
            'message': 'V1 + V2 account reset successfully. All counters forced to zero.'
        })
    except Exception as e:
        logger.error(f"Error resetting paper trading for user {current_user.id}: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/v2/reset', methods=['POST'])
@login_required
def reset_v2_paper_trading():
    """Reset V2 paper trading state only."""
    try:
        user_id = current_user.id

        # Reset V2 paper trader
        v2_paper_trader.reset(user_id)

        # Reset V2 bot stats
        from v2.engine.bot_manager_v2 import V2BotStats
        for bot in bot_manager_v2.bots.values():
            if bot.user_id == user_id:
                bot.stats = V2BotStats()

        # Clear V2 trade logs
        try:
            db_manager.v2_clear_user_trades(user_id)
        except Exception as e:
            logger.warning(f"Could not clear V2 trade logs: {e}")

        logger.info(f"🔥 V2 Paper Trading Reset for user {user_id}")
        return jsonify({'success': True, 'message': 'V2 account reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting V2 paper trading: {e}")
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/crypto/depth/<symbol>')
def get_crypto_depth(symbol):
    """Get order book depth for symbol."""
    try:
        import requests
        response = requests.get(
            f'https://api.binance.com/api/v3/depth',
            params={'symbol': symbol.upper(), 'limit': 10},
            timeout=5
        )
        data = response.json()
        return jsonify({
            'bids': data.get('bids', []),
            'asks': data.get('asks', [])
        })
    except Exception as e:
        # Return mock data if API fails
        return jsonify({
            'bids': [
                ['89400.00', '2.5'],
                ['89350.00', '5.2'],
                ['89300.00', '8.1'],
                ['89250.00', '12.4'],
                ['89200.00', '15.8']
            ],
            'asks': [
                ['89450.00', '1.8'],
                ['89500.00', '4.3'],
                ['89550.00', '6.9'],
                ['89600.00', '10.2'],
                ['89650.00', '14.5']
            ]
        })


# ============================================================
# BACKTEST & ALERTS API
# ============================================================

@app.route('/api/backtest', methods=['POST'])
def run_backtest():
    """Run backtest on historical data."""
    try:
        data = request.json
        symbol = data.get('symbol', 'BTCUSDT')
        strategy = data.get('strategy', 'ichimoku')
        days = data.get('days', 30)
        stop_loss = data.get('stop_loss', 5.0)
        take_profit = data.get('take_profit', 10.0)

        # Get historical data
        klines = crypto_provider.get_klines(symbol, interval='1h', limit=days * 24)

        if klines.empty:
            return jsonify({'success': False, 'error': 'No data available'})

        # Import backtester
        from v1.engine.core.backtester import get_backtest_engine
        engine = get_backtest_engine()

        # Get strategy function
        def strategy_func(df):
            signal = strategy_engine.get_signal(df, strategy)
            return signal.signal if signal else 'HOLD'

        # Run backtest
        result = engine.run(
            strategy_name=strategy,
            data=klines,
            strategy_func=strategy_func,
            stop_loss_pct=stop_loss,
            take_profit_pct=take_profit
        )

        return jsonify({
            'success': True,
            'result': result.to_dict()
        })

    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/api/alerts/config', methods=['GET', 'POST'])
def alert_config():
    """Get or set alert configuration."""
    if request.method == 'GET':
        return jsonify({
            'telegram_configured': bool(os.getenv('TELEGRAM_BOT_TOKEN')),
            'email_configured': bool(os.getenv('SMTP_USER')),
            'alerts_enabled': True
        })
    else:
        # POST - save config (would typically save to file/db)
        data = request.json
        return jsonify({'success': True, 'message': 'Config saved'})


# ============================================================
# CRYPTO API ENDPOINTS
# ============================================================

@app.route('/api/price/<symbol>')
def get_crypto_price(symbol):
    """Get current crypto price."""
    data = crypto_provider.get_current_price(symbol.upper())
    return jsonify(data)


@app.route('/api/ticker/<symbol>')
def get_crypto_ticker(symbol):
    """Get 24h crypto ticker stats."""
    data = crypto_provider.get_ticker_24h(symbol.upper())
    if 'timestamp' in data:
        data['timestamp'] = data['timestamp'].isoformat()
    return jsonify(data)


@app.route('/api/klines/<symbol>')
def get_crypto_klines(symbol):
    """Get historical crypto klines."""
    interval = request.args.get('interval', '1m')
    limit = int(request.args.get('limit', 100))

    df = crypto_provider.get_historical_klines(
        symbol=symbol.upper(),
        interval=interval,
        limit=limit
    )

    if df.empty:
        logger.warning(f"⚠️ No klines found for {symbol}")
        return jsonify([])

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            'time': int(idx.timestamp()),
            'open': clean_nan(row['open']),
            'high': clean_nan(row['high']),
            'low': clean_nan(row['low']),
            'close': clean_nan(row['close']),
            'volume': clean_nan(row['volume'])
        })

    return jsonify(candles)


# ============================================================
# AUTHENTICATION ENDPOINTS
# ============================================================

@app.route('/api/auth/register', methods=['POST'])
def register():
    data = request.json
    mobile = data.get('mobile')

    if not mobile:
        return jsonify({'success': False, 'error': 'Mobile number is required'})

    user = db_manager.get_user_by_mobile(mobile)
    if not user:
        if not db_manager.create_user(mobile):
            return jsonify({'success': False, 'error': 'Database error'})
        user = db_manager.get_user_by_mobile(mobile)

    otp = generate_otp()
    db_manager.update_user_otp(user['id'], otp)

    # MOCK OTP Log
    logger.info(f"🔑 [MOCK OTP] For {mobile}: {otp}")

    return jsonify({
        'success': True,
        'message': 'OTP sent to mobile',
        'otp_sent': True # In demo/mock, we tell them it's sent
    })

@app.route('/api/auth/verify', methods=['POST'])
def verify():
    data = request.json
    mobile = data.get('mobile')
    otp = data.get('otp')
    username = data.get('username')
    password = data.get('password') # In real app, hash it

    user = db_manager.get_user_by_mobile(mobile)
    if not user or user['otp'] != otp:
        return jsonify({'success': False, 'error': 'Invalid OTP'})

    # Store user (verification)
    db_manager.verify_user(user['id'], username, password)

    # Log them in
    login_user(User(db_manager.get_user_by_mobile(mobile)))

    return jsonify({'success': True, 'message': 'Account verified and logged in'})

@app.route('/api/auth/login', methods=['POST'])
def login():
    data = request.json
    username = data.get('username')
    password = data.get('password')

    user = db_manager.get_user_by_username(username)
    if user and user['password_hash'] == password: # Crude password check for demo
        login_user(User(user))
        return jsonify({'success': True, 'message': 'Logged in successfully'})

    return jsonify({'success': False, 'error': 'Invalid credentials'})

@app.route('/api/auth/logout', methods=['POST'])
@login_required
def logout():
    logout_user()
    return jsonify({'success': True, 'message': 'Logged out'})

@app.route('/api/auth/status')
def auth_status():
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': {
                'id': current_user.id,
                'username': current_user.username,
                'is_verified': current_user.is_verified
            }
        })
    return jsonify({'authenticated': False})

@app.route('/api/user/profile')
@login_required
def get_user_profile():
    user = db_manager.get_user_by_id(current_user.id)
    if not user:
        return jsonify({'error': 'User not found'}), 404
    
    return jsonify({
        'id': user['id'],
        'public_id': user.get('public_id', '—'),
        'username': user['username'],
        'mobile': user['mobile'],
        'name': user.get('name') or user['username']
    })


# ============================================================
# STOCK API ENDPOINTS
# ============================================================

@app.route('/api/stocks/price/<symbol>')
def get_stock_price(symbol):
    """Get current stock price."""
    data = stock_provider.get_current_quote(symbol)
    if 'timestamp' in data:
        data['timestamp'] = data['timestamp'].isoformat()
    return jsonify(data)


@app.route('/api/stocks/klines/<symbol>')
def get_stock_klines(symbol):
    """Get historical stock klines."""
    interval = request.args.get('interval', '1d')
    limit = int(request.args.get('limit', 100))

    df = stock_provider.get_historical_data(
        symbol=symbol,
        interval=interval,
        limit=limit
    )

    if df.empty:
        return jsonify([])

    candles = []
    for idx, row in df.iterrows():
        candles.append({
            'time': int(idx.timestamp()),
            'open': clean_nan(row['open']),
            'high': clean_nan(row['high']),
            'low': clean_nan(row['low']),
            'close': clean_nan(row['close']),
            'volume': clean_nan(row['volume'])
        })

    return jsonify(candles)


@app.route('/api/stocks/market-status/<symbol>')
def get_market_status(symbol):
    """Check if stock market is open."""
    data = stock_provider.get_market_hours(symbol)
    return jsonify(data)


# ============================================================
# NEWS SENTIMENT API
# ============================================================

# Import sentiment analyzer (lazy load to avoid startup issues)
_sentiment_analyzer = None

def get_sentiment():
    """Get or create sentiment analyzer singleton."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        try:
            from shared.logic.sentiment.news_sentiment import get_sentiment_analyzer
            _sentiment_analyzer = get_sentiment_analyzer()
        except Exception as e:
            logger.error(f"Failed to load sentiment analyzer: {e}")
            return None
    return _sentiment_analyzer


@app.route('/api/news')
def get_news():
    """Get recent news with sentiment, filtered by market type."""
    symbol = request.args.get('symbol', None)
    market = request.args.get('market', None)  # 'crypto' or 'stocks'
    limit = int(request.args.get('limit', 15))

    analyzer = get_sentiment()
    if not analyzer:
        return jsonify({'success': False, 'error': 'Sentiment analyzer not available', 'news': []})

    try:
        news = analyzer.get_recent_news(limit=limit, symbol=symbol, market=market)
        return jsonify({
            'success': True,
            'news': news,
            'count': len(news),
            'market': market or 'all'
        })
    except Exception as e:
        logger.error(f"Error getting news: {e}")
        return jsonify({'success': False, 'error': str(e), 'news': []})

# ============================================================
# NEWS WEBSOCKET LOOP
# ============================================================

def news_websocket_loop():
    """Background thread to emit live news updates via Socket.IO."""
    logger.info("📰 Starting News WebSocket background loop")

    # Use localized import to avoid circular dependency
    from shared.logic.sentiment.news_sentiment import get_sentiment_analyzer
    analyzer = get_sentiment_analyzer()

    while True:
        try:
            # Fetch recent news for the ACTIVE market
            market = current_market or 'crypto'
            news = analyzer.get_recent_news(limit=10, market=market)

            if news:
                socketio.emit('news_update', news)
                logger.debug(f"📰 Emitted news_update event ({len(news)} items)")

        except Exception as e:
            logger.error(f"Error in news_websocket_loop: {e}")

        # Wait 60 seconds before next update
        time.sleep(60)

# Start news loop in background
news_thread = threading.Thread(target=news_websocket_loop, daemon=True)
news_thread.start()

def calculate_live_pulse(market, symbol):
    """Calculate an accurate momentum pulse score (0-100) from real technical data."""
    try:
        # 1. Fetch data based on market type
        provider = crypto_provider if market == 'crypto' else stock_provider
        interval = '1m' if market == 'crypto' else '1m' # Simplified for pulse
        
        # Get enough data for indicators (need ~100 bars for EMA/MACD)
        df = None
        if market == 'crypto':
            df = provider.get_historical_klines(symbol, interval, limit=100)
        else:
            df = provider.get_historical_data(symbol, interval, period='1d') # Yahoo needs 'period'
            
        if df is None or df.empty or len(df) < 20:
            return 50 # Default middle-ground if data fails

        # 2. Run Technical Indicators (Reuse engine logic)
        engine = get_strategy_engine()
        df = engine.get_indicators(df)
        
        # 3. Calculate Weighted Score
        last_row = df.iloc[-1]
        rsi = last_row.get('rsi', 50)
        macd = last_row.get('macd', 0)
        macd_hist = last_row.get('macd_hist', 0)
        price = last_row['close']
        sma20 = last_row.get('sma20', price)
        
        # Start at 50
        score = 50
        
        # RSI Influence (Max +/- 25)
        # Above 70 is overbought but high momentum strength
        if rsi > 70: score += 15
        elif rsi < 30: score -= 15
        elif rsi > 55: score += 5
        elif rsi < 45: score -= 5
        
        # MACD Influence (Max +/- 15)
        if macd_hist > 0: score += 10
        else: score -= 10
        
        # SMA20 Relationship (Max +/- 10)
        if price > sma20: score += 10
        else: score -= 10
        
        return max(5, min(95, score)) # Keep within 5-95 range
        
    except Exception as e:
        logger.error(f"Error calculating live pulse for {symbol}: {e}")
        return 50

def intelligence_loop():
    """Background thread to generate market pulse and AI insights."""
    import random

    pulse_labels = [
        "EXTREME FEAR", "FEARFUL", "NEUTRAL",
        "BULLISH MOMENTUM", "EXTREME GREED"
    ]

    ai_templates = {
        'bullish': [
            "BTC momentum is accelerating. Convergence on the 1m chart suggests a potential breakout above resistance.",
            "Price reclaimed the 20-period moving average with rising volume — classic bullish continuation pattern.",
            "Higher lows forming on the 5m chart with compressing Bollinger Bands. Expect a volatility expansion to the upside.",
            "Ichimoku cloud turned green on the 1-hour timeframe. Kumo twist confirms bullish trend shift.",
            "MACD histogram flipped positive with a bullish crossover. RSI trending up but not yet overbought — room to run.",
            "Price broke above a key descending trendline with a strong green candle. Retest of breakout level as support likely.",
            "Whale buy orders detected stacking in the bid wall. Institutional accumulation phase may be underway.",
            "Large market buy orders sweeping through asks. Aggressive buying pressure dominating the order book.",
            "Open interest surging alongside price — fresh capital entering longs. Funding rate still neutral, healthy rally.",
            "Exchange net outflows spiking — coins moving to cold storage. Supply squeeze could fuel further upside.",
            "Fear & Greed Index shifting from Neutral to Greed. Historically, early greed phases precede 5-10% rallies.",
            "Social sentiment turning sharply positive across crypto Twitter. Retail interest re-entering the market.",
            "Strategy 5 confluence is at maximum. All five sub-signals aligned bullish — high-conviction setup.",
            "Golden cross forming on the daily chart. Long-term momentum shifting decisively in favor of bulls.",
            "Breakout above the weekly pivot point confirmed. Next target: upper Fibonacci extension at 1.618 level.",
            "Buyers absorbing all sell-side liquidity at this level. The path of least resistance is higher.",
        ],
        'bearish': [
            "Volatility is spiking to the downside. Resistance at current levels is holding firm — rejection candle forming.",
            "Death cross on the 4-hour chart as the 50 MA drops below the 200 MA. Bearish momentum building.",
            "Price rejected from the upper Bollinger Band with a long wick. Distribution pattern emerging.",
            "RSI divergence detected — price made a higher high but RSI made a lower high. Classic bearish divergence.",
            "Ichimoku cloud is deep red with a bearish Kumo twist ahead. Trend structure favors sellers.",
            "Lower highs and lower lows confirmed on the 15m chart. Downtrend structure intact.",
            "Sell pressure increasing in the order book. Large limit asks stacking above current price.",
            "Whale wallets moved significant holdings to exchanges in the last hour. Distribution event possible.",
            "Funding rate spiking positive — overleveraged longs may get liquidated, adding downside pressure.",
            "Open interest dropping while price falls — long liquidation cascade in progress.",
            "Exchange inflows surging — potential sell pressure incoming as coins arrive on trading desks.",
            "Trend reversal detected on multiple timeframes. Awaiting lower support confirmation before re-entry.",
            "Fear & Greed Index plunging toward Extreme Fear territory. Panic selling could accelerate near-term.",
            "Macro headwinds intensifying — rising bond yields and dollar strength pressuring risk assets.",
            "Market structure broke down below the weekly support. Bears targeting the next demand zone.",
            "Volume profile shows a gap below current price — thin liquidity could accelerate the drop.",
        ],
        'neutral': [
            "Market is currently ranging. Consolidation pattern forming between well-defined support and resistance.",
            "Price coiling inside a symmetrical triangle. A decisive breakout in either direction is imminent.",
            "Bollinger Bands are at their tightest squeeze in 48 hours. Major move loading — direction TBD.",
            "Choppy price action between the 200 and 50 Moving Averages. No clear trend — range-trading optimal.",
            "Doji candles forming consecutively — market indecision at a key level. Wait for confirmation.",
            "Low relative volume detected. Strategic patience recommended — avoid forcing trades in thin markets.",
            "Weekend liquidity conditions in effect. Spreads are wider and slippage risk is elevated.",
            "Volume delta is flat — neither buyers nor sellers have conviction. Sidelined capital waiting for a catalyst.",
            "Asian session showing minimal volatility. European and US sessions likely to set the directional tone.",
            "Sentiment is balanced. GodBot confirms no immediate breakout signals — monitoring for setup.",
            "Fear & Greed Index sitting at 50 — perfectly neutral. Market awaiting the next macro catalyst.",
            "Funding rate near zero across major exchanges. No directional bias from derivatives markets.",
            "Oscillators are mean-reverting to the midline. Neither overbought nor oversold — equilibrium phase.",
            "Institutional flow data shows equal buy and sell activity. Smart money is hedged and waiting.",
            "Accumulation/Distribution indicator is flat. No clear institutional positioning detected at this level.",
            "Price hugging the VWAP — fair value zone. Breakout traders should wait for a clear deviation.",
        ]
    }

    while True:
        try:
            logger.info("🧠 Running intelligence loop...")
            # 1. Fetch current global state
            # In a real scenario, we'd track the symbol most users are looking at
            # For now, we use the global current_symbol
            market = current_market
            symbol = current_symbol
            
            # 2. Calculate Real Pulse
            pulse_score = calculate_live_pulse(market, symbol)

            # 3. Pick AI Thought based on real data
            mood = 'neutral'
            if pulse_score > 65: mood = 'bullish'
            elif pulse_score < 35: mood = 'bearish'
            
            # Additional context for insights
            thought = random.choice(ai_templates[mood])
            
            # Replace placeholder mentions with real symbol
            thought = thought.replace("BTC", symbol).replace("Market", f"{symbol} market")

            # 4. Emit updates
            payload = {
                'score': int(pulse_score),
                'momentum': int(pulse_score),
                'pulse_label': pulse_labels[min(int(pulse_score / 20), 4)],
                'insights': [thought],
                'timestamp': datetime.now().strftime("%H:%M:%S")
            }
            socketio.emit('market_intel', payload)
            logger.info(f"🧠 Emitted market_intel: {payload}")

            time.sleep(15) # Update every 15 seconds
        except Exception as e:
            logger.error(f"Intelligence loop error: {e}")
            time.sleep(10)

intel_thread = threading.Thread(target=intelligence_loop, daemon=True)
intel_thread.start()


@app.route('/api/sentiment/<symbol>')
@login_required
def get_symbol_sentiment(symbol):
    """Get sentiment analysis for a specific symbol."""
    analyzer = get_sentiment()
    if not analyzer:
        return jsonify({
            'success': False,
            'symbol': symbol,
            'score': 0,
            'label': 'NEUTRAL',
            'confidence': 0,
            'news_count': 0,
            'headlines': []
        })

    try:
        result = analyzer.get_sentiment_for_symbol(symbol)
        return jsonify({
            'success': True,
            'symbol': result.symbol,
            'score': result.score,
            'label': result.label,
            'confidence': result.confidence,
            'news_count': result.news_count,
            'headlines': result.top_headlines[:5]
        })
    except Exception as e:
        logger.error(f"Error getting sentiment for {symbol}: {e}")
        return jsonify({
            'success': False,
            'symbol': symbol,
            'score': 0,
            'label': 'NEUTRAL',
            'confidence': 0,
            'news_count': 0,
            'headlines': []
        })


@app.route('/api/news/movers')
@login_required
def get_top_movers():
    """Get stocks mentioned most in bullish/bearish news."""
    analyzer = get_sentiment()
    if not analyzer:
        return jsonify({'success': False, 'top_gainers': [], 'top_losers': []})

    try:
        movers = analyzer.get_top_movers()
        return jsonify({
            'success': True,
            'top_gainers': movers['top_gainers'],
            'top_losers': movers['top_losers']
        })
    except Exception as e:
        logger.error(f"Error getting movers: {e}")
        return jsonify({'success': False, 'top_gainers': [], 'top_losers': []})


# ============================================================
# WEBSOCKET - REAL-TIME STREAMING
# ============================================================

# State for multi-user streaming
user_watched_symbols = {}  # sid -> {'symbol': str, 'market': str}

def get_active_symbols():
    """Get all symbols currently being watched by users or bots."""
    symbols = set()
    # 1. From active users
    for watcher in user_watched_symbols.values():
        symbols.add((watcher['symbol'], watcher['market']))

    # 2. From default symbols (always keep BTCUSDT and ETHUSDT live)
    symbols.add(('BTCUSDT', 'crypto'))
    symbols.add(('ETHUSDT', 'crypto'))

    # 3. From active bots (ensure bot_manager is accessible)
    try:
        from v1.engine.core.bot_manager import get_bot_manager
        bm = get_bot_manager()
        for bot in bm.bots.values():
            if bm.is_running(bot.bot_id):
                symbols.add((bot.config.symbol, 'crypto' if 'USDT' in bot.config.symbol else 'stock'))
    except Exception:
        pass

    return symbols

def price_stream():
    """Background thread to stream prices for all active symbols."""
    global is_streaming

    while is_streaming:
        try:
            active_symbols = get_active_symbols()

            for symbol, market in active_symbols:
                try:
                    # Fetch data
                    is_crypto = market == 'crypto'
                    if is_crypto:
                        price_data = crypto_provider.get_current_price(symbol)
                        ticker = crypto_provider.get_ticker_24h(symbol)
                    else:
                        price_data = stock_provider.get_current_quote(symbol)
                        ticker = {
                            'price_change': price_data.get('change', 0),
                            'price_change_pct': price_data.get('change_pct', 0),
                            'high_24h': price_data.get('high', 0),
                            'low_24h': price_data.get('low', 0),
                            'volume_24h': price_data.get('volume', 0)
                        }

                    price = price_data.get('price', 0)
                    if price <= 0:
                        logger.warning(f"⚠️ Skipping 0 price for {symbol}")
                        continue

                    if not system_state.is_paused():
                        paper_trader.set_prices({symbol: price})
                        v2_paper_trader.set_prices({symbol: price})

                    # Emit update to symbol-specific room
                    update_payload = {
                        'symbol': symbol,
                        'market': market,
                        'price': price,
                        'change_pct': ticker.get('price_change_pct', 0),
                        'high_24h': ticker.get('high_24h', 0),
                        'low_24h': ticker.get('low_24h', 0),
                        'volume_24h': ticker.get('volume_24h', 0),
                        'timestamp': datetime.now().isoformat()
                    }
                    socketio.emit('price_update', update_payload, room=f"ticker_{symbol}")


                except Exception as e:
                    logger.error(f"Error streaming {symbol}: {e}")

            # Per-user account updates
            for user_id in list(paper_trader.accounts.keys()):
                try:
                    acc = paper_trader.get_account_info(user_id)
                    positions = paper_trader.get_positions(user_id)
                    summary = trade_logger.get_daily_summary(user_id, datetime.now().strftime("%Y-%m-%d"))

                    socketio.emit('account_update', {
                        'cash': acc['cash'],
                        'total_value': acc['total_value'],
                        'buying_power': acc['buying_power'],
                        'pnl': acc['pnl'],
                        'pnl_pct': acc['pnl_pct'],
                        'total_trades': summary['total_trades'],
                        'positions': positions
                    }, room=f"user_{user_id}")
                except Exception as e:
                    pass

            # Per-user V2 account updates (separate from V1)
            for user_id in list(v2_paper_trader.accounts.keys()):
                try:
                    v2_acc = v2_paper_trader.get_account_info(user_id)
                    v2_positions = v2_paper_trader.get_positions(user_id)
                    
                    socketio.emit('v2_account_update', {
                        **v2_acc,
                        'positions': v2_positions
                    }, room=f"user_{user_id}")
                except Exception as e:
                    pass

            time.sleep(1)

        except Exception as e:
            logger.error(f"Stream outer error: {e}")
            time.sleep(2)


@socketio.on('set_symbol')
def handle_set_symbol(data):
    """Update global state for live data loops based on user selection."""
    global current_symbol, current_market
    symbol = data.get('symbol')
    market = data.get('market')
    if symbol:
        current_symbol = symbol
        logger.info(f"📍 Global symbol updated to: {current_symbol}")
    if market:
        current_market = market
        logger.info(f"📍 Global market updated to: {current_market}")
    
    # Trigger immediate pulse update if possible
    emit('symbol_updated', {'symbol': current_symbol, 'market': current_market})

@socketio.on('join_bot')
def handle_connect():
    """Handle client connection with user rooms and streaming."""
    sid = request.sid
    global is_streaming, stream_thread

    # Set default watched symbol for this session
    user_watched_symbols[sid] = {'symbol': 'BTCUSDT', 'market': 'crypto'}
    join_room('ticker_BTCUSDT')

    if current_user.is_authenticated:
        user_room = f"user_{current_user.id}"
        join_room(user_room)
        logger.info(f"👤 User {current_user.username} (ID: {current_user.id}) connected (sid: {sid})")
    else:
        logger.info(f"🌐 Anonymous client connected (sid: {sid})")

    # Start price stream if not already running
    if not is_streaming or stream_thread is None or not stream_thread.is_alive():
        logger.info("📡 Starting price stream loop...")
        is_streaming = True
        stream_thread = threading.Thread(target=price_stream, daemon=True)
        stream_thread.start()

    emit('connected', {
        'status': 'ok',
        'user_id': current_user.id if current_user.is_authenticated else None,
        'symbol': current_symbol,
        'market': current_market
    })

@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    sid = request.sid
    if sid in user_watched_symbols:
        del user_watched_symbols[sid]

    if current_user.is_authenticated:
        user_room = f"user_{current_user.id}"
        leave_room(user_room)
        logger.info(f"🔌 User {current_user.id} disconnected (sid: {sid})")
    else:
        logger.info(f"🔌 Anonymous client disconnected (sid: {sid})")

@socketio.on('change_symbol')
def handle_symbol_change(data):
    """Change the trading symbol and update subscriptions."""
    sid = request.sid
    old_symbol = user_watched_symbols.get(sid, {}).get('symbol')
    new_symbol = data.get('symbol', 'BTCUSDT').upper()
    market = user_watched_symbols.get(sid, {}).get('market', 'crypto')

    if old_symbol:
        leave_room(f"ticker_{old_symbol}")

    user_watched_symbols[sid] = {'symbol': new_symbol, 'market': market}
    join_room(f"ticker_{new_symbol}")

    emit('symbol_changed', {'symbol': new_symbol})
    logger.info(f"🔄 Symbol changed to {new_symbol} for sid {sid}")

@socketio.on('join_ticker_rooms')
def handle_join_ticker_rooms(data):
    """Join multiple ticker rooms for the Market Watch list."""
    sid = request.sid
    symbols = data.get('symbols', [])
    for s in symbols:
        room = f"ticker_{s.upper()}"
        join_room(room)
    logger.info(f"📡 Sid {sid} joined {len(symbols)} ticker rooms for Market Watch")

@socketio.on('change_market')
def handle_market_change(data):
    """Change the trading market and update subscriptions."""
    sid = request.sid
    old_symbol = user_watched_symbols.get(sid, {}).get('symbol')

    market = data.get('market', 'crypto')
    symbol = data.get('symbol', 'BTCUSDT' if market == 'crypto' else 'AAPL').upper()

    if old_symbol:
        leave_room(f"ticker_{old_symbol}")

    user_watched_symbols[sid] = {'symbol': symbol, 'market': market}
    join_room(f"ticker_{symbol}")

    emit('market_changed', {
        'market': market,
        'symbol': symbol
    })
    logger.info(f"🔄 Market changed to {market} ({symbol}) for sid {sid}")


# ============================================================
# MAIN
# ============================================================

if __name__ == '__main__':
    print("⚡ GoatBotTrade Server starting...")
    
    # STARTUP BANNER (Legacy/Institutional Enforcement)
    logger.info("====================================")
    logger.info("Trading Mode: V2 Institutional Engine")
    if not is_v1_enabled():
        logger.info("V1 Trading Engine: DISABLED")
        logger.info("Strategy Source: V2 ONLY")
    else:
        logger.warning("V1 Trading Engine: ENABLED (Legacy Mode)")
    logger.info("====================================")

    print("📊 V2 Institutional Dashboard: http://localhost:5050")
    print("🚀 V2 Mode: ACTIVE (Modular Pipeline)")
    print("🪙 Crypto: 24/7 live data from Binance")
    print("📈 Stocks: Yahoo Finance (market hours)")
    print("🚀 Target Port: 5050")

    # Restore any previously running bots
    if is_v1_enabled():
        restore_bots_on_startup()
    else:
        logger.info("🛡️ Skipping V1 bot restore (Engine Disabled)")

    # Start the bot watchdog in a background thread
    if is_v1_enabled():
        watchdog_thread = threading.Thread(target=bot_watchdog_loop, daemon=True)
        watchdog_thread.start()
    else:
        logger.info("🛡️ Skipping V1 bot watchdog (Engine Disabled)")

    # Get port from environment variable for cloud deployment (Railway/Heroku/etc)
    port = int(os.getenv('PORT', 5050))
    socketio.run(app, host='0.0.0.0', port=port, debug=True, use_reloader=False, allow_unsafe_werkzeug=True)

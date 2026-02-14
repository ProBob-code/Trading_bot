"""
GodBotTrade API Server
======================

Multi-asset trading platform API with WebSocket support.
Supports: Crypto (24/7) and Stocks (market hours)
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user
import random
import string
from src.database.db_manager import db_manager
from src.services.trade_logger import get_trade_logger
from src.services.system_state import get_system_state  # <--- SystemState import
import sys
import os
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import sys
import pandas as pd
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
    try:
        record = message.record
        # Avoid circular logging
        if record["name"] == "mysql.connector":
            return
        
        # Determine bot_id and user_id if present in extra
        extra = record.get("extra", {})
        bot_id = extra.get("bot_id")
        user_id = extra.get("user_id")
        
        db_manager.add_log(
            level=record["level"].name,
            module=record["name"],
            message=record["message"],
            bot_id=bot_id,
            user_id=user_id
        )
    except Exception:
        pass

logger.add(mysql_log_handler, level="INFO")

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.crypto_provider import BinanceCryptoProvider
from src.data.stock_provider import YahooFinanceProvider
from src.execution.brokers.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager, Order, OrderSide, OrderType
from src.strategies.strategy_engine import StrategyEngine, get_strategy_engine
from src.engine.bot_manager import get_bot_manager, BotManager, BotStats

# Initialize Flask
app = Flask(__name__, static_folder='web', static_url_path='')
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
    running_bots = [b for b in user_bots if bot_manager.is_running(b.bot_id)]
    
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

@app.route('/')
def index():
    """Serve the main frontend."""
    return send_from_directory('web', 'index.html')


@app.route('/login.html')
def login_page():
    """Serve the login page."""
    return send_from_directory('web', 'login.html')


@app.route('/live-settings.html')
def live_settings():
    """Serve the live trading settings page."""
    return send_from_directory('web', 'live-settings.html')


@app.route('/report.html')
def report_page():
    """Serve the trading report page."""
    return send_from_directory('web', 'report.html')


# Bot manager (singleton)
bot_manager = get_bot_manager()


# ============================================================
# BOT MANAGEMENT API
# ============================================================











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
    """Get all daily reports (Legacy Endpoint - redirects to new service)."""
    reports = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Get user's today's summary
        summary = trade_logger.get_daily_summary(current_user.id, today)
        
        # Convert to expected format
        if summary['total_trades'] > 0:
            win_rate = (summary['wins'] / summary['total_trades'] * 100) if summary['total_trades'] > 0 else 0
            avg_profit = (summary['total_pnl'] / summary['total_trades']) if summary['total_trades'] > 0 else 0
            
            reports.append({
                'date': today,
                'user': current_user.username,
                'total_trades': summary['total_trades'],
                'win_loss': f"{summary['wins']}/{summary['losses']}",
                'win_rate': f"{win_rate:.1f}%",
                'total_pnl': round(summary['total_pnl'], 2),
                'avg_profit': round(avg_profit, 2)
            })
            
    except Exception as e:
        logger.error(f"Error fetching legacy reports: {e}")
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
    return jsonify(summary)

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
            
    return jsonify({
        'success': True,
        'open_positions': open_positions,
        'closed_positions': filtered_closed,
        'trade_history': [], # Redirect to /api/reports/trades
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
    """Reset paper trading state for the user."""
    try:
        # 1. Reset Broker state for this user
        paper_trader.reset(current_user.id)
        
        # 2. Reset Bot Manager stats for all bots of this user
        user_bots = bot_manager.get_all_bots(current_user.id)
        for bot in user_bots:
            bot.stats = BotStats()
            
        logger.info(f"🔥 Paper Trading Reset for user {current_user.id}")
        return jsonify({'success': True, 'message': 'Account reset successfully'})
    except Exception as e:
        logger.error(f"Error resetting paper trading for user {current_user.id}: {e}")
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
        from src.engine.backtester import get_backtest_engine
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
    
    # Calculate indicators
    df_with_indicators = strategy_engine.get_indicators(df)
    
    candles = []
    for idx, row in df_with_indicators.iterrows():
        candles.append({
            'time': int(idx.timestamp()),
            'open': clean_nan(row['open']),
            'high': clean_nan(row['high']),
            'low': clean_nan(row['low']),
            'close': clean_nan(row['close']),
            'volume': clean_nan(row['volume']),
            'indicators': {k: clean_nan(v) for k, v in row.items() if k not in ['open', 'high', 'low', 'close', 'volume']}
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
    
    # Calculate indicators
    df_with_indicators = strategy_engine.get_indicators(df)
    
    candles = []
    for idx, row in df_with_indicators.iterrows():
        candles.append({
            'time': int(idx.timestamp()),
            'open': clean_nan(row['open']),
            'high': clean_nan(row['high']),
            'low': clean_nan(row['low']),
            'close': clean_nan(row['close']),
            'volume': clean_nan(row['volume']),
            'indicators': {k: clean_nan(v) for k, v in row.items() if k not in ['open', 'high', 'low', 'close', 'volume']}
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
            from src.sentiment.news_sentiment import get_sentiment_analyzer
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
    from src.sentiment.news_sentiment import get_sentiment_analyzer
    analyzer = get_sentiment_analyzer()
    
    while True:
        try:
            # Fetch recent news for the current market
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

def intelligence_loop():
    """Background thread to generate market pulse and AI insights."""
    import random
    
    pulse_labels = [
        "EXTREME FEAR", "FEARFUL", "NEUTRAL", 
        "BULLISH VOLATILITY", "EXTREME GREED"
    ]
    
    ai_templates = {
        'bullish': [
            "BTC momentum is accelerating. Convergence on the 1m chart suggests a potential breakout.",
            "Whale buy orders detected in the $68k-$69k range. Market pulse is heating up.",
            "Strategy 5 confluence is high. Bulls are in control of the current range."
        ],
        'bearish': [
            "Volatility is spiking to the downside. Resistance at current levels is holding firm.",
            "Sell pressure increasing in order book. Caution advised for long positions.",
            "Trend reversal detected on multiple timeframes. Awaiting support confirmation."
        ],
        'neutral': [
            "Market is currently ranging. Consolidation pattern forming on hourly chart.",
            "Low relative volume detected. Strategic patience recommended for new entries.",
            "Sentiment is balanced. GodBot confirms no immediate breakout signals."
        ]
    }

    while True:
        try:
            # 1. Calculate Pulse (Demo logic based on current symbol)
            # In a real app, we'd use ATR, RSI, and News Sentiment
            pulse_score = random.randint(30, 85) # Base range
            
            # 2. Pick AI Thought
            mood = 'neutral'
            if pulse_score > 70: mood = 'bullish'
            elif pulse_score < 40: mood = 'bearish'
            
            thought = random.choice(ai_templates[mood])
            
            # 3. Emit updates
            socketio.emit('market_intel', {
                'pulse_score': pulse_score,
                'pulse_label': pulse_labels[min(int(pulse_score / 20), 4)],
                'ai_thought': thought,
                'timestamp': datetime.now().strftime("%H:%M:%S")
            })
            
            time.sleep(15) # Update every 15 seconds
        except Exception as e:
            logger.error(f"Intelligence loop error: {e}")
            time.sleep(10)

intel_thread = threading.Thread(target=intelligence_loop, daemon=True)
intel_thread.start()


@app.route('/api/sentiment/<symbol>')
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
# ACCOUNT & TRADING
# ============================================================

@app.route('/api/account')
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


@app.route('/api/trade', methods=['POST'])
@login_required
def execute_trade():
    """Execute a manual trade for the user."""
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


@app.route('/api/panic-sell', methods=['POST'])
@login_required
def panic_sell():
    """Close all open positions immediately for the user."""
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

        return jsonify({
            'success': True,
            'message': f'Panic Sell executed. Closed {closed_count} positions.',
            'closed_count': closed_count
        })
    except Exception as e:
        logger.error(f"Panic sell failed for user {current_user.id}: {e}")
        return jsonify({'success': False, 'error': str(e)})


# ============================================================
# BOT EXECUTION ENGINE (MULTI-THREADED)
# ============================================================

def bot_execution_loop(bot_id):
    """Execution loop for a specific trading bot."""
    bot = bot_manager.bots.get(bot_id)
    if not bot:
        logger.error(f"Execution loop failed: Bot {bot_id} not found")
        return

    config = bot.config
    symbol = config.symbol
    interval = config.interval
    strategy = config.strategy
    
    logger.info(f"🚀 Bot execution starting: {bot_id} ({symbol}, {strategy})")
    
    while not bot.stop_flag.is_set():
        try:
            # Check Global Pause
            if system_state.is_paused():
                logger.info(f"⏸️ System Paused. Bot {bot_id} sleeping...")
                time.sleep(5)
                continue
                
            # Check if bot still exists in manager (wasn't deleted)
            if bot_id not in bot_manager.bots:
                break

            # Fetch fresh data
            if config.market == 'crypto':
                df = crypto_provider.get_historical_klines(
                    symbol=symbol,
                    interval=interval,
                    limit=60
                )
                price_data = crypto_provider.get_current_price(symbol)
            else:
                df = stock_provider.get_historical_data(
                    symbol=symbol,
                    interval=interval,
                    limit=60
                )
                price_data = stock_provider.get_current_quote(symbol)
            
            if df.empty or len(df) < 52:
                logger.warning(f"⚠️ Bot {bot_id} skipped analysis: Insufficient data ({len(df)} candles, need 52)")
                time.sleep(10)
                continue
            
            current_price = price_data.get('price', 0)
            if current_price <= 0:
                logger.warning(f"⚠️ Bot {bot_id} skipped: Invalid price {current_price}")
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
            strategy_engine.min_confluence = 2
            if bot_id not in bot_manager.bots:
                logger.warning(f"⚠️ Bot {bot_id} no longer in manager. Exiting loop.")
                break
                
            current_strat = bot.config.strategy
            signal = strategy_engine.analyze(df, strategy=current_strat)
            
            # Update bot stats
            bot.stats.last_price = current_price
            bot.stats.last_signal = signal.signal
            bot.stats.signals_generated += 1

            # Store signal
            signal_data = {
                'time': datetime.now().isoformat(),
                'signal': signal.signal,
                'strength': signal.strength,
                'price': current_price,
                'strategy': current_strat,
                'reasons': signal.reasons[:3],
                'bot_id': bot_id,
                'symbol': symbol
            }
            
            # Emit signal to user room
            socketio.emit('auto_trade_signal', signal_data, room=f"user_{user_id}")
            
            # 1. Check for TP/SL Exit Conditions (Proactive Exit)
            symbol_pos = next((p for p in paper_trader.get_positions(user_id) if p['symbol'] == symbol), None)
            if symbol_pos:
                pnl_pct = symbol_pos.get('unrealized_pnl_pct', 0)
                tp_pct = bot.config.take_profit
                sl_pct = bot.config.stop_loss
                
                if pnl_pct >= tp_pct:
                    logger.info(f"🎯 TAKE PROFIT Triggered: {symbol} at {pnl_pct:.2f}% for user {user_id}")
                    close_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Take Profit Hit']})
                    execute_bot_trade(bot, close_signal, current_price)
                elif pnl_pct <= -sl_pct:
                    logger.info(f"🛑 STOP LOSS Triggered: {symbol} at {pnl_pct:.2f}% for user {user_id}")
                    close_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Stop Loss Hit']})
                    execute_bot_trade(bot, close_signal, current_price)
                else:
                    execute_bot_trade(bot, signal, current_price)
            else:
                execute_bot_trade(bot, signal, current_price)
            
            # Sleep (check every 5 seconds)
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in bot {bot_id} loop: {e}")
            time.sleep(10)
    
    logger.info(f"🛑 Bot execution stopped: {bot_id}")

def execute_bot_trade(bot, signal, current_price):
    """Execute trades for a specific bot."""
    symbol = bot.config.symbol
    user_id = bot.config.user_id
    
    # Check current positions for this user
    positions = paper_trader.get_positions(user_id)
    has_long = symbol in [p['symbol'] for p in positions if p['side'] == 'LONG' and p['quantity'] > 0]
    has_short = symbol in [p['symbol'] for p in positions if p['side'] == 'SHORT']
    
    # Debug logging for every signal
    logger.info(f"📊 Trade Decision for user {user_id} | {symbol}: Signal={signal.signal}, HasLong={has_long}, HasShort={has_short}")
    
    # SAFETY: If both LONG and SHORT exist (illegal state), close both immediately
    if has_long and has_short:
        logger.warning(f"⚠️ ILLEGAL STATE: user {user_id} | {symbol} has BOTH LONG and SHORT. Closing both.")
        # Close the LONG
        long_pos = next((p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG'), None)
        if long_pos:
            order = order_manager.create_order(
                user_id=user_id,
                symbol=symbol,
                side='sell',
                quantity=long_pos['quantity']
            )
            if order_manager.submit_order(order):
                entry_price = long_pos.get('avg_price', current_price)
                pnl = (current_price - entry_price) * long_pos['quantity']
                bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
                emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl)
        
        # Cover the SHORT
        short_pos = next((p for p in positions if p['symbol'] == symbol and p['side'] == 'SHORT'), None)
        if short_pos:
            order = order_manager.create_order(
                user_id=user_id,
                symbol=symbol,
                side='buy',
                quantity=short_pos['quantity']
            )
            if order_manager.submit_order(order):
                pnl = (short_pos['avg_price'] - current_price) * short_pos['quantity']
                bot_manager.increment_trades(bot.bot_id, 'buy', pnl)
                emit_trade_event(bot, 'COVER SHORT', short_pos['quantity'], current_price, pnl)
        return  # Exit to let state settle
    
    if signal.signal == 'BUY' and has_short:
        logger.warning(f"⚠️ {symbol} has SHORT while processing BUY. Closing SHORT first.")
        short_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'SHORT')
        order = order_manager.create_order(
            user_id=user_id,
            symbol=symbol,
            side='buy',
            quantity=short_pos['quantity']
        )
        if order_manager.submit_order(order):
            pnl = (short_pos['avg_price'] - current_price) * short_pos['quantity']
            bot_manager.increment_trades(bot.bot_id, 'buy', pnl)
            emit_trade_event(bot, 'COVER SHORT', short_pos['quantity'], current_price, pnl)
        return 

    if signal.signal == 'SELL' and has_long:
        logger.warning(f"⚠️ {symbol} has LONG while processing SELL. Closing LONG first.")
        long_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG' and p['quantity'] > 0)
        pnl = (current_price - long_pos['avg_price']) * long_pos['quantity']
        order = order_manager.create_order(
            user_id=user_id,
            symbol=symbol,
            side='sell',
            quantity=long_pos['quantity']
        )
        if order_manager.submit_order(order):
            bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
            emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl)
        return 

    if signal.signal == 'BUY':
        if has_long:
            logger.debug(f"⏸️ BUY skipped - already have long position for {symbol}")
        else:
            # Open new LONG
            logger.info(f"🚀 {bot.bot_id} executing LONG for {symbol} @ {current_price}")
            account = paper_trader.get_account_info(user_id)
            trade_value = account['buying_power'] * (bot.config.position_size / 100)
            quantity = min(trade_value / current_price, bot.config.max_quantity)
            
            if quantity > 0:
                order = order_manager.create_order(
                    user_id=user_id,
                    symbol=symbol,
                    side='buy',
                    quantity=quantity
                )
                if order_manager.submit_order(order):
                    bot_manager.increment_trades(bot.bot_id, 'buy', 0)
                    emit_trade_event(bot, 'LONG BUY', quantity, current_price, 0)

    elif signal.signal == 'SELL':
        if has_short:
            logger.debug(f"⏸️ SELL skipped - already have short position for {symbol}")
        elif has_long:
            # handled above, but just in case
            return
        else:
            # Open new SHORT
            logger.info(f"🚀 {bot.bot_id} executing SHORT for {symbol} @ {current_price}")
            account = paper_trader.get_account_info(user_id)
            trade_value = account['buying_power'] * (bot.config.position_size / 100)
            quantity = min(trade_value / current_price, bot.config.max_quantity)
            
            if quantity > 0:
                order = order_manager.create_order(
                    user_id=user_id,
                    symbol=symbol,
                    side='sell',
                    quantity=quantity
                )
                if order_manager.submit_order(order):
                    bot_manager.increment_trades(bot.bot_id, 'sell', 0)
                    emit_trade_event(bot, 'SHORT SELL', quantity, current_price, 0)

def emit_trade_event(bot, side, quantity, price, pnl=0):
    user_id = bot.config.user_id
    trade_msg = {
        'type': 'trade',
        'side': side,
        'symbol': bot.config.symbol,
        'quantity': quantity,
        'price': price,
        'pnl': pnl,
        'strategy': bot.config.strategy,
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


@app.route('/api/auto-trade/start', methods=['POST'])
def start_auto_trade():
    """Start live auto-trading."""
    global live_auto_trading, live_auto_thread, auto_trade_stats
    global current_symbol, current_interval, current_strategy, current_market
    global auto_trade_settings, strategy_engine
    
    if live_auto_trading:
        return jsonify({'success': False, 'error': 'Already running'})
    
    data = request.json or {}
    current_symbol = data.get('symbol', current_symbol)
    current_interval = data.get('interval', current_interval)
    current_strategy = data.get('strategy', current_strategy)
    current_market = data.get('market', current_market)
    
    # Update settings
    settings = data.get('settings', {})
    if settings:
        auto_trade_settings.update({
            'confluence': settings.get('confluence', 3),
            'position_size': settings.get('positionSize', 10),
            'check_interval': settings.get('checkInterval', 5),
            'stop_loss': settings.get('stopLoss', 5),
            'take_profit': settings.get('takeProfit', 10)
        })
    
    # Reset stats
    # Initialize Logger
    # Reset stats
    auto_trade_stats = {
        'total_trades': 0,
        'buy_trades': 0,
        'sell_trades': 0,
        'total_pnl': 0,
        'signals': [],
        'start_time': None,
        'trades_log': []
    }
    
    live_auto_trading = True
    live_auto_thread = threading.Thread(target=live_auto_trade_loop, daemon=True)
    live_auto_thread.start()
    
    return jsonify({
        'success': True, 
        'message': f'GodBotTrade started: {current_market.upper()} - {current_symbol} - {current_strategy}'
    })


@app.route('/api/auto-trade/stop', methods=['POST'])
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


@app.route('/api/auto-trade/report')
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


@app.route('/api/auto-trade/status')
@login_required
def auto_trade_status():
    """Get live auto-trade status for the user."""
    acc = paper_trader.get_account_info(current_user.id)
    summary = trade_logger.get_daily_summary(current_user.id, datetime.now().strftime("%Y-%m-%d"))
    
    # Check if user has any active bots
    user_bots = bot_manager.get_all_bots(current_user.id)
    any_running = any(bot_manager.is_running(b.bot_id) for b in user_bots)
    
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


@app.route('/api/report/download')
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
        from src.engine.bot_manager import get_bot_manager
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
                    # logger.debug(f"📡 Emitted price_update for {symbol}: {price}")
                    
                    
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
            
            time.sleep(1)
            
        except Exception as e:
            logger.error(f"Stream outer error: {e}")
            time.sleep(2)


# ============================================================
# BOT MANAGEMENT ROUTES
# ============================================================

@app.route('/api/bots', methods=['GET'])
@login_required
def list_bots():
    """List all active bots for the current user."""
    bots = bot_manager.get_all_bots(current_user.id)
    running_bots = [b.to_dict() for b in bots if bot_manager.is_running(b.bot_id)]
    logger.info(f"📊 User {current_user.id} active bots: {len(running_bots)}")
    return jsonify({
        'success': True,
        'bots': running_bots,
        'running_count': len(running_bots)
    })

@app.route('/api/bots/start', methods=['POST'])
@login_required
def start_bot():
    """Start a new trading bot for the user."""
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

@app.route('/api/bots/<bot_id>/stop', methods=['POST'])
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

@app.route('/api/bots/stop-all', methods=['POST'])
def stop_all_bots():
    """Stop all active bots."""
    bot_manager.stop_all()
    return jsonify({'success': True, 'message': 'All bots stopped'})

@app.route('/api/bots/<bot_id>/strategy', methods=['PUT'])
def update_bot_strategy(bot_id):
    """Update strategy for a running bot."""
    data = request.json
    new_strategy = data.get('strategy')
    
    if bot_manager.update_bot_config(bot_id, strategy=new_strategy):
        return jsonify({'success': True, 'strategy': new_strategy})
    return jsonify({'success': False, 'error': 'Bot not found'})


@socketio.on('connect')
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
    for cfg in configs:
        user_id = cfg.get('user_id', 1)
        bot_id = cfg.get('id')
        
        # Check if bot should be running (status was 'running' or auto_restart is true)
        if cfg.get('status') == 'running' or cfg.get('auto_restart_enabled', 0):
            logger.info(f"🚀 Auto-restoring bot: {bot_id}")
            
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
                logger.info(f"✅ Restored bot {bot_id} (Thread started)")
            else:
                logger.error(f"❌ Failed to restore bot {bot_id}: {result.get('error')}")


if __name__ == '__main__':
    print("⚡ GodBotTrade Server starting...")
    print("📊 Open http://localhost:5050 in your browser")
    print("🪙 Crypto: 24/7 live data from Binance")
    print("📈 Stocks: Yahoo Finance (market hours)")
    
    # Restore any previously running bots
    restore_bots_on_startup()
    
    # Start the bot watchdog in a background thread
    watchdog_thread = threading.Thread(target=bot_watchdog_loop, daemon=True)
    watchdog_thread.start()
    
    # Get port from environment variable for cloud deployment (Railway/Heroku/etc)
    port = int(os.getenv('PORT', 5050))
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)

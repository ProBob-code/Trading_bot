"""
GodBotTrade API Server
======================

Multi-asset trading platform API with WebSocket support.
Supports: Crypto (24/7) and Stocks (market hours)
"""

from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from src.services.trade_logger import get_trade_logger
from src.services.system_state import get_system_state  # <--- SystemState import
import threading
import time
import json
from datetime import datetime
from pathlib import Path
import sys
from loguru import logger

# Configure logging for maximum visibility
logger.remove()
logger.add(
    sys.stdout, 
    level="DEBUG", 
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)
logger.add("trading_bot.log", rotation="10 MB", level="DEBUG")

# Import our modules
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.data.crypto_provider import BinanceCryptoProvider
from src.data.stock_provider import YahooFinanceProvider
from src.execution.brokers.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager
from src.strategies.strategy_engine import StrategyEngine, get_strategy_engine
from src.engine.bot_manager import get_bot_manager, BotManager

# Initialize Flask
app = Flask(__name__, static_folder='web', static_url_path='')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

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
def get_auto_trade_status():
    """Get current auto-trading status and trade counts."""
    return jsonify({
        'total_trades': auto_trade_stats['total_trades'],
        'buy_trades': auto_trade_stats['buy_trades'],
        'sell_trades': auto_trade_stats['sell_trades'],
        'total_pnl': auto_trade_stats['total_pnl'],
        'start_time': auto_trade_stats['start_time'],
        'active_bots': len(bot_manager.get_running_bots())
    })

@app.route('/')
def index():
    """Serve the main frontend."""
    return send_from_directory('web', 'index.html')


@app.route('/login.html')
def login():
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
def get_reports_legacy():
    """Get all daily reports (Legacy Endpoint - redirects to new service)."""
    # For backward compatibility, return format similar to old endpoint but from new service
    reports = []
    today = datetime.now().strftime("%Y-%m-%d")
    
    try:
        # Get today's summary
        summary = trade_logger.get_daily_summary(today)
        
        # Convert to expected format
        if summary['total_trades'] > 0:
            win_rate = (summary['wins'] / summary['total_trades'] * 100)
            avg_profit = (summary['total_pnl'] / summary['total_trades'])
            
            reports.append({
                'date': today,
                'user': 'system', # aggregated
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
def get_all_trades():
    """Get all trades with filters."""
    # Accept 'date' as alias for start_date+end_date (used by report.html)
    date_filter = request.args.get('date')
    start_date = request.args.get('start_date', date_filter)
    end_date = request.args.get('end_date', date_filter)
    symbol = request.args.get('symbol')
    limit = int(request.args.get('limit', 100))
    
    trades = trade_logger.get_history(start_date=start_date, end_date=end_date, symbol=symbol, limit=limit)
    return jsonify(trades)

@app.route('/api/reports/summary')
def get_daily_summary():
    """Get summarized daily stats."""
    date_str = request.args.get('date', datetime.now().strftime("%Y-%m-%d"))
    summary = trade_logger.get_daily_summary(date_str)
    return jsonify(summary)

@app.route('/api/positions')
def get_positions():
    """Get all positions and trade history."""
    positions = paper_trader.get_positions()
    
    # Build open positions with current prices
    open_positions = []
    for pos in positions:
        symbol = pos['symbol']
        current_price = pos.get('current_price', pos['avg_price'])
        
        open_positions.append({
            'symbol': symbol,
            'side': 'LONG' if pos['quantity'] > 0 else 'SHORT',
            'qty': abs(pos['quantity']),
            'avg_price': pos['avg_price'],
            'current_price': current_price,
            'net_pnl': pos.get('unrealized_pnl', 0),
            'open_interest': 0  # Placeholder
        })
    
    # Add short positions
    for symbol, short in paper_trader.short_positions.items():
        open_positions.append({
            'symbol': symbol,
            'side': 'SHORT',
            'qty': short['quantity'],
            'avg_price': short['entry_price'],
            'current_price': short.get('current_price', short['entry_price']),
            'net_pnl': short.get('unrealized_pnl', 0),
            'open_interest': 0
        })
    
    # Filter closed positions: Only show if the bot that opened/closed them is STOPPED
    all_closed = paper_trader.get_closed_positions()
    filtered_closed = []
    
    for pos in all_closed:
        # Assuming pos has bot_id or we check if a bot for this symbol is running
        symbol = pos['symbol']
        bot_id = f"{current_market}_{symbol}".lower()
        
        # If bot is not running, show it in closed
        if not bot_manager.is_running(bot_id):
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
        'trade_history': auto_trade_stats.get('trades_log', []),
        'journal': auto_trade_stats.get('journal', []),
        'pending_orders': []
    })


@app.route('/api/balance', methods=['POST'])
def update_balance():
    """Update paper trading balance."""
    data = request.json
    # Accept either 'cash' or 'balance' key for compatibility
    new_balance = data.get('cash') or data.get('balance', 100000)
    
    try:
        # Update paper trader's cash and initial_capital to reset P&L relative to new balance
        paper_trader.cash = float(new_balance)
        paper_trader.initial_capital = float(new_balance)
        logger.info(f"💰 Paper trading balance reset to: ${new_balance:,.2f}")
        
        return jsonify({
            'success': True,
            'balance': paper_trader.cash,
            'message': f'Balance updated to ${new_balance:,.2f}'
        })
    except Exception as e:
        logger.error(f"Error updating balance: {e}")
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
        return jsonify([])
    
    candles = []
    for idx, row in df.iterrows():
        candles.append({
            'time': int(idx.timestamp()),
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
        })
    
    return jsonify(candles)


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
            'open': row['open'],
            'high': row['high'],
            'low': row['low'],
            'close': row['close'],
            'volume': row['volume']
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
def execute_trade():
    """Execute a trade."""
    global auto_trade_stats
    
    data = request.json
    symbol = data.get('symbol', current_symbol)
    side = data.get('side', 'buy')
    quantity = float(data.get('quantity', 0))
    market = data.get('market', current_market)
    
    if quantity <= 0:
        return jsonify({'success': False, 'error': 'Invalid quantity'})
    
    # Get current price
    if market == 'crypto':
        price_data = crypto_provider.get_current_price(symbol)
    else:
        price_data = stock_provider.get_current_quote(symbol)
    
    price = price_data.get('price', 0)
    
    if price <= 0:
        return jsonify({'success': False, 'error': 'Could not get price'})
    
    # Update paper trader price
    paper_trader.set_prices({symbol: price})
    
    # Create and execute order
    order = order_manager.create_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type='market'
    )
    
    success = order_manager.submit_order(order)
    
    if success:
        # Update trade counter
        user = data.get('user', 'GodBot')
        auto_trade_stats['total_trades'] += 1
        
        # Log trade with standardized fields
        trade_log = {
            'timestamp': datetime.now().isoformat(),
            'symbol': symbol,
            'side': side.upper(),
            'quantity': quantity,
            'price': price,
            'pnl': 0, # Manual trades start with 0 P&L until closed (simplified)
            'value': quantity * price,
            'strategy': 'manual',
            'user': user,
            'market': market
        }
        auto_trade_stats['trades_log'].append(trade_log)
        
        # Update daily summary report -> Log via TradeLogger
        trade_logger.log_trade(
            symbol=symbol,
            side=side,
            quantity=quantity,
            price=price,
            pnl=0, # Manual opening trade usually 0 PnL
            strategy='manual',
            bot_id='manual_control',
            mode='paper', # Assuming paper for now
            account_value=paper_trader.get_account_info()['total_value'],
            notes=f"Manual {side} Trade"
        )
        
        if side.upper() == 'BUY':
            auto_trade_stats['buy_trades'] += 1
        else:
            auto_trade_stats['sell_trades'] += 1
        
        return jsonify({
            'success': True,
            'order_id': order.order_id,
            'symbol': symbol,
            'side': side.upper(),
            'quantity': quantity,
            'price': price,
            'value': quantity * price,
            'total_trades': auto_trade_stats['total_trades']
        })
    else:
        return jsonify({'success': False, 'error': 'Order failed'})


@app.route('/api/panic-sell', methods=['POST'])
def panic_sell():
    """Close all open positions immediately."""
    try:
        positions = paper_trader.get_positions()
        closed_count = 0
        
        for pos in positions:
            symbol = pos['symbol']
            quantity = abs(pos['quantity'])
            side = 'sell' if pos['quantity'] > 0 else 'buy'
            
            order = order_manager.create_order(
                symbol=symbol,
                side=side,
                quantity=quantity,
                order_type='market'
            )
            if order_manager.submit_order(order):
                closed_count += 1
                logger.warning(f"🚨 PANIC SELL: Closed {quantity} {symbol}")
                
                # Log the panic sell
                trade_logger.log_trade(
                    symbol=symbol,
                    side=side,
                    quantity=quantity,
                    price=pos.get('current_price', 0),
                    pnl=pos.get('unrealized_pnl', 0), # Realized effectively
                    strategy='panic_sell',
                    bot_id='panic_button',
                    mode='paper',
                    account_value=paper_trader.get_account_info()['total_value'],
                    notes="Panic Sell Triggered"
                )
        
        # Also handle shorts not in positions list
        for symbol, short in list(paper_trader.short_positions.items()):
             order = order_manager.create_order(
                symbol=symbol,
                side='buy',
                quantity=short['quantity'],
                order_type='market'
            )
             if order_manager.submit_order(order):
                closed_count += 1
                logger.warning(f"🚨 PANIC COVER: Closed short {symbol}")
                
                # Log the panic cover
                trade_logger.log_trade(
                    symbol=symbol,
                    side='buy', # Cover is a buy
                    quantity=short['quantity'],
                    price=short.get('current_price', 0),
                    pnl=short.get('unrealized_pnl', 0),
                    strategy='panic_sell',
                    bot_id='panic_button',
                    mode='paper',
                    account_value=paper_trader.get_account_info()['total_value'],
                    notes="Panic Cover Triggered"
                )

        return jsonify({
            'success': True,
            'message': f'Panic Sell executed. Closed {closed_count} positions.',
            'closed_count': closed_count
        })
    except Exception as e:
        logger.error(f"Panic sell failed: {e}")
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
            
            # Update Shared Paper Trader Prices
            paper_trader.set_prices({symbol: current_price})
            
            # Calculate Unrealized P&L for this bot
            positions = paper_trader.get_positions()
            symbol_pos = next((p for p in positions if p['symbol'] == symbol), None)
            
            if symbol_pos:
                bot.stats.unrealized_pnl = symbol_pos.get('net_pnl', 0)
            else:
                bot.stats.unrealized_pnl = 0
            
            bot.stats.total_pnl = bot.stats.realized_pnl + bot.stats.unrealized_pnl

            # Analyze with selected strategy
            # Lower confluence for more active trading (2 instead of 3)
            strategy_engine.min_confluence = 2  # More active: 2 indicators needed vs 3
            # Note: We use the strategy from bot.config which can be hot-swapped
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
            
            # Emit signal to frontend
            socketio.emit('auto_trade_signal', signal_data)
            
            # 3. Log to Audit Journal (only for BUY/SELL actions or significant signals)
            if signal.signal in ['BUY', 'SELL']:
                auto_trade_stats['journal'].append({
                    'time': signal_data['time'],
                    'symbol': signal_data['symbol'],
                    'side': signal_data['signal'],
                    'price': signal_data['price'],
                    'qty': bot.config.position_size, # Log target Qty
                    'strategy': signal_data['strategy'],
                    'reasons': signal_data['reasons']
                })
                # Keep journal at reasonable size
                if len(auto_trade_stats['journal']) > 100:
                    auto_trade_stats['journal'] = auto_trade_stats['journal'][-100:]
            
            # 1. Check for TP/SL Exit Conditions (Proactive Exit)
            if symbol_pos:
                pnl_pct = symbol_pos.get('pnl_pct', 0)
                tp_pct = bot.config.take_profit
                sl_pct = bot.config.stop_loss
                
                if pnl_pct >= tp_pct:
                    logger.info(f"🎯 TAKE PROFIT Triggered: {symbol} at {pnl_pct:.2f}%")
                    # Send special "CLOSE" signal
                    close_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Take Profit Hit']})
                    execute_bot_trade(bot, close_signal, current_price)
                elif pnl_pct <= -sl_pct:
                    logger.info(f"🛑 STOP LOSS Triggered: {symbol} at {pnl_pct:.2f}%")
                    close_signal = type('Signal', (), {'signal': 'SELL' if symbol_pos['quantity'] > 0 else 'BUY', 'strength': 1.0, 'reasons': ['Stop Loss Hit']})
                    execute_bot_trade(bot, close_signal, current_price)
                else:
                    # 2. Normal Strategy Execution
                    execute_bot_trade(bot, signal, current_price)
            else:
                # 2. Normal Strategy Execution
                execute_bot_trade(bot, signal, current_price)
            
            # Sleep (check every 5 seconds)
            time.sleep(5)
            
        except Exception as e:
            logger.error(f"Error in bot {bot_id} loop: {e}")
            time.sleep(10)
    
    logger.info(f"🛑 Bot execution stopped: {bot_id}")

def execute_bot_trade(bot, signal, current_price):
    """Execute trades for a specific bot."""
    global auto_trade_stats
    symbol = bot.config.symbol
    
    # Check current positions in shared paper trader
    positions = paper_trader.get_positions()
    has_long = symbol in [p['symbol'] for p in positions if p['side'] == 'LONG' and p['quantity'] > 0]
    has_short = symbol in paper_trader.short_positions
    
    # Debug logging for every signal
    logger.info(f"📊 Trade Decision for {symbol}: Signal={signal.signal}, HasLong={has_long}, HasShort={has_short}")
    
    # SAFETY: If both LONG and SHORT exist (illegal state), close both immediately
    if has_long and has_short:
        logger.warning(f"⚠️ ILLEGAL STATE: {symbol} has BOTH LONG and SHORT. Closing both.")
        # Close the LONG
        long_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG')
        sell_order = order_manager.create_order(symbol=symbol, side='sell', quantity=long_pos['quantity'], order_type='market')
        if order_manager.submit_order(sell_order):
            pnl = (current_price - long_pos['entry_price']) * long_pos['quantity']
            bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
            auto_trade_stats['total_trades'] += 1
            emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl)
        # Cover the SHORT
        short_pos = paper_trader.short_positions[symbol]
        cover_order = order_manager.create_order(symbol=symbol, side='buy', quantity=short_pos['quantity'], order_type='market')
        if order_manager.submit_order(cover_order):
            pnl = (short_pos['entry_price'] - current_price) * short_pos['quantity']
            bot_manager.increment_trades(bot.bot_id, 'buy', pnl)
            auto_trade_stats['total_trades'] += 1
            emit_trade_event(bot, 'COVER SHORT', short_pos['quantity'], current_price, pnl)
        return  # Exit to let state settle
    
    # STRICT EXCLUSIVITY: If we hold BOTH (bug state), or the WRONG side, close it first.
    if signal.signal == 'BUY' and has_short:
        logger.warning(f"⚠️ {symbol} has SHORT while processing BUY. Closing SHORT first.")
        short_pos = paper_trader.short_positions[symbol]
        short_qty = short_pos['quantity']
        pnl = (short_pos['entry_price'] - current_price) * short_qty
        cover_order = order_manager.create_order(symbol=symbol, side='buy', quantity=short_qty, order_type='market')
        if order_manager.submit_order(cover_order):
            bot_manager.increment_trades(bot.bot_id, 'buy', pnl)
            auto_trade_stats['total_trades'] += 1
            auto_trade_stats['buy_trades'] += 1
            logger.info(f"✅ Covered SHORT for {symbol} P&L: ${pnl:.2f}")
            emit_trade_event(bot, 'COVER SHORT', short_qty, current_price, pnl)
        return # Exit to let state settle

    if signal.signal == 'SELL' and has_long:
        logger.warning(f"⚠️ {symbol} has LONG while processing SELL. Closing LONG first.")
        long_pos = next(p for p in positions if p['symbol'] == symbol and p['side'] == 'LONG' and p['quantity'] > 0)
        pnl = (current_price - long_pos['entry_price']) * long_pos['quantity']
        sell_order = order_manager.create_order(symbol=symbol, side='sell', quantity=long_pos['quantity'], order_type='market')
        if order_manager.submit_order(sell_order):
            bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
            auto_trade_stats['total_trades'] += 1
            auto_trade_stats['sell_trades'] += 1
            logger.info(f"✅ Closed LONG for {symbol} P&L: ${pnl:.2f}")
            emit_trade_event(bot, 'CLOSE LONG', long_pos['quantity'], current_price, pnl)
        return # Exit to let state settle

    if signal.signal == 'BUY':
        if has_long:
            logger.debug(f"⏸️ BUY skipped - already have long position for {symbol}")
        else:
            # Open new LONG
            logger.info(f"🚀 {bot.bot_id} executing LONG for {symbol} @ {current_price}")
            account = paper_trader.get_account_info()
            # Use buying_power (cash minus short margin) to prevent exponential sizing
            equity = max(0, account['buying_power'])
            # Safety cap: never size beyond initial capital
            equity = min(equity, paper_trader.initial_capital)
            trade_value = equity * (bot.config.position_size / 100)
            quantity = trade_value / current_price
            
            # Apply max quantity limit
            max_qty = getattr(bot.config, 'max_quantity', 1.0)
            quantity = min(quantity, max_qty)
            
            if quantity > 0:
                order = order_manager.create_order(symbol=symbol, side='buy', quantity=quantity, order_type='market')
                if order_manager.submit_order(order):
                    # Calculate realized P&L if covering a short
                    pnl = 0
                    if has_short:
                        short_pos = paper_trader.short_positions.get(symbol)
                        if short_pos:
                            pnl = (short_pos['entry_price'] - current_price) * short_pos['quantity']
                    
                    bot_manager.increment_trades(bot.bot_id, 'buy', pnl)
                    auto_trade_stats['total_trades'] += 1
                    auto_trade_stats['buy_trades'] += 1
                    logger.info(f"✅ EXECUTED: LONG BUY {quantity:.6f} {symbol} @ ${current_price:.2f}")
                    emit_trade_event(bot, 'LONG BUY', quantity, current_price, pnl)

    elif signal.signal == 'SELL':
        if has_short:
            logger.debug(f"⏸️ SELL skipped - already have short position for {symbol}")
        elif has_long:
            # FORCE SELL LONG before going SHORT
            logger.info(f"🔄 Symbol {symbol} has LONG. Closing LONG before opening SHORT.")
            long_pos = next(p for p in positions if p['symbol'] == symbol and p['quantity'] > 0)
            sell_order = order_manager.create_order(symbol=symbol, side='sell', quantity=long_pos['quantity'], order_type='market')
            if order_manager.submit_order(sell_order):
                # Calculate realized P&L for closure (manual P&L calculation as fallback)
                pnl = (current_price - long_pos['entry_price']) * long_pos['quantity']
                bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
                auto_trade_stats['total_trades'] += 1
                logger.info(f"✅ Closed LONG for {symbol} with P&L: ${pnl:.2f}")
            
            # Wait for next loop for cleaner state transition
            return
        else:
            # Open new SHORT
            logger.info(f"🚀 {bot.bot_id} executing SHORT for {symbol} @ {current_price}")
            account = paper_trader.get_account_info()
            # Use buying_power (cash minus short margin) to prevent exponential sizing
            equity = max(0, account['buying_power'])
            # Safety cap: never size beyond initial capital
            equity = min(equity, paper_trader.initial_capital)
            trade_value = equity * (bot.config.position_size / 100)
            quantity = trade_value / current_price
            
            # Apply max quantity limit
            max_qty = getattr(bot.config, 'max_quantity', 1.0)
            quantity = min(quantity, max_qty)
            
            if quantity > 0:
                order = order_manager.create_order(symbol=symbol, side='sell', quantity=quantity, order_type='market')
                if order_manager.submit_order(order):
                    # Calculate realized P&L if closing a long
                    pnl = 0
                    if has_long:
                        long_pos = next((p for p in positions if p['symbol'] == symbol and p['quantity'] > 0), None)
                        if long_pos:
                            pnl = (current_price - long_pos['entry_price']) * long_pos['quantity']
                    
                    bot_manager.increment_trades(bot.bot_id, 'sell', pnl)
                    auto_trade_stats['total_trades'] += 1
                    auto_trade_stats['sell_trades'] += 1
                    logger.info(f"✅ EXECUTED: SHORT SELL {quantity:.6f} {symbol} @ ${current_price:.2f}")
                    emit_trade_event(bot, 'SHORT SELL', quantity, current_price, pnl)

def emit_trade_event(bot, side, quantity, price, pnl=0):
    # Calculate realized P&L if closing a position and pnl not provided
    if pnl == 0:
        if ('SELL' in side and 'LONG' in side) or ('BUY' in side and 'COVER' in side):
            if paper_trader.closed_positions:
                last_closed = paper_trader.closed_positions[-1]
                if last_closed['symbol'] == bot.config.symbol:
                    pnl = last_closed['realized_pnl']

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
    
    # Persistent log for reports
    auto_trade_stats['trades_log'].append(trade_msg)
    
    # Update Daily report -> Log via TradeLogger
    trade_logger.log_trade(
        symbol=bot.config.symbol,
        side=side,
        quantity=quantity,
        price=price,
        pnl=pnl,
        strategy=bot.config.strategy,
        bot_id=bot.bot_id,
        mode=bot.config.mode.value if hasattr(bot.config, 'mode') and hasattr(bot.config.mode, 'value') else 'paper',
        account_value=paper_trader.get_account_info()['total_value'],
        notes=f"Auto Bot Trade {side}"
    )
    
    # Update session P&L
    auto_trade_stats['total_pnl'] += pnl
    
    socketio.emit('auto_trade_executed', trade_msg)

def start_bot_thread(bot_id):
    """Helper to start a bot thread."""
    thread = threading.Thread(target=bot_execution_loop, args=(bot_id,), daemon=True)
    thread.start()
    if bot_id in bot_manager.bots:
        bot_manager.bots[bot_id].thread = thread

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


@app.route('/api/auto-trade/status')
def auto_trade_status():
    """Get live auto-trade status."""
    account = paper_trader.get_account_info()
    
    return jsonify({
        'running': live_auto_trading,
        'market': current_market,
        'symbol': current_symbol,
        'strategy': current_strategy,
        'total_trades': auto_trade_stats['total_trades'],
        'buy_trades': auto_trade_stats['buy_trades'],
        'sell_trades': auto_trade_stats['sell_trades'],
        'current_pnl': account['pnl'],
        'signals': auto_trade_stats['signals'][-10:]
    })


@app.route('/api/report/download')
def download_report():
    """Generate downloadable trading report."""
    account = paper_trader.get_account_info()
    
    report = {
        'generated_at': datetime.now().isoformat(),
        'account': {
            'initial_capital': INITIAL_CAPITAL,
            'current_balance': account['total_value'],
            'total_pnl': account['pnl'],
            'roi_percent': account['pnl_pct']
        },
        'trading_session': {
            'market': current_market,
            'symbol': current_symbol,
            'strategy': current_strategy,
            'total_trades': auto_trade_stats['total_trades'],
            'buy_trades': auto_trade_stats['buy_trades'],
            'sell_trades': auto_trade_stats['sell_trades']
        },
        'trades_log': auto_trade_stats['trades_log'],
        'signals_log': auto_trade_stats['signals'][-50:]
    }
    
    return jsonify(report)


# ============================================================
# WEBSOCKET - REAL-TIME STREAMING
# ============================================================

def price_stream():
    """Background thread to stream prices."""
    global is_streaming, current_symbol, current_market
    
    count = 0
    while is_streaming:
        try:
            count += 1
            if count % 10 == 0:
                logger.debug(f"📡 Price stream heartbeat ({current_symbol})")
            
            # Determine which provider to use. 
            # We are robustness-first: if the symbol is in the known stock list, use stock_provider.
            symbol_is_stock = current_symbol in ['AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN']
            
            if current_market == 'crypto' and not symbol_is_stock:
                price_data = crypto_provider.get_current_price(current_symbol)
                ticker = crypto_provider.get_ticker_24h(current_symbol)
            else:
                price_data = stock_provider.get_current_quote(current_symbol)
                ticker = {
                    'price_change': price_data.get('change', 0),
                    'price_change_pct': price_data.get('change_pct', 0),
                    'high_24h': price_data.get('high', 0),
                    'low_24h': price_data.get('low', 0),
                    'volume_24h': price_data.get('volume', 0)
                }
            
            # Update paper trader for MAIN focus symbol
            if price_data.get('price', 0) > 0:
                paper_trader.set_prices({current_symbol: price_data['price']})
            
            # Update prices for ALL ACTIVE BOTS (so their P&L updates on cards)
            active_bots = bot_manager.get_running_bots()
            other_symbols = set(bot['symbol'] for bot in active_bots if bot['symbol'] != current_symbol)
            
            for s in other_symbols:
                bot_item = next(b for b in active_bots if b['symbol'] == s)
                if bot_item['market'] == 'crypto':
                    p = crypto_provider.get_current_price(s)
                else:
                    p = stock_provider.get_current_quote(s)
                
                if p.get('price', 0) > 0:
                    paper_trader.set_prices({s: p['price']})
            
            # Get account
            account = paper_trader.get_account_info()
            
            # Emit to all clients
            socketio.emit('price_update', {
                'symbol': current_symbol,
                'market': current_market,
                'price': price_data.get('price', 0),
                'change_24h': ticker.get('price_change', 0),
                'change_pct': ticker.get('price_change_pct', 0),
                'high_24h': ticker.get('high_24h', 0),
                'low_24h': ticker.get('low_24h', 0),
                'volume_24h': ticker.get('volume_24h', 0),
                'timestamp': datetime.now().isoformat(),
                'account': {
                    'cash': account['cash'],
                    'total_value': account['total_value'],
                    'pnl': account['pnl']
                }
            })
            
            time.sleep(1)  # Update every second
            
        except Exception as e:
            print(f"Stream error: {e}")
            time.sleep(1)


# ============================================================
# BOT MANAGEMENT ROUTES
# ============================================================

@app.route('/api/bots', methods=['GET'])
def list_bots():
    """List all active bots."""
    bots = bot_manager.get_running_bots()
    logger.info(f"📊 Active bots query: {len(bots)} running, total stored: {len(bot_manager.bots)}")
    return jsonify({
        'success': True,
        'bots': bots,
        'running_count': len(bots)
    })

@app.route('/api/bots/start', methods=['POST'])
def start_bot():
    """Start a new trading bot."""
    data = request.json
    try:
        # Extract settings from nested object
        settings = data.get('settings', {})
        
        # Call bot_manager.start_bot with keyword arguments (not BotConfig)
        result = bot_manager.start_bot(
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
            
            logger.info(f"✅ Bot started: {result['bot_id']}")
            return jsonify({'success': True, 'bot_id': result['bot_id']})
        else:
            return jsonify(result)
        
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/bots/<bot_id>/stop', methods=['POST'])
def stop_bot(bot_id):
    """Stop a specific bot and auto-close its positions."""
    try:
        # 1. Identify bot symbol before stopping
        bot = bot_manager.bots.get(bot_id)
        if bot:
            symbol = bot.config.symbol
            
            # 2. Check for open positions for this symbol
            positions = paper_trader.get_positions()
            symbol_pos = next((p for p in positions if p['symbol'] == symbol), None)
            
            if symbol_pos:
                logger.info(f"🛑 Bot {bot_id} stop requested. Auto-closing position for {symbol}")
                
                # Create a cover order
                qty = symbol_pos['quantity']
                side = 'sell' if qty > 0 else 'buy'
                
                # Log to journal BEFORE executing to ensure record exists
                # Also log via TradeLogger
                pnl = symbol_pos.get('realized_pnl', 0)
                trade_logger.log_trade(
                    symbol=symbol,
                    side='CLOSE',
                    quantity=abs(qty),
                    price=symbol_pos.get('current_price', 0),
                    pnl=pnl,
                    strategy=bot.config.strategy,
                    bot_id=bot.bot_id,
                    mode=bot.config.mode if hasattr(bot.config, 'mode') else 'paper',
                    notes="Stop Command Received - Auto Liquidating"
                )
                
                # Sync counters
                bot_manager.increment_trades(bot.bot_id, side, pnl)
                
                auto_trade_stats['journal'].append({
                    'time': datetime.now().isoformat(),
                    'symbol': symbol,
                    'side': 'CLOSE',
                    'price': symbol_pos.get('current_price', 0),
                    'qty': abs(qty),
                    'strategy': bot.config.strategy,
                    'reasons': ['Stop Command Received - Auto Liquidating']
                })
                
                # Execute closure via order manager
                from src.execution.order_manager import Order
                order = Order(symbol=symbol, side=side, quantity=abs(qty), order_type='market')
                paper_trader.submit_order(order)
                
                # Update total trades count precisely
                auto_trade_stats['total_trades'] = auto_trade_stats.get('total_trades', 0) + 1

        # 3. Stop the bot
        result = bot_manager.stop_bot(bot_id)
        
        # Persist updated bot configs to disk
        bot_manager.save_configs()
        
        return jsonify(result)
    except Exception as e:
        logger.error(f"Error during bot stop/closure: {e}")
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
    """Handle client connection."""
    global is_streaming, stream_thread
    
    print(f"Client connected to GodBotTrade. Stream active: {is_streaming}")
    
    if not is_streaming or stream_thread is None or not stream_thread.is_alive():
        logger.info("📡 Starting/Restarting price stream...")
        is_streaming = True
        stream_thread = threading.Thread(target=price_stream, daemon=True)
        stream_thread.start()
    
    emit('connected', {
        'status': 'ok', 
        'symbol': current_symbol,
        'market': current_market
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    print("Client disconnected")


@socketio.on('change_symbol')
def handle_symbol_change(data):
    """Change the trading symbol."""
    global current_symbol
    
    current_symbol = data.get('symbol', 'BTCUSDT')
    emit('symbol_changed', {'symbol': current_symbol})


@socketio.on('change_market')
def handle_market_change(data):
    """Change the trading market."""
    global current_market, current_symbol
    
    current_market = data.get('market', 'crypto')
    current_symbol = data.get('symbol', 'BTCUSDT' if current_market == 'crypto' else 'AAPL')
    
    emit('market_changed', {
        'market': current_market,
        'symbol': current_symbol
    })


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
    """Restore bots from saved configuration on server startup."""
    configs = bot_manager.load_configs()
    
    if not configs:
        logger.info("No saved bot configurations to restore")
        return
    
    for cfg in configs:
        if cfg.get('auto_restart_enabled', True):
            logger.info(f"🚀 Auto-restoring bot: {cfg.get('bot_id')}")
            
            result = bot_manager.start_bot(
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
                # Start the execution thread
                start_bot_thread(result['bot_id'])


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
    
    socketio.run(app, host='0.0.0.0', port=5050, debug=True, allow_unsafe_werkzeug=True)

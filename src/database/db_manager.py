import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional, Any
import sqlite3

class DatabaseManager:
    def __init__(self):
        self._detect_config()
        
        # Retry logic: Attempt to connect to MySQL 3 times before failing back to SQLite
        # This gives the MySQL container time to start up if running via docker-compose
        max_retries = 3
        retry_delay = 2
        
        self.use_sqlite = True
        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"🔌 database_manager: Connecting to MySQL (Attempt {attempt}/{max_retries})...")
                conn = mysql.connector.connect(
                    host=self.host,
                    port=self.port,
                    user=self.user,
                    password=self.password,
                    connect_timeout=5
                )
                conn.close()
                self.use_sqlite = False
                logger.info("✅ MySQL connection verified.")
                break
            except Error as e:
                # Handle specific errors for clearer feedback
                if e.errno == 1045:
                    logger.error(f"❌ MySQL Access Denied (Check credentials): {e}")
                elif e.errno == 2003:
                    logger.warning(f"⚠️ MySQL Host not reachable (Is it running?): {e}")
                else:
                    logger.warning(f"⚠️ MySQL connection attempt {attempt} failed: {e}")
                
                if attempt < max_retries:
                    logger.info(f"⏳ Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
        
        if self.use_sqlite:
            logger.warning("🚫 MySQL unavailable after all attempts. Falling back to SQLite.")
            self.sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trading_bot.db")
            
        self._init_db()

    def _detect_config(self):
        """Prioritize environment variables (Railway default vs Local custom)."""
        # Railway provides MYSQL* by default
        # Local setups often use DB_*
        self.host = (os.getenv('MYSQLHOST') or os.getenv('DB_HOST') or 'localhost').strip()
        
        port_str = (os.getenv('MYSQLPORT') or os.getenv('DB_PORT') or '3306').strip()
        try:
            self.port = int(port_str)
        except ValueError:
            self.port = 3306
            
        self.user = (os.getenv('MYSQLUSER') or os.getenv('DB_USER') or 'root').strip()
        self.password = (os.getenv('MYSQLPASSWORD') or os.getenv('DB_PASSWORD') or '').strip()
        self.database = (os.getenv('MYSQLDATABASE') or os.getenv('DB_NAME') or 'trading_bot').strip()
        
        logger.info(f"🔍 database_config: {self.host}:{self.port} user={self.user} db={self.database}")

    def _get_connection(self):
        """Create a new database connection (MySQL or SQLite)."""
        if self.use_sqlite:
            conn = sqlite3.connect(self.sqlite_path)
            conn.row_factory = sqlite3.Row
            return conn
            
        try:
            return mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password,
                database=self.database
            )
        except Error as e:
            # If database doesn't exist (1049), try creating it
            if e.errno == 1049:
                logger.info(f"🛠️ Database '{self.database}' not found. Attempting to create it...")
                try:
                    conn = mysql.connector.connect(
                        host=self.host,
                        port=self.port,
                        user=self.user,
                        password=self.password
                    )
                    cursor = conn.cursor()
                    cursor.execute(f"CREATE DATABASE IF NOT EXISTS {self.database}")
                    conn.commit()
                    cursor.close()
                    conn.close()
                    return self._get_connection()
                except Error as create_err:
                    logger.error(f"❌ Failed to create database: {create_err}")
                    raise create_err
            
            # For other errors, log specifically
            if e.errno == 1045:
                logger.error("❌ MySQL Authentication failed (1045). Check DB_USER/DB_PASSWORD.")
            elif e.errno == 2003:
                logger.error(f"❌ MySQL Server unreachable at {self.host}:{self.port} (2003).")
            
            raise e

    def _clean_sql(self, sql: str) -> str:
        """Adjust SQL syntax for SQLite if needed."""
        if not self.use_sqlite:
            return sql
            
        # Basic SQLite transformations
        sql = sql.replace('ENGINE=InnoDB', '')
        sql = sql.replace('INT AUTO_INCREMENT PRIMARY KEY', 'INTEGER PRIMARY KEY AUTOINCREMENT')
        sql = sql.replace('AUTO_INCREMENT', 'AUTOINCREMENT')
        sql = sql.replace('DOUBLE', 'REAL')
        sql = sql.replace('%s', '?')
        
        # Handle ON DUPLICATE KEY UPDATE (MySQL) -> ON CONFLICT (SQLite)
        # This is a very simplistic conversion for our specific needs
        if "ON DUPLICATE KEY UPDATE" in sql:
            # We assume the ID is the conflict target for bots table
            if "INSERT INTO bots" in sql:
                 parts = sql.split("ON DUPLICATE KEY UPDATE")
                 updates = parts[1].strip()
                 # SQLite syntax: ON CONFLICT(id) DO UPDATE SET ...
                 sql = f"{parts[0]} ON CONFLICT(id) DO UPDATE SET {updates}"
            # For system_state table
            elif "INSERT INTO system_state" in sql:
                 parts = sql.split("ON DUPLICATE KEY UPDATE")
                 updates = parts[1].strip()
                 sql = f"{parts[0]} ON CONFLICT(state_key) DO UPDATE SET {updates}"
            # For users table
            elif "INSERT INTO users" in sql:
                 parts = sql.split("ON DUPLICATE KEY UPDATE")
                 updates = parts[1].strip()
                 sql = f"{parts[0]} ON CONFLICT(id) DO UPDATE SET {updates}"
            # For v2_strategy_metrics table
            elif "INSERT INTO v2_strategy_metrics" in sql:
                 parts = sql.split("ON DUPLICATE KEY UPDATE")
                 updates = parts[1].strip()
                 sql = f"{parts[0]} ON CONFLICT(user_id, strategy) DO UPDATE SET {updates}"
            # For v2_strategy_profiles table
            elif "v2_strategy_profiles" in sql:
                 parts = sql.split("ON DUPLICATE KEY UPDATE")
                 updates = parts[1].strip()
                 sql = f"{parts[0]} ON CONFLICT(strategy) DO UPDATE SET {updates}"
        
        # Handle VALUES(...) in UPDATE part (MySQL-specific)
        if self.use_sqlite and "SET" in sql and "VALUES(" in sql:
             # Example: SET b = VALUES(b) -> SET b = excluded.b
             import re
             sql = re.sub(r'VALUES\((\w+)\)', r'excluded.\1', sql)

        # Handle INSERT IGNORE (MySQL) -> INSERT OR IGNORE (SQLite)
        if self.use_sqlite:
            sql = sql.replace('INSERT IGNORE', 'INSERT OR IGNORE')

        return sql

    def _execute(self, cursor, query: str, params: Any = None):
        """Execute a query with standard MySQL placeholders, handled for SQLite."""
        sql = self._clean_sql(query)
        if params:
            cursor.execute(sql, params)
        else:
            cursor.execute(sql)

    def _safe_close(self, conn, cursor=None):
        """Safely close connection and cursor for both MySQL and SQLite."""
        try:
            if cursor:
                cursor.close()
            if conn:
                if not self.use_sqlite:
                    if hasattr(conn, 'is_connected') and conn.is_connected():
                        conn.close()
                else:
                    conn.close()
        except Exception as e:
            logger.debug(f"Error during safe close: {e}")

    def _init_db(self):
        """Initialize database with required tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS users (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    username VARCHAR(255) UNIQUE,
                    password_hash TEXT,
                    mobile VARCHAR(20) UNIQUE,
                    otp VARCHAR(10),
                    is_verified TINYINT DEFAULT 0,
                    initial_capital DOUBLE DEFAULT 100000.0,
                    cash DOUBLE DEFAULT 100000.0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            # Bots table
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS bots (
                    id VARCHAR(255) PRIMARY KEY,
                    user_id INT,
                    symbol VARCHAR(50),
                    market VARCHAR(50),
                    strategy VARCHAR(100),
                    mode VARCHAR(20),
                    interval_str VARCHAR(10),
                    position_size DOUBLE,
                    stop_loss DOUBLE,
                    take_profit DOUBLE,
                    max_quantity DOUBLE,
                    auto_restart_enabled TINYINT DEFAULT 1,
                    status VARCHAR(20) DEFAULT 'stopped',
                    FOREIGN KEY (user_id) REFERENCES users (id)
                ) ENGINE=InnoDB
            ''')
            
            # Trades table
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS trades (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT,
                    trade_id VARCHAR(50) UNIQUE,
                    round_trip_id VARCHAR(50),
                    symbol VARCHAR(50),
                    side VARCHAR(20),
                    quantity DOUBLE,
                    price DOUBLE,
                    pnl DOUBLE,
                    strategy VARCHAR(100),
                    bot_id VARCHAR(255),
                    mode VARCHAR(20),
                    account_value DOUBLE,
                    trade_type VARCHAR(20),
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date DATE,
                    FOREIGN KEY (user_id) REFERENCES users (id)
                ) ENGINE=InnoDB
            ''')
            
            # Migration: Ensure 'side' column in 'trades' is long enough
            # SKIP for SQLite as it doesn't support ALTER TABLE ... MODIFY
            if not self.use_sqlite:
                try:
                    cursor.execute("ALTER TABLE trades MODIFY COLUMN side VARCHAR(20)")
                    conn.commit()
                except Exception:
                    pass

            # System Logs table
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS system_logs (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    level VARCHAR(20),
                    module VARCHAR(255),
                    message TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    bot_id VARCHAR(255),
                    user_id INT
                ) ENGINE=InnoDB
            ''')

            # News Sentiment table
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS news_sentiment (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    url VARCHAR(767) UNIQUE,
                    title TEXT,
                    source VARCHAR(255),
                    sentiment_score DOUBLE,
                    sentiment_label VARCHAR(20),
                    symbols TEXT,
                    timestamp TIMESTAMP NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')

            # System State table
            # SQLite doesn't support ON UPDATE CURRENT_TIMESTAMP
            state_sql = '''
                CREATE TABLE IF NOT EXISTS system_state (
                    state_key VARCHAR(100) PRIMARY KEY,
                    state_value TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            '''
            if not self.use_sqlite:
                state_sql = state_sql.replace('DEFAULT CURRENT_TIMESTAMP', 'DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP')
            
            self._execute(cursor, state_sql)

            # ── V2 Tables (isolated from V1) ──

            # V2 Trade Ledger — full execution audit trail
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_trades (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    trade_id VARCHAR(50),
                    symbol VARCHAR(50),
                    side VARCHAR(20),
                    position_side VARCHAR(20),
                    quantity DOUBLE,
                    fill_price DOUBLE,
                    market_price DOUBLE,
                    spread_pct DOUBLE,
                    slippage_pct DOUBLE,
                    commission DOUBLE,
                    volatility_input DOUBLE,
                    volume_input DOUBLE,
                    leverage DOUBLE DEFAULT 1,
                    margin_mode VARCHAR(20) DEFAULT 'isolated',
                    realized_pnl DOUBLE,
                    net_pnl DOUBLE,
                    entry_price DOUBLE,
                    strategy VARCHAR(100),
                    bot_id VARCHAR(255),
                    trade_type VARCHAR(20),
                    account_value DOUBLE,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    date DATE
                ) ENGINE=InnoDB
            ''')

            # V2 Strategy Metrics — aggregated per user per strategy
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_strategy_metrics (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    total_trades INT DEFAULT 0,
                    wins INT DEFAULT 0,
                    losses INT DEFAULT 0,
                    total_pnl DOUBLE DEFAULT 0,
                    avg_win DOUBLE DEFAULT 0,
                    avg_loss DOUBLE DEFAULT 0,
                    expectancy DOUBLE DEFAULT 0,
                    profit_factor DOUBLE DEFAULT 0,
                    sharpe_ratio DOUBLE DEFAULT 0,
                    sortino_ratio DOUBLE DEFAULT 0,
                    calmar_ratio DOUBLE DEFAULT 0,
                    recovery_factor DOUBLE DEFAULT 0,
                    max_drawdown_pct DOUBLE DEFAULT 0,
                    avg_r_multiple DOUBLE DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, strategy)
                ) ENGINE=InnoDB
            ''')

            # V2 Strategy Equity Curve
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_strategy_equity_curve (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    strategy VARCHAR(50) NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    equity_value DOUBLE NOT NULL
                ) ENGINE=InnoDB
            ''')

            # V2 Strategy Profiles — per-strategy risk parameters
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_strategy_profiles (
                    strategy VARCHAR(50) PRIMARY KEY,
                    default_risk_pct DOUBLE DEFAULT 2.0,
                    default_leverage DOUBLE DEFAULT 1.0,
                    max_drawdown_limit DOUBLE DEFAULT 10.0,
                    volatility_filter TINYINT DEFAULT 0
                ) ENGINE=InnoDB
            ''')

            # Seed default V2 strategy profiles
            for strat in ['ichimoku', 'bollinger', 'macd_rsi', 'ml_forecast', 'combined']:
                try:
                    self._execute(cursor,
                        "INSERT IGNORE INTO v2_strategy_profiles (strategy) VALUES (%s)",
                        (strat,)
                    )
                except Exception:
                    pass

            conn.commit()
            logger.info(f"{'SQLite' if self.use_sqlite else 'MySQL'} database initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")
        finally:
            self._safe_close(conn, cursor)

    # --- User Management ---
    
    def create_user(self, mobile: str) -> Optional[int]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "INSERT INTO users (mobile) VALUES (%s)", (mobile,))
            conn.commit()
            return cursor.lastrowid
        finally:
            self._safe_close(conn, cursor)

    def get_user_by_mobile(self, mobile: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT * FROM users WHERE mobile = %s", (mobile,))
            row = cursor.fetchone()
            if row and self.use_sqlite:
                return dict(row)
            return row
        finally:
            self._safe_close(conn, cursor)

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT * FROM users WHERE id = %s", (user_id,))
            row = cursor.fetchone()
            if row and self.use_sqlite:
                return dict(row)
            return row
        finally:
            self._safe_close(conn, cursor)

    def update_user_otp(self, user_id: int, otp: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "UPDATE users SET otp = %s WHERE id = %s", (otp, user_id))
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def verify_user(self, user_id: int, username: str, password_hash: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, 
                "UPDATE users SET is_verified = 1, username = %s, password_hash = %s WHERE id = %s",
                (username, password_hash, user_id)
            )
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT * FROM users WHERE username = %s", (username,))
            row = cursor.fetchone()
            if row and self.use_sqlite:
                return dict(row)
            return row
        finally:
            self._safe_close(conn, cursor)

    # --- Trade Management ---
    
    def add_trade(self, record: Dict):
        """Add a trade record to MySQL or SQLite."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            fields = list(record.keys())
            placeholders = ", ".join(["%s"] * len(fields))
            columns = ", ".join(fields)
            
            # Convert date str if needed
            data = list(record.values())
            
            self._execute(cursor, 
                f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                data
            )
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def save_trade(self, trade_data: Dict):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO trades (
                    user_id, trade_id, round_trip_id, symbol, side, 
                    quantity, price, pnl, strategy, bot_id, mode, 
                    account_value, trade_type, notes, date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                trade_data['user_id'], trade_data['trade_id'],
                trade_data.get('round_trip_id'), trade_data['symbol'],
                trade_data['side'], trade_data['quantity'],
                trade_data['price'], trade_data.get('pnl', 0),
                trade_data['strategy'], trade_data['bot_id'],
                trade_data['mode'], trade_data['account_value'],
                trade_data.get('trade_type', 'market'),
                trade_data.get('notes'), trade_data.get('date')
            )
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def get_user_trades(self, user_id: int, start_date: str = None, end_date: str = None, symbol: str = None, limit: int = 100) -> List[Dict]:
        """Get trades for a user with optional date and symbol filters."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            query = "SELECT * FROM trades WHERE user_id = %s"
            params = [user_id]
            
            if start_date:
                query += " AND date >= %s"
                params.append(start_date)
            if end_date:
                query += " AND date <= %s"
                params.append(end_date)
            if symbol:
                query += " AND symbol = %s"
                params.append(symbol.upper())
                
            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)
            
            self._execute(cursor, query, params)
            rows = cursor.fetchall()
            if self.use_sqlite:
                return [dict(row) for row in rows]
            return rows
        finally:
            self._safe_close(conn, cursor)

    # --- Bot Management ---
    
    def save_bot_config(self, config: Dict):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO bots (
                    id, user_id, symbol, market, strategy, mode, interval_str,
                    position_size, stop_loss, take_profit, max_quantity, auto_restart_enabled, status
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE 
                    symbol=%s, market=%s, strategy=%s, mode=%s, interval_str=%s,
                    position_size=%s, stop_loss=%s, take_profit=%s, max_quantity=%s, 
                    auto_restart_enabled=%s, status=%s
            """
            params = (
                config['id'], config['user_id'], config['symbol'],
                config['market'], config.get('strategy', 'hybrid'),
                config.get('mode', 'backtest'), config.get('interval', '1m'),
                config.get('position_size', 10.0), config.get('stop_loss', 2.0),
                config.get('take_profit', 4.0), config.get('max_quantity', 1.0),
                config.get('auto_restart_enabled', 1), config.get('status', 'running'),
                # For update
                config['symbol'], config['market'], config.get('strategy', 'hybrid'),
                config.get('mode', 'backtest'), config.get('interval', '1m'),
                config.get('position_size', 10.0), config.get('stop_loss', 2.0),
                config.get('take_profit', 4.0), config.get('max_quantity', 1.0),
                config.get('auto_restart_enabled', 1), config.get('status', 'running')
            )
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def get_all_bots(self, user_id: int = None) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            if user_id:
                self._execute(cursor, "SELECT * FROM bots WHERE user_id = %s", (user_id,))
            else:
                self._execute(cursor, "SELECT * FROM bots")
            
            rows = cursor.fetchall()
            bots = []
            for row in rows:
                bot = dict(row) if self.use_sqlite else row
                # Map interval_str back to interval for the app
                bot['interval'] = bot.pop('interval_str', '1m')
                bots.append(bot)
            return bots
        finally:
            self._safe_close(conn, cursor)

    # --- Logging ---
    
    def add_log(self, level: str, module: str, message: str, bot_id: Optional[str] = None, user_id: Optional[int] = None):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, 
                "INSERT INTO system_logs (level, module, message, bot_id, user_id) VALUES (%s, %s, %s, %s, %s)",
                (level, module, message, bot_id, user_id)
            )
            conn.commit()
        except Exception:
            # Don't use loguru here to avoid recursion
            pass
        finally:
            self._safe_close(conn, cursor)

    # --- News Sentiment ---
    
    def save_news(self, news_item: Dict):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            fields = ['url', 'title', 'source', 'sentiment_score', 'sentiment_label', 'symbols', 'timestamp']
            data = [
                news_item.get('url'),
                news_item.get('title'),
                news_item.get('source'),
                news_item.get('sentiment'),
                news_item.get('sentiment_label'),
                ",".join(news_item.get('symbols', [])),
                news_item.get('timestamp')
            ]
            
            columns = ", ".join(fields)
            placeholders = ", ".join(["%s"] * len(fields))
            
            self._execute(cursor, 
                f"INSERT IGNORE INTO news_sentiment ({columns}) VALUES ({placeholders})",
                data
            )
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    # --- System State ---
    
    def get_system_state_val(self, key: str) -> Optional[str]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "SELECT state_value FROM system_state WHERE state_key = %s", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            self._safe_close(conn, cursor)

    def set_system_state_val(self, key: str, value: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, 
                "INSERT INTO system_state (state_key, state_value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE state_value = VALUES(state_value)",
                (key, value)
            )
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    # ══════════════════════════════════════════════════════════════
    # V2 Methods — Fully isolated from V1
    # ══════════════════════════════════════════════════════════════

    def v2_save_trade(self, trade_data: Dict):
        """Save a V2 trade with full execution audit trail."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO v2_trades (
                    user_id, trade_id, symbol, side, position_side,
                    quantity, fill_price, market_price, spread_pct, slippage_pct,
                    commission, volatility_input, volume_input, leverage, margin_mode,
                    realized_pnl, net_pnl, entry_price, strategy, bot_id,
                    trade_type, account_value, notes, date
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                trade_data.get('user_id'), trade_data.get('trade_id'),
                trade_data.get('symbol'), trade_data.get('side'),
                trade_data.get('position_side'), trade_data.get('quantity'),
                trade_data.get('fill_price'), trade_data.get('market_price'),
                trade_data.get('spread_pct'), trade_data.get('slippage_pct'),
                trade_data.get('commission'), trade_data.get('volatility_input'),
                trade_data.get('volume_input'), trade_data.get('leverage', 1.0),
                trade_data.get('margin_mode', 'isolated'),
                trade_data.get('realized_pnl'), trade_data.get('net_pnl'),
                trade_data.get('entry_price'), trade_data.get('strategy'),
                trade_data.get('bot_id'), trade_data.get('trade_type'),
                trade_data.get('account_value'), trade_data.get('notes'),
                trade_data.get('date')
            )
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_user_trades(
        self, user_id: int, strategy: str = None,
        trade_type: str = None, limit: int = 500
    ) -> List[Dict]:
        """Get V2 trades with optional strategy and type filters."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            query = "SELECT * FROM v2_trades WHERE user_id = %s"
            params: list = [user_id]

            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            if trade_type:
                query += " AND trade_type = %s"
                params.append(trade_type)

            query += " ORDER BY timestamp DESC LIMIT %s"
            params.append(limit)

            self._execute(cursor, query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)

    def v2_upsert_strategy_metrics(self, user_id: int, strategy: str, metrics: Dict):
        """Insert or update V2 strategy metrics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO v2_strategy_metrics (
                    user_id, strategy, total_trades, wins, losses, total_pnl,
                    avg_win, avg_loss, expectancy, profit_factor, sharpe_ratio,
                    sortino_ratio, calmar_ratio, recovery_factor, max_drawdown_pct,
                    avg_r_multiple
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    total_trades=%s, wins=%s, losses=%s, total_pnl=%s,
                    avg_win=%s, avg_loss=%s, expectancy=%s, profit_factor=%s,
                    sharpe_ratio=%s, sortino_ratio=%s, calmar_ratio=%s,
                    recovery_factor=%s, max_drawdown_pct=%s, avg_r_multiple=%s
            """
            vals = (
                metrics.get('total_trades', 0), metrics.get('wins', 0),
                metrics.get('losses', 0), metrics.get('total_pnl', 0),
                metrics.get('avg_win', 0), metrics.get('avg_loss', 0),
                metrics.get('expectancy', 0), metrics.get('profit_factor', 0),
                metrics.get('sharpe_ratio', 0), metrics.get('sortino_ratio', 0),
                metrics.get('calmar_ratio', 0), metrics.get('recovery_factor', 0),
                metrics.get('max_drawdown_pct', 0), metrics.get('avg_r_multiple', 0),
            )
            params = (user_id, strategy) + vals + vals
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_append_equity_point(self, user_id: int, strategy: str, equity_value: float):
        """Append a point to the V2 strategy equity curve."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "INSERT INTO v2_strategy_equity_curve (user_id, strategy, equity_value) VALUES (%s, %s, %s)",
                (user_id, strategy, equity_value)
            )
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_strategy_metrics(self, user_id: int, strategy: str = None) -> List[Dict]:
        """Get V2 strategy metrics (all strategies if strategy=None)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            if strategy:
                self._execute(cursor,
                    "SELECT * FROM v2_strategy_metrics WHERE user_id = %s AND strategy = %s",
                    (user_id, strategy)
                )
            else:
                self._execute(cursor,
                    "SELECT * FROM v2_strategy_metrics WHERE user_id = %s",
                    (user_id,)
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)

    def v2_get_equity_curve(self, user_id: int, strategy: str, start_date: str = None) -> List[Dict]:
        """Get V2 equity curve points for a strategy."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            query = "SELECT * FROM v2_strategy_equity_curve WHERE user_id = %s AND strategy = %s"
            params: list = [user_id, strategy]

            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)

            query += " ORDER BY timestamp ASC"
            self._execute(cursor, query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)

    def v2_get_strategy_profile(self, strategy: str = None) -> List[Dict]:
        """Get V2 strategy profile(s)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            if strategy:
                self._execute(cursor, "SELECT * FROM v2_strategy_profiles WHERE strategy = %s", (strategy,))
            else:
                self._execute(cursor, "SELECT * FROM v2_strategy_profiles")
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)

    def v2_upsert_strategy_profile(self, strategy: str, profile: Dict):
        """Insert or update V2 strategy profile."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                INSERT INTO v2_strategy_profiles (strategy, default_risk_pct, default_leverage, max_drawdown_limit, volatility_filter)
                VALUES (%s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    default_risk_pct=%s, default_leverage=%s, max_drawdown_limit=%s, volatility_filter=%s
            """
            vals = (
                profile.get('default_risk_pct', 2.0),
                profile.get('default_leverage', 1.0),
                profile.get('max_drawdown_limit', 10.0),
                1 if profile.get('volatility_filter') else 0,
            )
            params = (strategy,) + vals + vals
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_sync_strategy_profiles(self, strategies: List[Dict]):
        """Ensure all strategies from registry have a profile in the database."""
        for strategy in strategies:
            s_id = strategy['id']
            conn = self._get_connection()
            try:
                cursor = conn.cursor()
                self._execute(cursor, "SELECT strategy FROM v2_strategy_profiles WHERE strategy = %s", (s_id,))
                if not cursor.fetchone():
                    logger.info(f"✨ Seeding default profile for new strategy: {s_id}")
                    self._execute(cursor, """
                        INSERT INTO v2_strategy_profiles (strategy, default_risk_pct, default_leverage, max_drawdown_limit, volatility_filter)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (s_id, 2.0, 1.0, 10.0, 0))
                    conn.commit()
            except Exception as e:
                logger.error(f"❌ Error syncing strategy profile for {s_id}: {e}")
            finally:
                self._safe_close(conn, cursor)

    # ── Trade Clearing (Reset) ────────────────────────────────

    def clear_user_trades(self, user_id: int):
        """Delete all V1 trades for a user (used by paper trading reset)."""
        conn, cursor = self._connect()
        try:
            self._execute(cursor, "DELETE FROM trades WHERE user_id = %s", (user_id,))
            conn.commit()
            logger.info(f"🗑️ Cleared V1 trades for user {user_id}")
        finally:
            self._safe_close(conn, cursor)

    def v2_clear_user_trades(self, user_id: int):
        """Delete all V2 trades for a user (used by V2 paper trading reset)."""
        conn, cursor = self._connect()
        try:
            self._execute(cursor, "DELETE FROM v2_trades WHERE user_id = %s", (user_id,))
            # Also clear V2 strategy metrics
            try:
                self._execute(cursor, "DELETE FROM v2_strategy_metrics WHERE user_id = %s", (user_id,))
            except Exception:
                pass  # Table may not exist yet
            conn.commit()
            logger.info(f"🗑️ Cleared V2 trades + metrics for user {user_id}")
        finally:
            self._safe_close(conn, cursor)


# Singleton instance
db_manager = DatabaseManager()

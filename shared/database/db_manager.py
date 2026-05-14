import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime, timezone
from loguru import logger
from typing import Dict, List, Optional, Any
import sqlite3
import time
import random
import string
import requests


class D1ProxyConnection:
    def __init__(self, url: str, secret: str):
        self.url = url
        self.secret = secret
        
    def cursor(self, dictionary=False):
        return D1ProxyCursor(self.url, self.secret)
        
    def commit(self):
        pass # D1 commits automatically
        
    def close(self):
        pass


class D1ProxyCursor:
    def __init__(self, url: str, secret: str):
        self.url = url
        self.secret = secret
        self.results = []
        self.current_idx = 0
        self.lastrowid = None
        
    def execute(self, query: str, params: Any = None):
        headers = {
            "Authorization": f"Bearer {self.secret}",
            "Content-Type": "application/json"
        }
        
        # Determine method based on query type
        method = "all"
        if query.strip().upper().startswith(("INSERT", "UPDATE", "DELETE", "CREATE", "DROP")):
            method = "run"
            
        payload = {
            "query": query,
            "params": params or [],
            "method": method
        }
        
        try:
            response = requests.post(self.url, headers=headers, json=payload, timeout=10)
            if response.status_code != 200:
                logger.error(f"❌ D1 Proxy Error ({response.status_code}): {response.text}")
                raise Exception(f"D1 Proxy Error: {response.text}")
                
            data = response.json()
            
            if method == "run":
                self.results = []
                meta = data.get("meta", {})
                self.lastrowid = meta.get("last_row_id")
            else:
                self.results = data.get("results", [])
                meta = data.get("meta", {})
                self.lastrowid = meta.get("last_row_id")
                
            self.current_idx = 0
            
        except Exception as e:
            logger.error(f"❌ Failed to execute query via D1 Proxy: {e}")
            raise e
        
    def fetchone(self):
        if self.current_idx < len(self.results):
            row = self.results[self.current_idx]
            self.current_idx += 1
            return row
        return None
        
    def fetchall(self):
        return self.results
        
    def close(self):
        pass


class DatabaseManager:
    def __init__(self):
        self._detect_config()
        
        # Check if D1 Proxy is enabled
        if self.use_d1_proxy:
            logger.info("🌐 database_manager: Using Cloudflare D1 Proxy.")
            self.use_sqlite = True # D1 uses SQLite syntax
            self._init_db()
            return
            
        # Retry logic: Attempt to connect to MySQL 3 times before failing back to SQLite
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
        self.use_d1_proxy = False
        self.proxy_url = os.getenv('DB_PROXY_URL')
        self.proxy_secret = os.getenv('DB_PROXY_SECRET')
        
        if self.proxy_url:
            self.use_d1_proxy = True
            logger.info(f"🔍 database_config: D1 Proxy enabled at {self.proxy_url}")
            return
            
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
        """Create a new database connection (MySQL, SQLite, or D1 Proxy)."""
        if self.use_d1_proxy:
            return D1ProxyConnection(self.proxy_url, self.proxy_secret)
            
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

            # ── V2 Sessions Table (Lifecycle Tracking) ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_sessions (
                    session_id VARCHAR(50) PRIMARY KEY,
                    start_time DATETIME,
                    end_time DATETIME,
                    total_trades INT DEFAULT 0,
                    total_pnl DECIMAL(18,8) DEFAULT 0,
                    status VARCHAR(20) -- RUNNING, STOPPED, ERROR
                ) ENGINE=InnoDB
            ''')

            # ── V2 Trade Ledger (Single Source of Truth) ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_trade_ledger (
                    trade_id VARCHAR(80) PRIMARY KEY,
                    session_id VARCHAR(50),
                    user_id INT,
                    symbol VARCHAR(20),
                    side VARCHAR(10),     -- BUY / SELL
                    action VARCHAR(20),   -- OPEN / CLOSE / REVERSAL / STOP_LOSS / TAKE_PROFIT / PARTIAL_CLOSE
                    quantity DECIMAL(18,8),
                    price DECIMAL(18,8),
                    pnl DECIMAL(18,8),
                    commission DECIMAL(18,8),
                    strategy VARCHAR(50),
                    timestamp DATETIME
                ) ENGINE=InnoDB
            ''')

            # V2 Trade Ledger — full execution audit trail
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_trades (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    trade_id VARCHAR(100) UNIQUE,
                    symbol VARCHAR(50),
                    side VARCHAR(20),
                    position_side VARCHAR(20),
                    quantity DOUBLE,
                    fill_price DECIMAL(18,8),
                    market_price DOUBLE,
                    spread_pct DOUBLE,
                    slippage_pct DOUBLE,
                    commission DECIMAL(18,8),
                    volatility_input DOUBLE,
                    volume_input DOUBLE,
                    leverage DOUBLE DEFAULT 1,
                    margin_mode VARCHAR(20) DEFAULT 'isolated',
                    realized_pnl DECIMAL(18,8),
                    net_pnl DECIMAL(18,8),
                    entry_price DECIMAL(18,8),
                    strategy VARCHAR(100),
                    bot_id VARCHAR(255),
                    session_id VARCHAR(32),
                    trade_type VARCHAR(20),
                    direction VARCHAR(20),
                    duration_seconds DOUBLE,
                    account_value DOUBLE,
                    notes TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    trade_time TIMESTAMP,
                    date DATE,
                    CONSTRAINT fk_v2_session
                        FOREIGN KEY (session_id)
                        REFERENCES v2_sessions(session_id)
                        ON DELETE CASCADE
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

            # ── V2 Active Positions (Exposure Tracking) ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_positions (
                    user_id INT,
                    symbol VARCHAR(20),
                    side VARCHAR(10),  -- LONG / SHORT
                    quantity DECIMAL(18,8),
                    entry_price DECIMAL(18,8),
                    stop_loss DECIMAL(18,8),
                    take_profit DECIMAL(18,8),
                    strategy VARCHAR(50),
                    open_time DATETIME,
                    last_update DATETIME,
                    PRIMARY KEY (user_id, symbol)
                ) ENGINE=InnoDB
            ''')

            # ── V2 Bot State (Runtime State Persistence) ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS v2_bot_state (
                    bot_id VARCHAR(255) PRIMARY KEY,
                    user_id INT,
                    status VARCHAR(20), -- running, stopped, error
                    last_updated DATETIME
                ) ENGINE=InnoDB
            ''')

            # Migration for v2_trades: Add session_id and trade_time
            if not self.use_sqlite:
                # MySQL Migrations
                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN session_id VARCHAR(32) AFTER bot_id")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN trade_type VARCHAR(20) AFTER session_id")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN direction VARCHAR(20) AFTER trade_type")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN duration_seconds DOUBLE AFTER direction")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN account_value DOUBLE AFTER duration_seconds")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN trade_time TIMESTAMP AFTER timestamp")
                    conn.commit()
                except Exception: pass
                
                # Performance Indices
                try:
                    cursor.execute("CREATE INDEX idx_trade_time ON v2_trades(trade_time)")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("CREATE INDEX idx_session_id ON v2_trades(session_id)")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("CREATE INDEX idx_session_time ON v2_trades(session_id, trade_time)")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("CREATE INDEX idx_symbol_time ON v2_trades(symbol, trade_time)")
                    conn.commit()
                except Exception: pass

                # v2_sessions migrations
                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN status VARCHAR(16) DEFAULT 'ACTIVE' AFTER engine_version")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN total_trades INT DEFAULT 0 AFTER status")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN total_pnl DECIMAL(18,8) DEFAULT 0 AFTER total_trades")
                    conn.commit()
                except Exception: pass

                # Financial Precision Conversions (MySQL)
                try:
                    cursor.execute("ALTER TABLE v2_sessions MODIFY total_pnl DECIMAL(18,8) DEFAULT 0")
                    cursor.execute("ALTER TABLE v2_trades MODIFY fill_price DECIMAL(18,8)")
                    cursor.execute("ALTER TABLE v2_trades MODIFY commission DECIMAL(18,8)")
                    cursor.execute("ALTER TABLE v2_trades MODIFY realized_pnl DECIMAL(18,8)")
                    cursor.execute("ALTER TABLE v2_trades MODIFY net_pnl DECIMAL(18,8)")
                    cursor.execute("ALTER TABLE v2_trades MODIFY entry_price DECIMAL(18,8)")
                    conn.commit()
                except Exception: pass

                # Additional Indices
                try:
                    cursor.execute("CREATE INDEX idx_strategy_time ON v2_trades(strategy, trade_time)")
                    conn.commit()
                except Exception: pass

                # Add FK Constraint to v2_trades (MySQL)
                try:
                    cursor.execute("""
                        ALTER TABLE v2_trades 
                        ADD CONSTRAINT fk_v2_session 
                        FOREIGN KEY (session_id) 
                        REFERENCES v2_sessions(session_id) 
                        ON DELETE CASCADE
                    """)
                    conn.commit()
                except Exception: pass
            else:
                # SQLite Migrations
                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN session_id TEXT")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("ALTER TABLE v2_trades ADD COLUMN trade_time TEXT")
                    conn.commit()
                except Exception: pass
                
                # Indices for SQLite
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_trade_time ON v2_trades(trade_time)")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_id ON v2_trades(session_id)")
                    conn.commit()
                except Exception: pass
                
                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_session_time ON v2_trades(session_id, trade_time)")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_symbol_time ON v2_trades(symbol, trade_time)")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("CREATE INDEX IF NOT EXISTS idx_strategy_time ON v2_trades(strategy, trade_time)")
                    conn.commit()
                except Exception: pass

                # v2_sessions migrations for SQLite
                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN status TEXT DEFAULT 'ACTIVE'")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN start_time TEXT")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN end_time TEXT")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN total_trades INTEGER DEFAULT 0")
                    conn.commit()
                except Exception: pass

                try:
                    cursor.execute("ALTER TABLE v2_sessions ADD COLUMN total_pnl REAL DEFAULT 0")
                    conn.commit()
                except Exception: pass


                # v2_positions migrations for SQLite
                for col in [
                    ("entry_price", "REAL DEFAULT 0"),
                    ("stop_loss", "REAL DEFAULT 0"),
                    ("take_profit", "REAL DEFAULT 0"),
                    ("open_time", "TEXT"),
                    ("last_update", "TEXT")
                ]:
                    try:
                        cursor.execute(f"ALTER TABLE v2_positions ADD COLUMN {col[0]} {col[1]}")
                        conn.commit()
                    except Exception: pass

            # ── Schema Versioning ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS db_schema_version (
                    version INT PRIMARY KEY,
                    applied_at DATETIME
                ) ENGINE=InnoDB
            ''')

            # ── User Watchlist (Favorites) ──
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS user_watchlist (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    user_id INT NOT NULL,
                    symbol VARCHAR(50) NOT NULL,
                    market VARCHAR(50) NOT NULL,
                    name VARCHAR(255),
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_id, symbol),
                    FOREIGN KEY (user_id) REFERENCES users(id)
                ) ENGINE=InnoDB
            ''')

            # Run Migrations
            if not self.use_sqlite:
                self._migrate_v2_positions(cursor)
                self._migrate_v2_sessions(cursor)
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

    # --- Watchlist / Favorites Management ---

    def add_to_watchlist(self, user_id: int, symbol: str, market: str, name: str = None) -> bool:
        """Add a symbol to the user's watchlist. Returns True if added, False if already exists."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "INSERT IGNORE INTO user_watchlist (user_id, symbol, market, name) VALUES (%s, %s, %s, %s)",
                (user_id, symbol.upper(), market.lower(), name)
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to add to watchlist: {e}")
            return False
        finally:
            self._safe_close(conn, cursor)

    def remove_from_watchlist(self, user_id: int, symbol: str) -> bool:
        """Remove a symbol from the user's watchlist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "DELETE FROM user_watchlist WHERE user_id = %s AND symbol = %s",
                (user_id, symbol.upper())
            )
            conn.commit()
            return cursor.rowcount > 0
        except Exception as e:
            logger.error(f"Failed to remove from watchlist: {e}")
            return False
        finally:
            self._safe_close(conn, cursor)

    def get_user_watchlist(self, user_id: int) -> List[Dict]:
        """Get all watchlist items for a user, ordered by most recently added."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor,
                "SELECT symbol, market, name, added_at FROM user_watchlist WHERE user_id = %s ORDER BY added_at DESC",
                (user_id,)
            )
            rows = cursor.fetchall()
            if self.use_sqlite:
                return [dict(row) for row in rows]
            return rows
        except Exception as e:
            logger.error(f"Failed to get watchlist: {e}")
            return []
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

    def v2_create_session(self, session_id: str, engine_version: str = "v2"):
        """Create a new V2 institutional session record."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc)
            self._execute(cursor, 
                """INSERT INTO v2_sessions (session_id, start_time, total_trades, total_pnl, status, engine_version) 
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON DUPLICATE KEY UPDATE status='RUNNING'""",
                (session_id, now, 0, 0.0, 'RUNNING', engine_version)
            )
            conn.commit()
            logger.info(f"🆕 [V2-SESSION] Started: {session_id} (Engine={engine_version})")
        finally:
            self._safe_close(conn, cursor)

    def v2_update_session_status(self, session_id: str, status: str):
        """Update the status of a trading session."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, 
                "UPDATE v2_sessions SET status = %s WHERE session_id = %s",
                (status, session_id)
            )
            conn.commit()
            logger.debug(f"🔄 [V2-SESSION] Status updated: {session_id} -> {status}")
        finally:
            self._safe_close(conn, cursor)

    def _migrate_v2_positions(self, cursor):
        """Idempotently add institutional columns to v2_positions."""
        cols_to_add = [
            ("entry_price", "DECIMAL(18,8)"),
            ("stop_loss", "DECIMAL(18,8)"),
            ("take_profit", "DECIMAL(18,8)"),
            ("open_time", "DATETIME"),
            ("last_update", "DATETIME")
        ]
        for col, data_type in cols_to_add:
            try:
                cursor.execute(f"SHOW COLUMNS FROM v2_positions LIKE '{col}'")
                if not cursor.fetchone():
                    logger.info(f"🛠️ Migrating v2_positions: Adding {col}")
                    cursor.execute(f"ALTER TABLE v2_positions ADD COLUMN {col} {data_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to add {col} to v2_positions: {e}")

        # Data Sync for legacy positions
        try:
            # Check if legacy columns exist before syncing
            cursor.execute("SHOW COLUMNS FROM v2_positions LIKE 'avg_price'")
            if cursor.fetchone():
                cursor.execute("UPDATE v2_positions SET entry_price = avg_price WHERE entry_price IS NULL AND avg_price IS NOT NULL")
            
            cursor.execute("SHOW COLUMNS FROM v2_positions LIKE 'opened_at'")
            if cursor.fetchone():
                cursor.execute("UPDATE v2_positions SET open_time = opened_at WHERE open_time IS NULL AND opened_at IS NOT NULL")
            
            cursor.execute("SHOW COLUMNS FROM v2_positions LIKE 'updated_at'")
            if cursor.fetchone():
                cursor.execute("UPDATE v2_positions SET last_update = updated_at WHERE last_update IS NULL AND updated_at IS NOT NULL")
        except Exception as e:
            logger.warning(f"⚠️ Failed to sync v2_positions data: {e}")

    def _migrate_v2_sessions(self, cursor):
        """Idempotently add institutional columns to v2_sessions."""
        cols_to_add = [
            ("engine_version", "VARCHAR(20)"),
            ("end_time", "DATETIME")
        ]
        for col, data_type in cols_to_add:
            try:
                cursor.execute(f"SHOW COLUMNS FROM v2_sessions LIKE '{col}'")
                if not cursor.fetchone():
                    logger.info(f"🛠️ Migrating v2_sessions: Adding {col}")
                    cursor.execute(f"ALTER TABLE v2_sessions ADD COLUMN {col} {data_type}")
            except Exception as e:
                logger.warning(f"⚠️ Failed to add {col} to v2_sessions: {e}")

    def v2_stop_session(self, session_id: str):
        """Stop a V2 session and finalize totals from the ledger."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Recalculate totals from ledger to ensure absolute consistency
            self._execute(cursor, "SELECT COUNT(*), SUM(pnl) FROM v2_trade_ledger WHERE session_id = %s", (session_id,))
            totals = cursor.fetchone()
            count = totals[0] if totals else 0
            pnl = float(totals[1]) if totals and totals[1] is not None else 0.0
            
            logger.debug(f"🔍 [V2-SESSION] Recalculated for {session_id}: trades={count}, pnl={pnl}")

            if count == 0:
                self._execute(cursor, "DELETE FROM v2_sessions WHERE session_id = %s", (session_id,))
                conn.commit()
                logger.info(f"🗑️ [V2-SESSION] Deleted empty session (zero trades): {session_id}")
            else:
                self._execute(cursor, """
                    UPDATE v2_sessions 
                    SET status = 'STOPPED', 
                        end_time = %s,
                        total_trades = %s,
                        total_pnl = %s
                    WHERE session_id = %s
                """, (datetime.now(timezone.utc), count, pnl, session_id))
                conn.commit()
                logger.info(f"🛑 [V2-SESSION] Stopped: {session_id} (PnL: {pnl:.2f}, Trades: {count})")
        finally:
            self._safe_close(conn, cursor)

    def v2_mark_crashed_sessions(self):
        """Mark any ACTIVE sessions as CRASHED (run on engine startup)."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "UPDATE v2_sessions SET status = 'CRASHED' WHERE status = 'ACTIVE'")
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_last_trade_time(self, user_id: int) -> Optional[str]:
        """Get the timestamp of the last executed V2 trade."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "SELECT MAX(trade_time) FROM v2_trades WHERE user_id = %s", (user_id,))
            result = cursor.fetchone()
            if result and result[0]:
                if hasattr(result[0], 'isoformat'):
                    return result[0].isoformat()
                return str(result[0])
            return None
        finally:
            self._safe_close(conn, cursor)

    def check_db_health(self) -> float:
        """Measure database response latency (ms)."""
        import time
        start = time.time()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "SELECT 1")
            cursor.fetchone()
            return (time.time() - start) * 1000
        finally:
            self._safe_close(conn, cursor)

    def v2_update_session_counters(self, session_id: str, profit_delta: float):
        """Atomically increment session trade count and PnL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                UPDATE v2_sessions 
                SET total_trades = total_trades + 1, 
                    total_pnl = total_pnl + %s 
                WHERE session_id = %s
            """
            self._execute(cursor, query, (profit_delta, session_id))
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_daily_pnl(self, user_id: int, date: Any) -> float:
        """Sum net PnL from ledger for a specific user and date."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT SUM(pnl) FROM v2_trade_ledger WHERE user_id = %s AND DATE(timestamp) = %s"
            self._execute(cursor, query, (user_id, date))
            row = cursor.fetchone()
            return float(row[0]) if row and row[0] is not None else 0.0
        finally:
            self._safe_close(conn, cursor)

    def v2_update_session_counters(self, session_id: str, pnl_delta: float):
        """Atomically update session trade count and total PnL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = """
                UPDATE v2_sessions 
                SET total_trades = total_trades + 1,
                    total_pnl = total_pnl + %s
                WHERE session_id = %s
            """
            self._execute(cursor, query, (pnl_delta, session_id))
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_sessions(self) -> List[Dict]:
        """Get all V2 sessions with metadata."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            # Filter out "dirty" test/verification sessions and unmeaningful empty sessions
            query = """
                SELECT * FROM v2_sessions 
                WHERE (total_trades > 0 OR status = 'ACTIVE')
                AND session_id NOT LIKE 'test_%' 
                AND session_id NOT LIKE 'VERIFY_%'
                ORDER BY start_time DESC
                LIMIT 20
            """
            self._execute(cursor, query)
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)

    def v2_get_active_session_id(self) -> Optional[str]:
        """Get the most recent ACTIVE session ID."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT session_id FROM v2_sessions WHERE status = 'ACTIVE' ORDER BY start_time DESC LIMIT 1")
            row = cursor.fetchone()
            if row:
                return row['session_id'] if not self.use_sqlite else row[0]
            return None
        finally:
            self._safe_close(conn, cursor)

    def v2_get_total_trade_count(self, user_id: int, session_id: Optional[str] = None, 
                                 strategy: Optional[str] = None, 
                                 start_date: Optional[str] = None, 
                                 end_date: Optional[str] = None) -> int:
        """Get total count of trades matching filters for pagination."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            query = "SELECT COUNT(*) FROM v2_trades WHERE user_id = %s"
            params = [user_id]
            
            if session_id:
                query += " AND session_id = %s"
                params.append(session_id)
            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            if start_date:
                query += " AND trade_time >= %s"
                params.append(start_date)
            if end_date:
                query += " AND trade_time <= %s"
                params.append(end_date)
            
            self._execute(cursor, query, tuple(params))
            row = cursor.fetchone()
            return row[0] if row else 0
        finally:
            self._safe_close(conn, cursor)

    def v2_save_trade(self, trade_data: Dict):
        """
        Save a V2 trade to the institutional ledger. 
        Ensures trade_id uniqueness and generates one if missing.
        """
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Generate trade_id if not provided
            if not trade_data.get('trade_id'):
                import uuid
                timestamp_str = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                short_uuid = str(uuid.uuid4())[:8]
                trade_id = f"{trade_data.get('symbol', 'UNKNOWN')}_{timestamp_str}_{trade_data.get('action', 'TRADE')}_{short_uuid}"
                trade_data['trade_id'] = trade_id

            query = """
                INSERT IGNORE INTO v2_trade_ledger (
                    trade_id, session_id, user_id, symbol, side, action, 
                    quantity, price, pnl, commission, strategy, timestamp
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            params = (
                trade_data.get('trade_id'),
                trade_data.get('session_id'),
                trade_data.get('user_id'),
                trade_data.get('symbol'),
                trade_data.get('side'),
                trade_data.get('action'),
                trade_data.get('quantity'),
                trade_data.get('price'),
                trade_data.get('pnl', 0.0),
                trade_data.get('commission', 0.0),
                trade_data.get('strategy'),
                trade_data.get('timestamp') or datetime.now(timezone.utc)
            )
            self._execute(cursor, query, params)
            conn.commit()
            logger.debug(f"📜 [V2-LEDGER] Trade saved: {trade_data['trade_id']}")
        finally:
            self._safe_close(conn, cursor)

    def v2_get_user_trades(
        self, user_id: int, strategy: str = None,
        trade_type: str = None, session_id: str = None,
        start_date: str = None, end_date: str = None,
        limit: int = 500, offset: int = 0
    ) -> List[Dict]:
        """Get V2 trades from the institutional ledger with optional filtering."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            query = "SELECT * FROM v2_trade_ledger WHERE user_id = %s"
            params: list = [user_id]

            if strategy:
                query += " AND strategy = %s"
                params.append(strategy)
            
            # Institutional: Map 'CLOSE' trade_type to all closing actions
            if trade_type == 'CLOSE':
                query += " AND action IN ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')"
            elif trade_type:
                query += " AND action = %s"
                params.append(trade_type)
                
            if session_id:
                query += " AND session_id = %s"
                params.append(session_id)
            if start_date:
                query += " AND timestamp >= %s"
                params.append(start_date)
            if end_date:
                query += " AND timestamp <= %s"
                params.append(end_date)

            query += " ORDER BY timestamp DESC, trade_id DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])

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
        """Delete all V2 trades, metrics, and positions for a user (used by V2 paper trading reset)."""
        conn, cursor = self._connect()
        try:
            self._execute(cursor, "DELETE FROM v2_trades WHERE user_id = %s", (user_id,))
            # Also clear V2 strategy metrics
            try:
                self._execute(cursor, "DELETE FROM v2_strategy_metrics WHERE user_id = %s", (user_id,))
            except Exception:
                pass  # Table may not exist yet
            # Clear V2 positions
            try:
                self._execute(cursor, "DELETE FROM v2_positions WHERE user_id = %s", (user_id,))
            except Exception:
                pass
            conn.commit()
            logger.info(f"🗑️ Cleared V2 trades + metrics + positions for user {user_id}")
        finally:
            self._safe_close(conn, cursor)

    # ── V2 Position Persistence (Refined) ──

    def v2_save_position(self, user_id: int, data: Dict):
        """Upsert active position state with high-precision decimals."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if self.use_sqlite:
                query = """
                    INSERT OR REPLACE INTO v2_positions 
                    (user_id, symbol, side, quantity, entry_price, stop_loss, take_profit, strategy, open_time, last_update)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """
            else:
                query = """
                    INSERT INTO v2_positions 
                    (user_id, symbol, side, quantity, entry_price, stop_loss, take_profit, strategy, open_time, last_update)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                    ON DUPLICATE KEY UPDATE 
                        side = VALUES(side),
                        quantity = VALUES(quantity),
                        entry_price = VALUES(entry_price),
                        stop_loss = VALUES(stop_loss),
                        take_profit = VALUES(take_profit),
                        strategy = VALUES(strategy),
                        last_update = VALUES(last_update)
                """
            params = (
                user_id, data['symbol'], data['side'], data['quantity'], data['entry_price'],
                data.get('stop_loss'), data.get('take_profit'),
                data.get('strategy'), data.get('open_time'), data.get('last_update')
            )
            self._execute(cursor, query, params)
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_position(self, user_id: int, symbol: str) -> Optional[Dict]:
        """Fetch active position for a symbol."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT * FROM v2_positions WHERE user_id = %s AND symbol = %s", (user_id, symbol))
            row = cursor.fetchone()
            return dict(row) if row and self.use_sqlite else row
        finally:
            self._safe_close(conn, cursor)

    def v2_delete_position(self, user_id: int, symbol: str):
        """Remove position from active tracking."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "DELETE FROM v2_positions WHERE user_id = %s AND symbol = %s", (user_id, symbol))
            conn.commit()
        finally:
            self._safe_close(conn, cursor)

    def v2_get_positions(self, user_id: int) -> List[Dict]:
        """Fetch all active positions for user."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor() if self.use_sqlite else conn.cursor(dictionary=True)
            self._execute(cursor, "SELECT * FROM v2_positions WHERE user_id = %s", (user_id,))
            rows = cursor.fetchall()
            return [dict(row) for row in rows] if self.use_sqlite else rows
        finally:
            self._safe_close(conn, cursor)


# Singleton instance
db_manager = DatabaseManager()

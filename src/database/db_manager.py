import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional, Any
import sqlite3

class DatabaseManager:
    def __init__(self):
        # Support both custom DB_* and Railway default MYSQL* environment variables
        # Priority: MYSQL* variables (Railway default) -> DB_* variables -> Defaults
        # We check MYSQL* first because if DB_* is set to "" (empty string) it might override valid MYSQL* values
        
        self.host = (os.getenv('MYSQLHOST') or os.getenv('DB_HOST') or 'localhost').strip()
        
        # Handle port parsing safely
        port_str = (os.getenv('MYSQLPORT') or os.getenv('DB_PORT') or '3306').strip()
        try:
            self.port = int(port_str)
        except ValueError:
            self.port = 3306
                
        self.user = (os.getenv('MYSQLUSER') or os.getenv('DB_USER') or 'root').strip()
        self.password = (os.getenv('MYSQLPASSWORD') or os.getenv('DB_PASSWORD') or '').strip()
        self.database = (os.getenv('MYSQLDATABASE') or os.getenv('DB_NAME') or 'trading_bot').strip()
        
        logger.info(f"🔌 database_manager: Connecting to {self.host}:{self.port} as {self.user} (DB: {self.database})")
        
        self.use_sqlite = False
        try:
            # Check if we can connect to MySQL
            conn = mysql.connector.connect(
                host=self.host,
                port=self.port,
                user=self.user,
                password=self.password
            )
            conn.close()
            logger.info("✅ MySQL connection verified.")
        except Exception as e:
            logger.warning(f"⚠️ MySQL connection failed: {e}. Falling back to SQLite.")
            self.use_sqlite = True
            self.sqlite_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "trading_bot.db")
            
        self._init_db()

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
            logger.error(f"Error connecting to MySQL: {e}")
            # If database doesn't exist, try connecting without it to create it
            if e.errno == 1049: # Unknown database
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

# Singleton instance
db_manager = DatabaseManager()

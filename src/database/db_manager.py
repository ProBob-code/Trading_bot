import mysql.connector
from mysql.connector import Error
import os
from datetime import datetime
from loguru import logger
from typing import Dict, List, Optional, Any

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
        
        self._init_db()

    def _get_connection(self):
        """Create a new MySQL connection."""
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

    def _init_db(self):
        """Initialize database with required tables."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Users table
            cursor.execute('''
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
            cursor.execute('''
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
            cursor.execute('''
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
            try:
                cursor.execute("ALTER TABLE trades MODIFY COLUMN side VARCHAR(20)")
                conn.commit()
            except Exception:
                pass

            # System Logs table
            cursor.execute('''
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
            cursor.execute('''
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
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS system_state (
                    state_key VARCHAR(100) PRIMARY KEY,
                    state_value TEXT,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            conn.commit()
            logger.info("MySQL database initialized successfully")
        except Error as e:
            logger.error(f"Failed to initialize database: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # --- User Management ---
    
    def create_user(self, mobile: str) -> bool:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (mobile) VALUES (%s)", (mobile,))
            conn.commit()
            return True
        except Error:
            return False
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_user_by_mobile(self, mobile: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE mobile = %s", (mobile,))
            return cursor.fetchone()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_user_by_id(self, user_id: int) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE id = %s", (user_id,))
            return cursor.fetchone()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def update_user_otp(self, user_id: int, otp: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("UPDATE users SET otp = %s WHERE id = %s", (otp, user_id))
            conn.commit()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def verify_user(self, user_id: int, username: str, password_hash: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "UPDATE users SET is_verified = 1, username = %s, password_hash = %s WHERE id = %s",
                (username, password_hash, user_id)
            )
            conn.commit()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_user_by_username(self, username: str) -> Optional[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
            return cursor.fetchone()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # --- Trade Management ---
    
    def add_trade(self, record: Dict):
        """Add a trade record to MySQL."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            fields = list(record.keys())
            placeholders = ", ".join(["%s"] * len(fields))
            columns = ", ".join(fields)
            
            # Convert date str if needed
            data = list(record.values())
            
            cursor.execute(
                f"INSERT INTO trades ({columns}) VALUES ({placeholders})",
                data
            )
            conn.commit()
        except Error as e:
            logger.error(f"Error adding trade to MySQL: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_user_trades(self, user_id: int, limit: int = 100) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute(
                "SELECT * FROM trades WHERE user_id = %s ORDER BY timestamp DESC LIMIT %s", 
                (user_id, limit)
            )
            return cursor.fetchall()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # --- Bot Management ---
    
    def save_bot_config(self, config: Dict):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            fields = list(config.keys())
            # Rename interval to interval_str for MySQL keyword avoidance if needed, 
            # but I already used interval_str in table definition.
            # Map 'interval' to 'interval_str' in the input dict if necessary.
            if 'interval' in config:
                config['interval_str'] = config.pop('interval')
                fields = list(config.keys())

            placeholders = ", ".join(["%s"] * len(fields))
            columns = ", ".join(fields)
            updates = ", ".join([f"{f} = VALUES({f})" for f in fields])
            
            data = list(config.values())
            
            cursor.execute(
                f"INSERT INTO bots ({columns}) VALUES ({placeholders}) "
                f"ON DUPLICATE KEY UPDATE {updates}",
                data
            )
            conn.commit()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def get_all_bots(self, user_id: Optional[int] = None) -> List[Dict]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            if user_id:
                cursor.execute("SELECT * FROM bots WHERE user_id = %s", (user_id,))
            else:
                cursor.execute("SELECT * FROM bots")
            rows = cursor.fetchall()
            # Rename interval_str back to interval
            for row in rows:
                if 'interval_str' in row:
                    row['interval'] = row.pop('interval_str')
            return rows
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # --- Logging ---
    
    def add_log(self, level: str, module: str, message: str, bot_id: Optional[str] = None, user_id: Optional[int] = None):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_logs (level, module, message, bot_id, user_id) VALUES (%s, %s, %s, %s, %s)",
                (level, module, message, bot_id, user_id)
            )
            conn.commit()
        except Error as e:
            # Don't use loguru here to avoid recursion
            pass
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

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
            
            placeholders = ", ".join(["%s"] * len(fields))
            columns = ", ".join(fields)
            
            cursor.execute(
                f"INSERT IGNORE INTO news_sentiment ({columns}) VALUES ({placeholders})",
                data
            )
            conn.commit()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    # --- System State ---
    
    def get_system_state_val(self, key: str) -> Optional[str]:
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT state_value FROM system_state WHERE state_key = %s", (key,))
            row = cursor.fetchone()
            return row[0] if row else None
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def set_system_state_val(self, key: str, value: str):
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO system_state (state_key, state_value) VALUES (%s, %s) "
                "ON DUPLICATE KEY UPDATE state_value = VALUES(state_value)",
                (key, value)
            )
            conn.commit()
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

# Singleton instance
db_manager = DatabaseManager()

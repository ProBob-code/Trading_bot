"""
GodBot Paper Trading Database
==============================

Schema:
- paper_trades: immutable trade records with CHECK constraints
- bot_performance: snapshot metrics with MC/regime/overfit columns
- bot_virtual_wallet: per-bot capital tracking

Supports MySQL and SQLite (auto-detect).
Trades are IMMUTABLE after insertion — no UPDATE allowed.
"""

import os
import json
import sqlite3
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

# Try MySQL
try:
    import MySQLdb
    HAS_MYSQL = True
except ImportError:
    HAS_MYSQL = False


class PaperDB:
    """
    Database manager for the paper trading system.
    Supports MySQL (preferred) and SQLite (fallback).
    """
    
    def __init__(self, db_path: str = "godbot_paper.db"):
        self.db_path = db_path
        self.use_mysql = HAS_MYSQL and os.getenv("MYSQLHOST")
        self._init_db()
    
    def _get_connection(self):
        """Get database connection."""
        if self.use_mysql:
            return MySQLdb.connect(
                host=os.getenv("MYSQLHOST", "localhost"),
                user=os.getenv("MYSQLUSER", "root"),
                passwd=os.getenv("MYSQLPASSWORD", ""),
                db=os.getenv("MYSQLDATABASE", "godbot_paper"),
                port=int(os.getenv("MYSQLPORT", 3306)),
                charset='utf8mb4',
            )
        return sqlite3.connect(self.db_path)
    
    def _execute(self, cursor, sql, params=None):
        """Execute SQL with dialect handling."""
        if not self.use_mysql:
            sql = sql.replace("AUTO_INCREMENT", "AUTOINCREMENT")
            sql = sql.replace("ENGINE=InnoDB", "")
            sql = sql.replace("DOUBLE", "REAL")
            sql = sql.replace("JSON", "TEXT")
            sql = sql.replace("TINYINT", "INTEGER")
            # SQLite doesn't support CHECK constraint enforcement by default
            # but we keep them for documentation
        if params:
            if self.use_mysql:
                cursor.execute(sql, params)
            else:
                sql = sql.replace("%s", "?")
                cursor.execute(sql, params)
        else:
            cursor.execute(sql)
    
    def _init_db(self):
        """Create tables if they don't exist."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # ── paper_trades ──
            # Immutable trade records with CHECK constraints
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS paper_trades (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    trade_id VARCHAR(100) UNIQUE NOT NULL,
                    bot_id VARCHAR(255) NOT NULL,
                    instrument VARCHAR(50) NOT NULL,
                    timeframe VARCHAR(10),
                    side VARCHAR(10) NOT NULL,
                    order_type VARCHAR(10) DEFAULT 'market',
                    
                    entry_price DOUBLE NOT NULL CHECK (entry_price > 0),
                    exit_price DOUBLE,
                    sl_price DOUBLE NOT NULL CHECK (sl_price > 0),
                    tp_price DOUBLE NOT NULL CHECK (tp_price > 0),
                    
                    position_size DOUBLE NOT NULL CHECK (position_size > 0),
                    risk_percent DOUBLE NOT NULL CHECK (risk_percent > 0 AND risk_percent <= 5),
                    r_multiple DOUBLE,
                    
                    slippage_applied DOUBLE DEFAULT 0,
                    fees_paid DOUBLE DEFAULT 0,
                    net_pnl DOUBLE DEFAULT 0,
                    
                    trade_result VARCHAR(10) CHECK (trade_result IN ('WIN','LOSS','BREAKEVEN')),
                    trade_reason TEXT,
                    regime_at_entry VARCHAR(50),
                    indicator_snapshot JSON,
                    
                    timestamp_open TIMESTAMP,
                    timestamp_close TIMESTAMP,
                    bars_held INTEGER DEFAULT 0,
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            # ── bot_performance ──
            # Snapshot metrics per bot including MC and regime columns
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS bot_performance (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    bot_id VARCHAR(255) UNIQUE NOT NULL,
                    bot_name VARCHAR(255),
                    strategy VARCHAR(100),
                    instrument VARCHAR(50),
                    
                    total_trades INTEGER DEFAULT 0,
                    win_rate DOUBLE DEFAULT 0,
                    avg_r DOUBLE DEFAULT 0,
                    expectancy DOUBLE DEFAULT 0,
                    net_expectancy DOUBLE DEFAULT 0,
                    profit_factor DOUBLE DEFAULT 0,
                    
                    sharpe_ratio DOUBLE DEFAULT 0,
                    sortino_ratio DOUBLE DEFAULT 0,
                    calmar_ratio DOUBLE DEFAULT 0,
                    ulcer_index DOUBLE DEFAULT 0,
                    recovery_factor DOUBLE DEFAULT 0,
                    
                    max_drawdown_pct DOUBLE DEFAULT 0,
                    max_consecutive_losses INTEGER DEFAULT 0,
                    risk_of_ruin DOUBLE DEFAULT 0,
                    
                    total_pnl DOUBLE DEFAULT 0,
                    total_return_pct DOUBLE DEFAULT 0,
                    total_slippage DOUBLE DEFAULT 0,
                    total_fees DOUBLE DEFAULT 0,
                    
                    stability_score DOUBLE DEFAULT 0,
                    composite_score DOUBLE DEFAULT 0,
                    safety_label VARCHAR(20) DEFAULT 'CAUTION',
                    
                    trend_expectancy DOUBLE DEFAULT 0,
                    range_expectancy DOUBLE DEFAULT 0,
                    high_vol_expectancy DOUBLE DEFAULT 0,
                    low_vol_expectancy DOUBLE DEFAULT 0,
                    
                    mc_worst_dd DOUBLE DEFAULT 0,
                    mc_95pct_dd DOUBLE DEFAULT 0,
                    mc_risk_of_ruin DOUBLE DEFAULT 0,
                    
                    is_overfit_flagged TINYINT DEFAULT 0,
                    is_negative_expectancy TINYINT DEFAULT 0,
                    
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            # ── bot_virtual_wallet ──
            # Per-bot capital tracking
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS bot_virtual_wallet (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    bot_id VARCHAR(255) UNIQUE NOT NULL,
                    initial_capital DOUBLE NOT NULL DEFAULT 100000,
                    current_equity DOUBLE NOT NULL DEFAULT 100000,
                    peak_equity DOUBLE NOT NULL DEFAULT 100000,
                    total_pnl DOUBLE DEFAULT 0,
                    realized_pnl DOUBLE DEFAULT 0,
                    current_drawdown_pct DOUBLE DEFAULT 0,
                    circuit_breaker_active TINYINT DEFAULT 0,
                    risk_multiplier DOUBLE DEFAULT 1.0,
                    open_positions INTEGER DEFAULT 0,
                    last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            
            conn.commit()
            logger.info("Paper trading database initialized")
        except Exception as e:
            logger.error(f"DB init error: {e}")
            raise
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════
    # Trade Operations (IMMUTABLE after insert)
    # ═══════════════════════════════════════
    
    def insert_trade(self, trade: Dict[str, Any]) -> bool:
        """Insert a validated trade record. Returns True on success."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Serialize indicator snapshot
            if 'indicator_snapshot' in trade and isinstance(trade['indicator_snapshot'], dict):
                trade['indicator_snapshot'] = json.dumps(trade['indicator_snapshot'])
            
            # Convert datetime objects to strings
            for ts_field in ['timestamp_open', 'timestamp_close']:
                if ts_field in trade and isinstance(trade[ts_field], datetime):
                    trade[ts_field] = trade[ts_field].strftime('%Y-%m-%d %H:%M:%S')
            
            fields = list(trade.keys())
            placeholders = ", ".join(["%s"] * len(fields))
            columns = ", ".join(fields)
            values = list(trade.values())
            
            self._execute(cursor,
                f"INSERT INTO paper_trades ({columns}) VALUES ({placeholders})",
                values
            )
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Insert trade error: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()
    
    def get_trades(
        self,
        bot_id: Optional[str] = None,
        instrument: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Retrieve trades with optional filters."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            sql = "SELECT * FROM paper_trades WHERE 1=1"
            params = []
            
            if bot_id:
                sql += " AND bot_id = %s"
                params.append(bot_id)
            if instrument:
                sql += " AND instrument = %s"
                params.append(instrument)
            
            sql += " ORDER BY created_at DESC LIMIT %s OFFSET %s"
            params.extend([limit, offset])
            
            self._execute(cursor, sql, params)
            
            if self.use_mysql:
                columns = [d[0] for d in cursor.description]
            else:
                columns = [d[0] for d in cursor.description]
            
            rows = cursor.fetchall()
            trades = []
            for row in rows:
                trade = dict(zip(columns, row))
                # Parse indicator snapshot
                if 'indicator_snapshot' in trade and isinstance(trade['indicator_snapshot'], str):
                    try:
                        trade['indicator_snapshot'] = json.loads(trade['indicator_snapshot'])
                    except (json.JSONDecodeError, TypeError):
                        pass
                trades.append(trade)
            
            return trades
        finally:
            cursor.close()
            conn.close()
    
    def get_all_trades_for_bot(self, bot_id: str) -> List[Dict[str, Any]]:
        """Get all trades for a specific bot (for metrics calculation)."""
        return self.get_trades(bot_id=bot_id, limit=100000)
    
    def count_trades(self, bot_id: Optional[str] = None) -> int:
        """Count total trades."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if bot_id:
                self._execute(cursor, "SELECT COUNT(*) FROM paper_trades WHERE bot_id = %s", [bot_id])
            else:
                self._execute(cursor, "SELECT COUNT(*) FROM paper_trades")
            return cursor.fetchone()[0]
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════
    # Bot Performance Operations
    # ═══════════════════════════════════════
    
    def upsert_performance(self, bot_id: str, metrics: Dict[str, Any]):
        """Insert or update bot performance metrics."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            
            # Remove non-DB fields
            db_metrics = {k: v for k, v in metrics.items()
                         if k not in ('equity_curve', 'drawdown_curve')}
            db_metrics['bot_id'] = bot_id
            db_metrics['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert bools to int for DB
            for k, v in db_metrics.items():
                if isinstance(v, bool):
                    db_metrics[k] = int(v)
            
            # Check if exists
            self._execute(cursor, "SELECT id FROM bot_performance WHERE bot_id = %s", [bot_id])
            exists = cursor.fetchone()
            
            if exists:
                sets = ", ".join([f"{k} = %s" for k in db_metrics.keys() if k != 'bot_id'])
                vals = [v for k, v in db_metrics.items() if k != 'bot_id']
                vals.append(bot_id)
                self._execute(cursor, f"UPDATE bot_performance SET {sets} WHERE bot_id = %s", vals)
            else:
                fields = list(db_metrics.keys())
                placeholders = ", ".join(["%s"] * len(fields))
                columns = ", ".join(fields)
                self._execute(cursor,
                    f"INSERT INTO bot_performance ({columns}) VALUES ({placeholders})",
                    list(db_metrics.values())
                )
            
            conn.commit()
        except Exception as e:
            logger.error(f"Upsert performance error: {e}")
            conn.rollback()
        finally:
            cursor.close()
            conn.close()
    
    def get_performance(self, bot_id: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get bot performance records."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            if bot_id:
                self._execute(cursor, "SELECT * FROM bot_performance WHERE bot_id = %s", [bot_id])
            else:
                self._execute(cursor, "SELECT * FROM bot_performance ORDER BY composite_score DESC")
            
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════
    # Wallet Operations
    # ═══════════════════════════════════════
    
    def init_wallet(self, bot_id: str, capital: float = 100000.0):
        """Initialize a bot's virtual wallet."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "SELECT id FROM bot_virtual_wallet WHERE bot_id = %s", [bot_id])
            if not cursor.fetchone():
                self._execute(cursor, '''
                    INSERT INTO bot_virtual_wallet
                    (bot_id, initial_capital, current_equity, peak_equity)
                    VALUES (%s, %s, %s, %s)
                ''', [bot_id, capital, capital, capital])
                conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def update_wallet(self, bot_id: str, updates: Dict[str, Any]):
        """Update a bot's wallet after a trade."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            updates['last_updated'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
            
            # Convert bools
            for k, v in updates.items():
                if isinstance(v, bool):
                    updates[k] = int(v)
            
            sets = ", ".join([f"{k} = %s" for k in updates.keys()])
            vals = list(updates.values())
            vals.append(bot_id)
            
            self._execute(cursor,
                f"UPDATE bot_virtual_wallet SET {sets} WHERE bot_id = %s", vals)
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    def get_wallet(self, bot_id: str) -> Optional[Dict[str, Any]]:
        """Get a bot's wallet state."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "SELECT * FROM bot_virtual_wallet WHERE bot_id = %s", [bot_id])
            row = cursor.fetchone()
            if row:
                columns = [d[0] for d in cursor.description]
                return dict(zip(columns, row))
            return None
        finally:
            cursor.close()
            conn.close()
    
    def get_all_wallets(self) -> List[Dict[str, Any]]:
        """Get all bot wallets."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "SELECT * FROM bot_virtual_wallet")
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════
    # Reset (for testing only)
    # ═══════════════════════════════════════
    
    def reset_all(self):
        """Reset all paper trading data. USE WITH CAUTION."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "DELETE FROM paper_trades")
            self._execute(cursor, "DELETE FROM bot_performance")
            self._execute(cursor, "DELETE FROM bot_virtual_wallet")
            conn.commit()
            logger.warning("All paper trading data reset")
        finally:
            cursor.close()
            conn.close()
    
    def reset_bot(self, bot_id: str):
        """Reset data for a specific bot."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, "DELETE FROM paper_trades WHERE bot_id = %s", [bot_id])
            self._execute(cursor, "DELETE FROM bot_performance WHERE bot_id = %s", [bot_id])
            self._execute(cursor, "DELETE FROM bot_virtual_wallet WHERE bot_id = %s", [bot_id])
            conn.commit()
        finally:
            cursor.close()
            conn.close()
    
    # ═══════════════════════════════════════
    # Data Integrity Log
    # ═══════════════════════════════════════
    
    def _init_integrity_table(self):
        """Create data integrity log table."""
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, '''
                CREATE TABLE IF NOT EXISTS data_integrity_log (
                    id INTEGER PRIMARY KEY AUTO_INCREMENT,
                    run_id VARCHAR(255) NOT NULL,
                    data_hash VARCHAR(64) NOT NULL,
                    source_id VARCHAR(255) DEFAULT 'unknown',
                    version_tag VARCHAR(50) DEFAULT 'v1.0',
                    n_bars INTEGER DEFAULT 0,
                    date_range_start VARCHAR(50),
                    date_range_end VARCHAR(50),
                    null_count INTEGER DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                ) ENGINE=InnoDB
            ''')
            conn.commit()
        except Exception as e:
            logger.error(f"Integrity table init error: {e}")
        finally:
            cursor.close()
            conn.close()
    
    def log_integrity(self, stamp: Dict[str, Any]) -> bool:
        """Log a data integrity stamp."""
        self._init_integrity_table()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor, '''
                INSERT INTO data_integrity_log
                (run_id, data_hash, source_id, version_tag, n_bars,
                 date_range_start, date_range_end, null_count)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ''', [
                stamp.get('run_id', ''),
                stamp.get('data_hash', ''),
                stamp.get('source_id', 'unknown'),
                stamp.get('version_tag', 'v1.0'),
                stamp.get('n_bars', 0),
                stamp.get('date_range_start', ''),
                stamp.get('date_range_end', ''),
                stamp.get('null_count', 0),
            ])
            conn.commit()
            return True
        except Exception as e:
            logger.error(f"Log integrity error: {e}")
            conn.rollback()
            return False
        finally:
            cursor.close()
            conn.close()
    
    def get_integrity_log(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get data integrity log entries."""
        self._init_integrity_table()
        conn = self._get_connection()
        try:
            cursor = conn.cursor()
            self._execute(cursor,
                "SELECT * FROM data_integrity_log ORDER BY created_at DESC LIMIT %s",
                [limit])
            columns = [d[0] for d in cursor.description]
            return [dict(zip(columns, row)) for row in cursor.fetchall()]
        finally:
            cursor.close()
            conn.close()


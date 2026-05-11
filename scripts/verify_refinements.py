import sys
import os
import time
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

load_dotenv()

from shared.database.db_manager import DatabaseManager
from decimal import Decimal

def test_crash_recovery():
    logger.info("🧪 Testing Crash Recovery...")
    db = DatabaseManager()
    session_id = f"test_crash_{int(time.time())}"
    
    # 1. Create session and manually set to ACTIVE
    db.v2_create_session(session_id, "v2-verify")
    db.v2_update_session_status(session_id, "ACTIVE")
    
    # 2. Trigger recovery
    db.v2_mark_crashed_sessions()
    
    # 3. Verify
    sessions = db.v2_get_sessions()
    test_session = next((s for s in sessions if s['session_id'] == session_id), None)
    if test_session and test_session['status'] == 'CRASHED':
        logger.success(f"✅ Crash Recovery verified: {session_id} is now CRASHED")
    else:
        logger.error(f"❌ Crash Recovery FAILED: {session_id} status is {test_session['status'] if test_session else 'NotFound'}")

def test_financial_precision():
    logger.info("🧪 Testing Financial Precision (DECIMAL)...")
    db = DatabaseManager()
    session_id = f"test_prec_{int(time.time())}"
    db.v2_create_session(session_id, "v2-verify")
    
    # High precision value (typical of some crypto assets or cumulative PnL)
    high_prec_pnl = Decimal("10.12345678")
    db.v2_update_session_counters(session_id, float(high_prec_pnl))
    
    sessions = db.v2_get_sessions()
    test_session = next((s for s in sessions if s['session_id'] == session_id), None)
    
    if test_session:
        # SQLite returns Decimal as float/string, MySQL as Decimal if using dictionary=True and connector support
        db_pnl = Decimal(str(test_session['total_pnl']))
        if db_pnl == high_prec_pnl:
            logger.success(f"✅ Financial Precision verified: {db_pnl}")
        else:
            logger.error(f"❌ Precision Lost: Sent {high_prec_pnl}, Got {db_pnl}")

def test_indexing():
    logger.info("🧪 Testing Strategy Index...")
    db = DatabaseManager()
    conn = db._get_connection()
    cursor = conn.cursor(dictionary=True)
    
    try:
        cursor.execute("SHOW INDEX FROM v2_trades")
        indexes = [r['Key_name'] for r in cursor.fetchall()]
        if 'idx_strategy_time' in indexes:
            logger.success("✅ Index 'idx_strategy_time' verified")
        else:
            logger.warning("⚠️ Index 'idx_strategy_time' not found (This may be expected in SQLite or if migration skipped)")
    finally:
        db._safe_close(conn, cursor)

def test_edge_filter_logic():
    logger.info("🧪 Testing Minimum Edge Filter Logic...")
    # Mock bot and signal
    class MockBot:
        def __init__(self):
            self.bot_id = "test_bot"
            self.config = type('obj', (object,), {
                'symbol': 'BTCUSDT',
                'user_id': 1,
                'strategy': 'MACD',
                'leverage': 1,
                'take_profit': 2.0,
                'stop_loss': 1.0,
                'position_size': 10.0,
                'max_quantity': 100.0
            })
            self.stats = type('obj', (object,), {'realized_pnl': 0, 'unrealized_pnl': 0})

    class MockSignal:
        def __init__(self, score, expected_move_pct, signal_type='BUY'):
            self.signal = signal_type
            self.score = score
            self.expected_move_pct = expected_move_pct

    from v2.api.routes import v2_execute_bot_trade
    
    bot = MockBot()
    
    # CASE 1: Low Edge (Filtered)
    signal_low = MockSignal(score=0.8, expected_move_pct=0.05) # 0.05% move < 0.1% cost
    logger.info("Checking Low Edge Signal (Should be filtered)...")
    v2_execute_bot_trade(bot, signal_low, 50000, None)
    
    # CASE 2: High Edge (Should proceed)
    signal_high = MockSignal(score=0.8, expected_move_pct=0.5) # 0.5% move > 0.1% cost
    logger.info("Checking High Edge Signal (Should proceed)...")
    # Note: This might trigger _v2_place_trade which would hit the DB. 
    # We just want to see if it passes the gate log.
    v2_execute_bot_trade(bot, signal_high, 50000, None)

if __name__ == "__main__":
    test_crash_recovery()
    test_financial_precision()
    test_indexing()
    test_edge_filter_logic()

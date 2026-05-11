import sys
import os
import time
from datetime import datetime
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables
load_dotenv()

from shared.database.db_manager import DatabaseManager

def test_db_schema():
    logger.info("🧪 Testing Database Schema...")
    db = DatabaseManager()
    
    # Check v2_sessions columns
    conn = db._get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("DESCRIBE v2_sessions")
    columns = {r['Field']: r['Type'] for r in cursor.fetchall()}
    
    required = ['status', 'total_trades', 'total_pnl']
    for col in required:
        if col in columns:
            logger.success(f"✅ Column '{col}' exists in v2_sessions")
        else:
            logger.error(f"❌ Column '{col}' MISSING in v2_sessions")
            
    # Check v2_trades FK
    cursor.execute("SHOW CREATE TABLE v2_trades")
    result = cursor.fetchone()
    create_stmt = result['Create Table']
    if 'CONSTRAINT `fk_v2_session` FOREIGN KEY (`session_id`) REFERENCES `v2_sessions` (`session_id`) ON DELETE CASCADE' in create_stmt:
        logger.success("✅ FK Constraint with ON DELETE CASCADE verified")
    else:
        logger.warning("⚠️ FK Constraint not found or different in CREATE statement")
        
    db._safe_close(conn, cursor)

def test_cascade_delete():
    logger.info("🧪 Testing Cascade Delete...")
    db = DatabaseManager()
    session_id = f"test_cascade_{int(time.time())}"
    
    # 1. Create Session
    db.v2_create_session(session_id, "v2-verify")
    
    # 2. Add Trade
    trade_data = {
        'user_id': 1,
        'trade_id': f"trade_{session_id}",
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'quantity': 1.0,
        'fill_price': 50000,
        'session_id': session_id,
        'trade_type': 'OPEN',
        'trade_time': datetime.utcnow()
    }
    db.v2_save_trade(trade_data)
    
    # Verify trade exists
    trades = db.v2_get_user_trades(1, session_id=session_id)
    if len(trades) == 1:
        logger.success("✅ Trade saved successfully")
    else:
        logger.error("❌ Trade NOT saved")
        return

    # 3. Delete Session
    conn = db._get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM v2_sessions WHERE session_id = %s", (session_id,))
    conn.commit()
    logger.info(f"🗑️ Session {session_id} deleted")
    
    # 4. Verify trades are gone
    trades_left = db.v2_get_user_trades(1, session_id=session_id)
    if len(trades_left) == 0:
        logger.success("✅ Cascade Delete verified: Trades removed automatically")
    else:
        logger.error(f"❌ Cascade Delete FAILED: {len(trades_left)} trades remain")
    
    db._safe_close(conn, cursor)

def test_atomic_counters():
    logger.info("🧪 Testing Atomic Session Counters...")
    db = DatabaseManager()
    session_id = f"test_counters_{int(time.time())}"
    db.v2_create_session(session_id, "v2-verify")
    
    # Mock some close trades
    num_trades = 5
    pnl_per_trade = 10.5
    
    for i in range(num_trades):
        db.v2_update_session_counters(session_id, pnl_per_trade)
        
    # Verify session summary
    sessions = db.v2_get_sessions()
    test_session = next((s for s in sessions if s['session_id'] == session_id), None)
    
    if test_session:
        if test_session['total_trades'] == num_trades and float(test_session['total_pnl']) == num_trades * pnl_per_trade:
            logger.success(f"✅ Atomic Counters verified: {test_session['total_trades']} trades, ${test_session['total_pnl']} PNL")
        else:
            logger.error(f"❌ Counter mismatch: Got {test_session['total_trades']} trades, ${test_session['total_pnl']} PNL")
    else:
        logger.error("❌ Test session not found")

if __name__ == "__main__":
    test_db_schema()
    test_cascade_delete()
    test_atomic_counters()

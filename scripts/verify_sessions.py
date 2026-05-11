import sys
import os
import time
from datetime import datetime, timedelta
from loguru import logger
from dotenv import load_dotenv

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Load environment variables from .env
load_dotenv()

from shared.database.db_manager import DatabaseManager
from shared.services.system_state import SystemStateService

def verify_session_isolation():
    logger.info("🧪 STARTING SESSION ISOLATION VERIFICATION")
    
    db = DatabaseManager()
    state = SystemStateService()
    
    user_id = 1
    
    # 1. Create Session A
    session_a = "VERIFY_A_" + datetime.now().strftime("%H%M%S")
    db.v2_create_session(session_a, "v2.0-test")
    
    # 2. Add trades to Session A
    trade_a1 = {
        'user_id': user_id,
        'trade_id': f"T_{session_a}_1",
        'symbol': 'BTCUSDT',
        'side': 'BUY',
        'fill_price': 50000,
        'quantity': 0.1,
        'session_id': session_a,
        'trade_time': datetime.utcnow() - timedelta(minutes=10),
        'strategy': 'verify_strat',
        'trade_type': 'OPEN'
    }
    db.v2_save_trade(trade_a1)
    
    # 3. Create Session B
    session_b = "VERIFY_B_" + datetime.now().strftime("%H%M%S")
    db.v2_create_session(session_b, "v2.0-test")
    
    # 4. Add trades to Session B
    trade_b1 = {
        'user_id': user_id,
        'trade_id': f"T_{session_b}_1",
        'symbol': 'ETHUSDT',
        'side': 'SELL',
        'fill_price': 3000,
        'quantity': 1.0,
        'session_id': session_b,
        'trade_time': datetime.utcnow(),
        'strategy': 'verify_strat',
        'trade_type': 'CLOSE',
        'realized_pnl': 100,
        'net_pnl': 95
    }
    db.v2_save_trade(trade_b1)
    
    # 5. Verify Isolation via DB helper
    all_trades = db.v2_get_user_trades(user_id)
    trades_a = db.v2_get_user_trades(user_id, session_id=session_a)
    trades_b = db.v2_get_user_trades(user_id, session_id=session_b)
    
    logger.info(f"📊 Total trades found: {len(all_trades)}")
    logger.info(f"📊 Session A trades: {len(trades_a)}")
    logger.info(f"📊 Session B trades: {len(trades_b)}")
    
    assert len(trades_a) == 1, "Session A should have exactly 1 trade"
    assert len(trades_b) == 1, "Session B should have exactly 1 trade"
    assert trades_a[0]['trade_id'] == trade_a1['trade_id'], "Trade ID mismatch in Session A"
    assert trades_b[0]['trade_id'] == trade_b1['trade_id'], "Trade ID mismatch in Session B"
    
    # 6. Verify Session Metadata
    sessions = db.v2_get_sessions()
    verify_a = next((s for s in sessions if s['session_id'] == session_a), None)
    verify_b = next((s for s in sessions if s['session_id'] == session_b), None)
    
    assert verify_a is not None, "Session A metadata missing"
    assert verify_b is not None, "Session B metadata missing"
    logger.info(f"✅ Session A trade count metadata: {verify_a['trade_count']}")
    logger.info(f"✅ Session B trade count metadata: {verify_b['trade_count']}")
    
    # 7. Date Range Filtering
    start_date = (datetime.utcnow() - timedelta(minutes=5)).isoformat()
    trades_range = db.v2_get_user_trades(user_id, start_date=start_date)
    logger.info(f"📊 Trades in last 5 mins: {len(trades_range)}")
    # trade_a1 was 10 mins ago, trade_b1 was now
    assert len(trades_range) >= 1, "Should find at least trade_b1"
    
    logger.success("✨ ALL SESSIONS VERIFICATION TESTS PASSED")

if __name__ == "__main__":
    try:
        verify_session_isolation()
    except Exception as e:
        logger.error(f"❌ VERIFICATION FAILED: {e}")
        sys.exit(1)

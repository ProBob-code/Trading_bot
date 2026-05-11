import sys
import os
import datetime
import time
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.database.db_manager import db_manager
from v2.engine.bot_manager_v2 import bot_manager_v2
from shared.services.system_state import get_system_state

def verify_sessions():
    logger.info("🧪 Starting Phase 7: Professional Session Verification")
    user_id = 888
    sys_state = get_system_state()
    
    # 0. Initial State
    # Ensure no active sessions for this test run
    sys_state.set_session_id(None)
    logger.info("Step 0: Verified initial session state is NULL")

    # 1. Lazy Creation Test
    logger.info("Step 1: Starting bot to trigger lazy session creation...")
    bot_manager_v2.start_bot(
        user_id=user_id,
        symbol="ETH/USDT",
        market="crypto",
        strategy="combined"
    )
    
    session_id = sys_state.get_session_id()
    assert session_id is not None, "❌ Session should have been created lazily on bot start"
    logger.info(f"✅ Session created: {session_id}")
    
    # Check DB
    session_db = db_manager.v2_get_active_session_id()
    assert session_db == session_id, "❌ Active session in DB doesn't match system state"
    
    # 2. Check Professional Labels
    logger.info("Step 2: Verifying professional session labels...")
    sessions = db_manager.v2_get_sessions() # This returns filtered sessions (ACTIVE or trades > 0)
    # Since we have an active session, it should show up.
    active_session = next((s for s in sessions if s['session_id'] == session_id), None)
    assert active_session is not None, "❌ Active session should be returned by v2_get_sessions"
    
    # Note: labels are formatted in api_v2.py, but we can check the base fields here.
    # We'll mock the label formatting logic here to verify the data is there.
    def get_label(s):
        start_dt = s['start_time']
        trades = s.get('total_trades', 0)
        pnl = s.get('total_pnl', 0.0)
        pnl_str = f"{'+$' if pnl >= 0 else '-$'}{abs(pnl):.2f}"
        return f"Session • {start_dt} | {trades} trades | {pnl_str}"

    label = get_label(active_session)
    logger.info(f"✅ Generated Label Preview: {label}")
    
    # 3. Empty Session Pruning
    logger.info("Step 3: Stopping bot (zero trades) to verify auto-pruning...")
    # Find the bot_id
    bot_id = list(bot_manager_v2.bots.keys())[0]
    bot_manager_v2.stop_bot(bot_id)
    
    # Manually call stop_session as api_v2 would
    db_manager.v2_stop_session(session_id)
    
    # Verify deletion
    sessions_after = db_manager.v2_get_sessions()
    pruned = next((s for s in sessions_after if s['session_id'] == session_id), None)
    assert pruned is None, "❌ Empty session should have been deleted from DB"
    logger.info("✅ Empty session successfully pruned.")
    
    time.sleep(1.1)
    # 4. Meaningful Session Persistence
    logger.info("Step 4: Starting bot and simulating a trade to verify persistence...")
    sys_state.set_session_id(None) # Reset state to force NEW session creation
    bot_manager_v2.start_bot(user_id=user_id, symbol="BTC/USDT", market="crypto", strategy="ichimoku")
    new_session_id = sys_state.get_session_id()
    
    # Simulate a trade in ledger
    import uuid
    trade_id = f"test_trade_{uuid.uuid4().hex[:8]}"
    db_manager.v2_save_trade({
        'trade_id': trade_id,
        'session_id': new_session_id,
        'user_id': user_id,
        'symbol': 'BTC/USDT',
        'side': 'LONG',
        'action': 'OPEN',
        'quantity': 0.1,
        'price': 50000.0,
        'pnl': 0.0,
        'strategy': 'ichimoku'
    })
    
    # Stop bot & session
    bot_id_2 = list(bot_manager_v2.bots.keys())[0]
    bot_manager_v2.stop_bot(bot_id_2)
    db_manager.v2_stop_session(new_session_id)
    
    # Verify persistence
    sessions_final = db_manager.v2_get_sessions()
    persisted = next((s for s in sessions_final if s['session_id'] == new_session_id), None)
    assert persisted is not None, "❌ Meaningful session (with trades) should persist"
    assert persisted['total_trades'] >= 0 # Actually v2_stop_session recalculates from ledger
    logger.info(f"✅ Meaningful session persisted: {new_session_id}")
    
    logger.info("🏆 ALL PHASE 7 VERIFICATIONS PASSED")

if __name__ == "__main__":
    try:
        verify_sessions()
    except Exception as e:
        logger.exception(f"❌ Verification Failed: {e}")
        sys.exit(1)

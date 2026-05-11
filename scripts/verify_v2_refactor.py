"""
V2 Institutional Refactor Verification Script
==============================================

Simulates institutional trading flows (OPEN, INCREASE, REVERSE, SL/TP)
and verifies state consistency across Ledger, Positions, and Sessions.
"""

import sys
import os
import uuid
import datetime
import time
from loguru import logger

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shared.database.db_manager import db_manager
from v2.engine.execution.execution_engine import ExecutionEngine
from v2.engine.execution.paper_trader_v2 import PaperTraderV2
from v2.engine.core.risk_engine import RiskEngineV2
from v2.engine.core.portfolio_engine import PortfolioEngineV2
from v2.engine.core.pipeline import TradingPipelineV2

def run_verification():
    logger.info("🧪 Starting V2 Institutional Refactor Verification")
    
    # 0. Setup
    user_id = 999  # Test user
    symbol = "BTC/USDT"
    session_id = f"INST_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Initialize components
    exec_engine = ExecutionEngine(deterministic=True) # Deterministic for testing
    trader = PaperTraderV2(initial_capital=100000, execution_engine=exec_engine)
    risk = RiskEngineV2(db_manager=db_manager)
    portfolio = PortfolioEngineV2(trader, db_manager)
    pipeline = TradingPipelineV2(risk, portfolio, trader, db_manager)
    
    # Mock bot object
    class MockBot:
        def __init__(self, uid, sym, strat):
            self.bot_id = f"test_bot_{uid}"
            class Config: pass
            self.config = Config()
            self.config.user_id = uid
            self.config.symbol = sym
            self.config.strategy = strat
            self.config.leverage = 1.0
            self.config.max_quantity = 0.5
            self.config.stop_loss = 0
            self.config.take_profit = 0
    
    bot = MockBot(user_id, symbol, "TEST_STRAT")
    
    # 1. Create Session
    logger.info(f"Phase 1: Session Creation ({session_id})")
    db_manager.v2_create_session(session_id)
    # Inject session ID into DB manager's active session tracking
    db_manager.v2_get_active_session_id = lambda: session_id
    
    # 2. Case 1: OPEN Position
    logger.info("Phase 2: OPEN Position (LONG)")
    trader.set_prices({symbol: 50000.0}) # MUST SET PRICES FOR PaperTraderV2
    class Signal: 
        signal = 'BUY'
        score = 0.8
        expected_move_pct = 1.0
    
    pipeline.run_tick(bot.bot_id, bot, Signal(), 50000.0, 500.0)
    
    # Verify Position in DB
    pos = db_manager.v2_get_position(user_id, symbol)
    assert pos is not None, "Position should be saved in DB"
    assert pos['side'] == 'LONG', f"Expected LONG, got {pos['side']}"
    assert float(pos['quantity']) == 0.5, f"Expected 0.5, got {pos['quantity']}"
    
    # 3. Case 2: INCREASE Position
    logger.info("Phase 3: INCREASE Position (LONG -> LONG)")
    trader.set_prices({symbol: 51000.0})
    pipeline.run_tick(bot.bot_id, bot, Signal(), 51000.0, 500.0)
    
    pos = db_manager.v2_get_position(user_id, symbol)
    assert float(pos['quantity']) == 1.0, f"Expected 1.0 after increase, got {pos['quantity']}"
    
    # 4. Case 3: REVERSAL (LONG -> SHORT)
    logger.info("Phase 4: REVERSAL (LONG -> SHORT)")
    trader.set_prices({symbol: 52000.0})
    Signal.signal = 'SELL'
    pipeline.run_tick(bot.bot_id, bot, Signal(), 52000.0, 500.0)
    
    pos = db_manager.v2_get_position(user_id, symbol)
    assert pos['side'] == 'SHORT', f"Expected SHORT after reversal, got {pos['side']}"
    # Reversal closes 1.0 and opens 0.5
    assert float(pos['quantity']) == 0.5, f"Expected 0.5 after reversal, got {pos['quantity']}"
    
    # Check Ledger for Closure during Reversal
    trades = db_manager.v2_get_user_trades(user_id, trade_type='CLOSE')
    assert len(trades) >= 1, "Should have at least one closing trade record for the reversal"
    reversal_close = [t for t in trades if t['action'] == 'REVERSAL']
    assert len(reversal_close) > 0, "Should have a REVERSAL action in ledger"
    
    # 5. Case 4: SL Breached
    logger.info("Phase 5: Risk Monitoring (STOP LOSS)")
    # Set SL in trader for this position
    trader_pos = trader._get_account(user_id).positions[symbol]
    trader_pos.stop_loss = 55000.0 # Price is currently ~52k, so if price stays or goes up to 55k for SHORT, it hits SL
    
    # Check risk
    breached, action, reason = risk.check_position_risk(trader_pos.to_dict(), 56000.0)
    assert breached == True, "Should breach SL at 56000 for SHORT"
    assert action == 'STOP_LOSS'
    
    # Execute SL trade manually (as pipeline would)
    res = trader._close_position(trader._get_account(user_id), user_id, symbol, 0.5, trader_pos, 56000.0, 0.02, 100e6, "TEST", action='STOP_LOSS')
    pipeline._handle_trade_results(bot, [res], 'BUY', 0.5, 56000.0)
    
    # 6. Finalize Session
    logger.info("Phase 6: Session Finalization")
    db_manager.v2_stop_session(session_id)
    
    # Verify Session Totals
    sessions = db_manager.v2_get_sessions()
    session = next(s for s in sessions if s['session_id'] == session_id)
    assert session['status'] == 'STOPPED'
    ledger_pnl = db_manager.v2_get_daily_pnl(user_id, datetime.datetime.utcnow().date())
    # Session PnL should match sum of ledger PnL
    # Wait, v2_calculate_pnl_from_ledger sum all pnl for that session
    calculated_pnl = 0
    all_trades = db_manager.v2_get_user_trades(user_id, session_id=session_id)
    for t in all_trades:
        calculated_pnl += float(t['pnl'])
    
    assert round(float(session['total_pnl']), 2) == round(calculated_pnl, 2), \
        f"Session PnL {session['total_pnl']} != Ledger Sum {calculated_pnl}"
        
    logger.info(f"✅ VERIFICATION COMPLETE: PnL={session['total_pnl']}, Trades={session['total_trades']}")
    
    # Cleanup
    # db_manager.v2_clear_user_trades(user_id) # Optional

if __name__ == "__main__":
    try:
        run_verification()
    except Exception as e:
        logger.exception(f"❌ Verification Failed: {e}")
        sys.exit(1)

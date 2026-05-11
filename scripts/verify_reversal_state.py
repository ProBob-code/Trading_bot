import sys
import os
from datetime import datetime
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v2.engine.execution.paper_trader_v2 import PaperTraderV2
from v2.api.routes import _v2_place_trade

def test_atomic_reversal_logic():
    print("\n--- Testing Atomic Reversal Logic ---")
    
    # 1. Setup PaperTrader
    trader = PaperTraderV2(initial_capital=100000)
    trader.set_prices({'BTC/USDT': 70000})
    user_id = 1
    symbol = 'BTC/USDT'
    
    # 2. Open initial LONG position
    print("Step 1: Opening initial LONG...")
    results = trader.execute_trade(user_id, symbol, 'BUY', 1.0, strategy='MACD')
    assert len(results) == 1
    assert results[0]['type'] == 'OPEN'
    assert results[0]['position_side'] == 'LONG'
    assert symbol in trader.accounts[user_id].positions
    
    # 3. Trigger REVERSAL (LONG -> SHORT)
    print("Step 2: Triggering REVERSAL (LONG -> SHORT)...")
    trader.set_prices({'BTC/USDT': 71000}) # Price went up
    results = trader.execute_trade(user_id, symbol, 'SELL', 1.0, strategy='MACD')
    
    # Verify results
    assert len(results) == 2
    assert results[0]['type'] == 'CLOSE'
    assert results[0]['position_side'] == 'LONG'
    
    # PnL will be around 1000 but less due to slippage/commission
    actual_pnl = results[0]['realized_pnl']
    print(f"DEBUG: Actual PnL: {actual_pnl}")
    assert 500 < actual_pnl < 1500
    
    assert results[1]['type'] == 'OPEN'
    assert results[1]['position_side'] == 'SHORT'
    assert trader.accounts[user_id].positions[symbol].side == 'SHORT'
    
    print("✅ Atomic Reversal Logic Verified.")

@patch('api_v2.v2_analytics')
@patch('api_v2.db_manager')
@patch('api_v2.system_state_fn')
@patch('api_v2.bot_manager_v2')
@patch('api_v2.socketio')
def test_ledger_persistence(mock_socket, mock_bot_mgr, mock_sys_state, mock_db, mock_analytics):
    print("\n--- Testing Ledger Persistence & Position Sync ---")
    
    mock_sys_state().get_session_id.return_value = "session_123"
    user_id = 1
    symbol = 'BTC/USDT'
    
    # Mock bot
    bot = MagicMock()
    bot.bot_id = "BOT_001"
    bot.config.user_id = user_id
    bot.config.symbol = symbol
    bot.config.strategy = "MACD"
    bot.config.max_quantity = 1.0
    
    # Trigger a reversal via _v2_place_trade
    # We'll use a real PaperTrader for this test to see the list of results
    with patch('api_v2.v2_paper_trader') as mock_trader:
        mock_trader.execute_trade.return_value = [
            {'success': True, 'type': 'CLOSE', 'side': 'SELL', 'position_side': 'LONG', 'realized_pnl': 500, 'fill_price': 70500, 'quantity': 1.0},
            {'success': True, 'type': 'OPEN', 'side': 'SELL', 'position_side': 'SHORT', 'fill_price': 70500, 'quantity': 1.0}
        ]
        
        _v2_place_trade(bot, 'SELL', 1.0, 70500, 1.0, 'REVERSAL')
        
        # Verify db_manager.v2_save_trade was called TWICE
        assert mock_db.v2_save_trade.call_count == 2
        
        # Verify position sync
        # First call should be delete (for CLOSE)
        mock_db.v2_delete_position.assert_called_with(user_id, symbol)
        # Second call should be save (for OPEN)
        mock_db.v2_save_position.assert_called()
        
        # Verify session counters updated on CLOSE
        mock_db.v2_update_session_counters.assert_called_with("session_123", 500)
        
    print("✅ Ledger Persistence & Position Sync Verified.")

def test_restart_recovery():
    print("\n--- Testing Restart Recovery ---")
    
    db_mock = MagicMock()
    db_mock.v2_get_positions.return_value = [
        {
            'symbol': 'BTC/USDT',
            'side': 'SHORT',
            'quantity': 1.5,
            'avg_price': 65000.0,
            'leverage': 2.0,
            'margin_mode': 'isolated',
            'margin_used': 48750.0,
            'opened_at': '2026-03-14T10:00:00'
        }
    ]
    
    trader = PaperTraderV2()
    trader.load_positions(1, db_mock)
    
    # Verify memory state
    pos = trader.accounts[1].positions['BTC/USDT']
    assert pos.side == 'SHORT'
    assert pos.quantity == 1.5
    assert pos.avg_price == 65000.0
    assert pos.leverage == 2.0
    
    print("✅ Restart Recovery Verified.")

if __name__ == "__main__":
    try:
        test_atomic_reversal_logic()
        test_ledger_persistence()
        test_restart_recovery()
        print("\n✨ ALL STATE MACHINE VERIFICATIONS PASSED ✨")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

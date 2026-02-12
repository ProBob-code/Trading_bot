import sys
import os
import shutil
from pathlib import Path

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.services.trade_logger import TradeLogger

def test_logger():
    # Use a temp directory for testing
    test_dir = Path("test_data")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()
    
    print("🚀 Initializing TradeLogger...")
    logger = TradeLogger(str(test_dir))
    
    # 1. Log ENTRIES
    print("\n📝 Logging Entries (Long & Short)...")
    t1 = logger.log_trade("BTCUSDT", "BUY", 1.0, 50000, bot_id="bot1", mode="paper")
    print(f"entry1: {t1['trade_id']} (Long 1.0 @ 50k)")
    
    t2 = logger.log_trade("ETHUSDT", "SELL", 10.0, 3000, bot_id="bot1", mode="paper") # Short
    print(f"entry2: {t2['trade_id']} (Short 10.0 @ 3k)")
    
    # 2. Log Partial EXIT (Long)
    print("\n📝 Logging Partial Exit (Long)...")
    # Closing 0.4 BTC at 55k -> Profit = 0.4 * (55k - 50k) = 2000
    t3 = logger.log_trade("BTCUSDT", "SELL", 0.4, 55000, pnl=2000, bot_id="bot1", mode="paper")
    
    print(f"exit1: {t3['trade_id']} type={t3['trade_type']} linked={t3['round_trip_id']}")
    assert t3['trade_type'] == 'EXIT'
    assert t3['round_trip_id'] == t1['trade_id'] # Should link to first entry
    
    # 3. Log Full EXIT (Short)
    print("\n📝 Logging Full Exit (Short)...")
    # Closing 10 ETH at 2800 -> Profit = 10 * (3000 - 2800) = 2000
    t4 = logger.log_trade("ETHUSDT", "BUY", 10.0, 2800, pnl=2000, bot_id="bot1", mode="paper")
    
    print(f"exit2: {t4['trade_id']} type={t4['trade_type']} linked={t4['round_trip_id']}")
    assert t4['round_trip_id'] == t2['trade_id']
    
    # 4. Verify Persistence
    print("\n💾 Verifying Persistence...")
    del logger
    
    logger2 = TradeLogger(str(test_dir))
    history = logger2.get_history()
    print(f"Loaded {len(history)} trades from disk.")
    assert len(history) == 4
    
    # Verify Open Positions State Reconstruction
    # BTCUSDT should have 0.6 remaining
    # ETHUSDT should have 0 remaining
    
    key_btc = "paper_bot1_BTCUSDT"
    key_eth = "paper_bot1_ETHUSDT"
    
    q_btc = logger2.open_positions.get(key_btc)
    q_eth = logger2.open_positions.get(key_eth)
    
    print(f"Reconstructed BTC Queue: {len(q_btc) if q_btc else 0} items")
    if q_btc:
        print(f"BTC Remaining Qty: {q_btc[0]['remaining_qty']}")
        assert abs(q_btc[0]['remaining_qty'] - 0.6) < 1e-9
        
    print(f"Reconstructed ETH Queue: {len(q_eth) if q_eth else 0} items")
    assert not q_eth or len(q_eth) == 0
    
    print("\n✅ TEST PASSED!")
    
    # Cleanup
    shutil.rmtree(test_dir)

if __name__ == "__main__":
    test_logger()

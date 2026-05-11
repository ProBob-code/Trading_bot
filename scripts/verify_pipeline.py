import sys
import os
from unittest.mock import MagicMock, patch

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v2.engine.core.risk_engine import RiskEngineV2
from v2.engine.core.portfolio_engine import PortfolioEngineV2
from v2.engine.core.pipeline import TradingPipelineV2

def test_risk_gating():
    print("\n--- Testing Risk Engine Gating ---")
    risk = RiskEngineV2()
    
    # 1. Reject low score
    allowed, reason = risk.pre_trade_gate(1, 'BTC/USDT', 'BUY', 0.5, 0.01, {'allowed': True})
    assert not allowed
    assert "below threshold" in reason
    
    # 2. Reject low edge
    allowed, reason = risk.pre_trade_gate(1, 'BTC/USDT', 'BUY', 0.8, 0.0005, {'allowed': True})
    assert not allowed
    assert "below trading costs" in reason
    
    # 3. Accept good signal
    allowed, reason = risk.pre_trade_gate(1, 'BTC/USDT', 'BUY', 0.9, 0.02, {'allowed': True})
    assert allowed
    print("✅ Risk Gating Verified.")

def test_portfolio_sizing():
    print("\n--- Testing Portfolio Engine Sizing ---")
    portfolio = PortfolioEngineV2(MagicMock(), MagicMock())
    
    config = {'position_size': 100, 'max_quantity': 2.0}
    equity = 100000.0
    
    # 1. ATR-based sizing
    # ATR=500, Price=50000. Risk 1% (1000). 2*ATR = 1000. Qty = 1.0
    qty = portfolio.calculate_units(1, 'BTC/USDT', 50000.0, 500.0, config, equity)
    assert qty == 1.0
    
    # 2. ATR-based sizing (higher risk/vol)
    # ATR=2000, Price=50000. Risk 1% (1000). 2*ATR = 4000. Qty = 0.25
    qty = portfolio.calculate_units(1, 'BTC/USDT', 50000.0, 2000.0, config, equity)
    assert qty == 0.25
    
    print("✅ Portfolio Sizing Verified.")

def test_pipeline_integration():
    print("\n--- Testing Pipeline Integration (Mocked) ---")
    mock_risk = MagicMock()
    mock_risk.pre_trade_gate.return_value = (True, "Passed")
    
    mock_portfolio = MagicMock()
    mock_portfolio.get_position_state.return_value = None # No position
    mock_portfolio.calculate_units.return_value = 0.5
    
    mock_trader = MagicMock()
    mock_trader.get_account_info.return_value = {'equity': 100000}
    mock_trader.execute_trade.return_value = [{'success': True, 'type': 'OPEN', 'side': 'BUY', 'fill_price': 50000, 'quantity': 0.5}]
    
    mock_db = MagicMock()
    mock_db.v2_get_active_session_id.return_value = "session_mock"
    
    pipeline = TradingPipelineV2(mock_risk, mock_portfolio, mock_trader, mock_db)
    
    bot = MagicMock()
    bot.config.user_id = 1
    bot.config.symbol = 'BTC/USDT'
    bot.config.strategy = 'PIPELINE_TEST'
    bot.config.leverage = 1.0
    
    signal = MagicMock()
    signal.signal = 'BUY'
    signal.score = 0.9
    
    pipeline.run_tick("BOT_001", bot, signal, 50000.0, 100.0)
    
    # Verify execution was called
    assert mock_trader.execute_trade.called
    assert mock_portfolio.sync_position.called
    assert mock_db.v2_save_trade.called
    
    print("✅ Pipeline Integration Verified.")

if __name__ == "__main__":
    try:
        test_risk_gating()
        test_portfolio_sizing()
        test_pipeline_integration()
        print("\n✨ ALL PIPELINE VERIFICATIONS PASSED ✨")
    except Exception as e:
        print(f"\n❌ VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

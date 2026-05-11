import unittest
from unittest.mock import MagicMock, patch
import sys

# Mock mysql.connector before importing src.database.db_manager
mock_mysql = MagicMock()
sys.modules['mysql'] = MagicMock()
sys.modules['mysql.connector'] = mock_mysql

from shared.config.settings import is_v1_enabled
from v2.engine.execution.execution_engine import ExecutionEngine
from v2.engine.execution.paper_trader_v2 import PaperTraderV2
from shared.services.trade_logger import get_trade_logger
from v1.engine.godbot.orchestrator import STRATEGY_MAP

class TestV1Disabled(unittest.TestCase):
    def test_flag_is_false(self):
        """Verify that the global flag is indeed False."""
        self.assertFalse(is_v1_enabled(), "V1 trading flag should be False")

    def test_v1_execution_engine_blocked(self):
        """Verify V2 engine blocks V1-sourced trades."""
        engine = ExecutionEngine(deterministic=True)
        result = engine.execute(
            side='BUY',
            market_price=100.0,
            quantity=1.0,
            source='v1'
        )
        self.assertIsNone(result, "V1-sourced trade should be blocked in V2 engine")

    def test_trade_logger_v1_blocked(self):
        """Verify TradeLogger blocks V1 trade logging."""
        # Mock db_manager to avoid DB calls
        with patch('src.services.trade_logger.db_manager') as mock_db:
            logger = get_trade_logger()
            result = logger.log_trade(
                symbol='BTCUSDT',
                side='BUY',
                quantity=1.0,
                price=50000.0,
                user_id=1,
                strategy='v1_strat'
            )
            self.assertEqual(result, {}, "TradeLogger should return empty dict when V1 is disabled")

    def test_strategy_registry_empty(self):
        """Verify V1 strategy registry is empty (or doesn't contain V1 strats)."""
        v1_strats = ['breakout', 'mean_reversion', 'ichimoku', 'ml_forecast']
        for strat in v1_strats:
            self.assertNotIn(strat, STRATEGY_MAP, f"Strategy {strat} should not be in STRATEGY_MAP")

    def test_v2_portfolio_enforcement(self):
        """Verify V2 strict single-position enforcement."""
        engine = ExecutionEngine(deterministic=True)
        trader = PaperTraderV2(initial_capital=100000, execution_engine=engine)
        trader.set_prices({'BTCUSDT': 50000.0})
        
        # 1. Open LONG
        res1 = trader.execute_trade(1, 'BTCUSDT', 'BUY', 1.0)
        self.assertTrue(res1['success'])
        self.assertEqual(res1['position_side'], 'LONG')
        
        # 2. Try another LONG (should BLOCK)
        res2 = trader.execute_trade(1, 'BTCUSDT', 'BUY', 1.0)
        self.assertFalse(res2['success'])
        self.assertEqual(res2.get('status'), 'BLOCKED')
        
        # 3. REVERSE (Open SHORT while LONG)
        res3 = trader.execute_trade(1, 'BTCUSDT', 'SELL', 1.0)
        self.assertTrue(res3['success'])
        self.assertEqual(res3['position_side'], 'SHORT')
        
        # Verify old position is closed and new one is open
        positions = trader.get_positions(1)
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]['side'], 'SHORT')

if __name__ == '__main__':
    unittest.main()

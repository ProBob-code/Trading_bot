import os
import sys
import unittest
from datetime import datetime, timezone
from typing import Dict, List

# Add src to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from v2.engine.execution.paper_trader_v2 import PaperTraderV2
from v2.engine.execution.execution_engine import ExecutionEngine, CommissionModel
from v2.engine.analytics.strategy_analytics import StrategyAnalytics

class TestV2Accounting(unittest.TestCase):
    def setUp(self):
        self.engine = ExecutionEngine()
        self.trader = PaperTraderV2(initial_capital=100000)
        self.analytics = StrategyAnalytics()

    def test_commission_rate(self):
        """Verify commission rate is 0.04% (0.0004)."""
        self.assertEqual(CommissionModel.DEFAULT_RATE, 0.0004)
        
        # Test calculation
        qty = 1.0
        price = 100.0
        comm = self.engine.commission_model.compute(price, qty)
        # 1.0 * 100.0 * 0.0004 = 0.04
        self.assertAlmostEqual(comm, 0.04)

    def test_trade_id_generation(self):
        """Verify unique trade IDs with microsecond precision."""
        id1 = self.trader._generate_trade_id("BTCUSDT", "OPEN")
        id2 = self.trader._generate_trade_id("BTCUSDT", "OPEN")
        
        self.assertNotEqual(id1, id2)
        self.assertIn("BTCUSDT", id1)
        self.assertIn("OPEN", id1)
        # Format: {symbol}_{ts_part1}_{ts_part2}_{ts_part3}_{trade_type}_{uid}
        # ts = YYYYMMDD_HHMMSS_ffffff
        parts = id1.split('_')
        self.assertEqual(len(parts), 6) 

    def test_metrics_pnl_source(self):
        """Verify StrategyAnalytics uses net_pnl and counts breakeven as loss."""
        trades = [
            {'net_pnl': 100.0, 'realized_pnl': 110.0}, # Win
            {'net_pnl': -50.0, 'realized_pnl': -40.0}, # Loss
            {'net_pnl': 0.0, 'realized_pnl': 10.0},   # Breakeven (should be loss)
        ]
        metrics = self.analytics.compute_metrics(trades)
        
        self.assertEqual(metrics['total_trades'], 3)
        self.assertEqual(metrics['wins'], 1)
        self.assertEqual(metrics['losses'], 2)
        self.assertEqual(metrics['win_rate'], 33.33)
        self.assertEqual(metrics['total_pnl'], 50.0) # 100 - 50 + 0

    def test_utc_timestamps(self):
        """Verify timestamps are UTC."""
        import inspect
        source = inspect.getsource(PaperTraderV2._close_position)
        self.assertIn("timezone.utc", source)
        
        source_pos = inspect.getsource(PaperTraderV2._open_position)
        self.assertIn("timezone.utc", source_pos)

if __name__ == '__main__':
    unittest.main()

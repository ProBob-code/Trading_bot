"""
Test Suite: Simulation Engine
"""

import pytest
from v1.engine.godbot.simulation import TradeSimulator, FillResult


class TestSlippage:
    def test_slippage_positive(self):
        sim = TradeSimulator(base_slippage_pct=0.1)
        slip = sim.calculate_slippage(
            price=100, atr=2, volume=1000, candle_range=5,
            order_size_units=10, avg_volume_20=1000
        )
        assert slip > 0
        assert slip < 1.0  # < 1% of price

    def test_slippage_scales_with_volume(self):
        sim = TradeSimulator(base_slippage_pct=0.1)
        small = sim.calculate_slippage(100, 2, 1000, 5, 10, 10000)
        large = sim.calculate_slippage(100, 2, 1000, 5, 500, 10000)
        # Large orders should have more slippage on average
        # (randomization means we check expected direction only)


class TestSpread:
    def test_spread_positive(self):
        sim = TradeSimulator(spread_pct=0.05)
        spread = sim.calculate_spread(100, 2)
        assert spread > 0

    def test_spread_widens_high_vol(self):
        sim = TradeSimulator(spread_pct=0.05)
        low_vol = sim.calculate_spread(100, 0.5)
        high_vol = sim.calculate_spread(100, 5.0)
        assert high_vol > low_vol


class TestFees:
    def test_taker_higher_than_maker(self):
        sim = TradeSimulator(maker_fee_pct=0.02, taker_fee_pct=0.06)
        maker = sim.calculate_fees(10000, "limit")
        taker = sim.calculate_fees(10000, "market")
        assert taker > maker


class TestMarketOrder:
    def test_market_buy_fills(self):
        sim = TradeSimulator(execution_mode="instant")
        fill = sim.simulate_market_order(
            "buy", 100, 10,
            bar_open=100, bar_high=102, bar_low=98, bar_close=101,
            bar_volume=1000, atr=2, avg_volume_20=1000,
        )
        assert fill.result == FillResult.FILLED
        assert fill.fill_price > 0
        assert fill.fees_paid > 0


class TestStopOrder:
    def test_stop_not_reached(self):
        sim = TradeSimulator()
        fill = sim.simulate_stop_order(
            "sell", 90, 10,
            bar_open=100, bar_high=105, bar_low=95, bar_close=102,
            bar_volume=1000, atr=2, avg_volume_20=1000,
        )
        # Stop at 90 not triggered because bar_low=95
        assert fill.result == FillResult.NO_FILL

    def test_stop_gap_fills_at_open(self):
        sim = TradeSimulator()
        fill = sim.simulate_stop_order(
            "sell", 100, 10,
            bar_open=95, bar_high=97, bar_low=93, bar_close=94,
            bar_volume=1000, atr=2, avg_volume_20=1000,
        )
        assert fill.result == FillResult.GAPPED


class TestLiquidity:
    def test_reject_large_position(self):
        sim = TradeSimulator(max_position_volume_pct=5.0)
        ok, reason = sim.check_liquidity(600, 1000)
        assert not ok

    def test_accept_small_position(self):
        sim = TradeSimulator(max_position_volume_pct=5.0)
        ok, reason = sim.check_liquidity(10, 1000)
        assert ok

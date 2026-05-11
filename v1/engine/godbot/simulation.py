"""
Trade Simulation Engine
=======================

Realistic paper trading execution simulator.
Models slippage, spread, fees, fill probability, execution delay, and gap risk.

Slippage = f(Volatility, Volume, Candle Range, Order Size)
Supports: Market, Limit, Stop orders with gap risk model.
"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple
from enum import Enum


class FillResult(Enum):
    """Order fill outcome."""
    FILLED = "filled"
    PARTIAL = "partial"
    NO_FILL = "no_fill"
    GAPPED = "gapped"


@dataclass
class SimulatedFill:
    """Result of a simulated order execution."""
    result: FillResult
    fill_price: float = 0.0
    slippage_applied: float = 0.0
    spread_applied: float = 0.0
    fees_paid: float = 0.0
    fill_probability: float = 0.0
    execution_bar_offset: int = 0       # 0 = same bar, 1 = next bar
    reason: str = ""


class TradeSimulator:
    """
    Realistic trade execution simulator.
    
    Models:
    - Slippage: f(volatility, volume, candle_range, order_size)
    - Spread: per-instrument, widens in high volatility
    - Fees: maker/taker model
    - Fill probability: volume-weighted
    - Execution delay: instant / next_bar_open / random
    - Gap risk for stops: fills at open if gap beyond SL
    """
    
    def __init__(
        self,
        base_slippage_pct: float = 0.05,
        spread_pct: float = 0.02,
        maker_fee_pct: float = 0.02,
        taker_fee_pct: float = 0.06,
        execution_mode: str = "next_bar_open",
        max_position_volume_pct: float = 5.0,
    ):
        self.base_slippage_pct = base_slippage_pct
        self.spread_pct = spread_pct
        self.maker_fee_pct = maker_fee_pct
        self.taker_fee_pct = taker_fee_pct
        self.execution_mode = execution_mode
        self.max_position_volume_pct = max_position_volume_pct
    
    def calculate_slippage(
        self,
        price: float,
        atr: float,
        volume: float,
        candle_range: float,
        order_size_units: float,
        avg_volume_20: float,
    ) -> float:
        """
        Calculate slippage as: f(volatility, volume, candle_range, order_size).
        
        - High volatility → exponential scaling
        - Large order vs volume → higher slippage
        - Wide candle → more slippage room
        """
        if price <= 0:
            return 0.0
        
        # Base slippage in price terms
        base = price * (self.base_slippage_pct / 100.0)
        
        # Volatility factor: ATR relative to price (exponential scaling in high vol)
        atr_pct = (atr / price) if price > 0 else 0
        vol_factor = 1.0 + math.exp(min(atr_pct * 20 - 1, 5))  # Caps at ~150x
        
        # Order size impact: larger orders relative to volume → more slippage
        if avg_volume_20 > 0:
            size_ratio = order_size_units / avg_volume_20
            size_factor = 1.0 + (size_ratio * 10)  # Linear scaling
        else:
            size_factor = 2.0  # No volume data → assume impact
        
        # Candle range factor: wider candles = more room for slippage
        if price > 0:
            range_pct = candle_range / price
            range_factor = 1.0 + range_pct * 5
        else:
            range_factor = 1.0
        
        # Combined slippage with randomization ±30%
        raw_slippage = base * vol_factor * size_factor * range_factor
        randomized = raw_slippage * random.uniform(0.7, 1.3)
        
        # Cap at 1% of price to prevent absurd slippage
        max_slippage = price * 0.01
        return min(randomized, max_slippage)
    
    def calculate_spread(self, price: float, atr: float) -> float:
        """
        Spread simulation. Widens under high volatility.
        """
        base_spread = price * (self.spread_pct / 100.0)
        atr_pct = (atr / price) if price > 0 else 0
        # Spread widens up to 3x in high vol
        vol_multiplier = 1.0 + min(atr_pct * 10, 2.0)
        return base_spread * vol_multiplier
    
    def calculate_fees(self, trade_value: float, order_type: str = "market") -> float:
        """Calculate execution fees (maker/taker model)."""
        if order_type == "limit":
            return trade_value * (self.maker_fee_pct / 100.0)
        return trade_value * (self.taker_fee_pct / 100.0)
    
    def check_fill_probability(
        self,
        order_type: str,
        order_price: float,
        bar_low: float,
        bar_high: float,
        bar_volume: float,
        order_size_units: float,
        avg_volume_20: float,
    ) -> Tuple[bool, float]:
        """
        Determine if an order would fill on this bar.
        
        Market: always fills (within bar range)
        Limit: fills if price reached, probability = volume-weighted
        Stop: fills if trigger hit
        
        Returns (would_fill, probability).
        """
        if order_type == "market":
            return True, 1.0
        
        # Check price reached
        if order_type == "limit":
            # Buy limit: fills if bar_low <= order_price
            # Sell limit: fills if bar_high >= order_price
            price_reached = bar_low <= order_price <= bar_high
        elif order_type == "stop":
            price_reached = bar_low <= order_price <= bar_high
        else:
            price_reached = True
        
        if not price_reached:
            return False, 0.0
        
        # Volume-weighted fill probability
        if avg_volume_20 > 0 and bar_volume > 0:
            volume_ratio = bar_volume / avg_volume_20
            size_impact = order_size_units / bar_volume if bar_volume > 0 else 1.0
            # Higher volume → higher fill prob; larger order → lower fill prob
            prob = min(1.0, volume_ratio * 0.8) * max(0.1, 1.0 - size_impact)
        else:
            prob = 0.7  # Default fill probability
        
        return random.random() < prob, prob
    
    def simulate_market_order(
        self,
        side: str,
        intended_price: float,
        position_size_units: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        atr: float,
        avg_volume_20: float,
    ) -> SimulatedFill:
        """
        Simulate a market order execution.
        
        For next_bar_open execution: base price is bar_open of next bar.
        For instant: base price is intended_price (candle close).
        """
        # Determine base fill price based on execution mode
        if self.execution_mode == "next_bar_open":
            base_price = bar_open
            bar_offset = 1
        elif self.execution_mode == "random_delay_ms":
            # Random between open and close
            base_price = bar_open + random.random() * (bar_close - bar_open)
            bar_offset = 1
        else:  # instant
            base_price = intended_price
            bar_offset = 0
        
        candle_range = bar_high - bar_low
        
        # Calculate slippage
        slippage = self.calculate_slippage(
            base_price, atr, bar_volume, candle_range,
            position_size_units, avg_volume_20
        )
        
        # Calculate spread (half-spread impact)
        spread = self.calculate_spread(base_price, atr) / 2.0
        
        # Apply: buy → price goes up; sell → price goes down
        if side == "buy":
            fill_price = base_price + slippage + spread
        else:
            fill_price = base_price - slippage - spread
        
        # Clamp to candle range (can't fill outside the bar)
        fill_price = max(bar_low, min(bar_high, fill_price))
        
        # Fees
        trade_value = fill_price * position_size_units
        fees = self.calculate_fees(trade_value, "market")
        
        return SimulatedFill(
            result=FillResult.FILLED,
            fill_price=round(fill_price, 8),
            slippage_applied=round(slippage, 8),
            spread_applied=round(spread, 8),
            fees_paid=round(fees, 8),
            fill_probability=1.0,
            execution_bar_offset=bar_offset,
            reason=f"Market {side} filled at {fill_price:.4f} (base={base_price:.4f}, "
                   f"slip={slippage:.4f}, spread={spread:.4f})"
        )
    
    def simulate_stop_order(
        self,
        side: str,
        stop_price: float,
        position_size_units: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        atr: float,
        avg_volume_20: float,
    ) -> SimulatedFill:
        """
        Simulate a stop order with gap risk.
        
        If bar opens beyond stop price (gap) → fills at open, not stop.
        This is critical for crypto overnight moves.
        """
        # Check for gap
        gapped = False
        if side == "sell":  # Stop loss for long
            if bar_open < stop_price:
                # Gapped below stop → fill at open (worse than stop)
                gapped = True
                base_price = bar_open
            elif bar_low <= stop_price:
                base_price = stop_price
            else:
                return SimulatedFill(
                    result=FillResult.NO_FILL,
                    reason="Stop price not reached"
                )
        else:  # Stop loss for short (buy stop)
            if bar_open > stop_price:
                gapped = True
                base_price = bar_open
            elif bar_high >= stop_price:
                base_price = stop_price
            else:
                return SimulatedFill(
                    result=FillResult.NO_FILL,
                    reason="Stop price not reached"
                )
        
        candle_range = bar_high - bar_low
        
        # Slippage (higher for gapped fills)
        slippage = self.calculate_slippage(
            base_price, atr, bar_volume, candle_range,
            position_size_units, avg_volume_20
        )
        if gapped:
            slippage *= 1.5  # Worse execution on gaps
        
        spread = self.calculate_spread(base_price, atr) / 2.0
        
        if side == "sell":
            fill_price = base_price - slippage - spread
        else:
            fill_price = base_price + slippage + spread
        
        fill_price = max(bar_low, min(bar_high, fill_price))
        trade_value = fill_price * position_size_units
        fees = self.calculate_fees(trade_value, "market")  # Stops execute as market
        
        return SimulatedFill(
            result=FillResult.GAPPED if gapped else FillResult.FILLED,
            fill_price=round(fill_price, 8),
            slippage_applied=round(slippage, 8),
            spread_applied=round(spread, 8),
            fees_paid=round(fees, 8),
            fill_probability=1.0,
            execution_bar_offset=0,
            reason=f"Stop {'GAPPED' if gapped else 'filled'} at {fill_price:.4f} "
                   f"(stop={stop_price:.4f})"
        )
    
    def simulate_limit_order(
        self,
        side: str,
        limit_price: float,
        position_size_units: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        bar_volume: float,
        atr: float,
        avg_volume_20: float,
    ) -> SimulatedFill:
        """
        Simulate a limit order with fill probability model.
        """
        # Check fill probability
        would_fill, prob = self.check_fill_probability(
            "limit", limit_price, bar_low, bar_high,
            bar_volume, position_size_units, avg_volume_20
        )
        
        if not would_fill:
            return SimulatedFill(
                result=FillResult.NO_FILL,
                fill_probability=prob,
                reason=f"Limit order not filled (prob={prob:.2%})"
            )
        
        # Limit orders get better price + lower fees (maker)
        fill_price = limit_price  # Fills at limit or better
        spread = self.calculate_spread(fill_price, atr) / 4.0  # Reduced spread for limits
        
        if side == "buy":
            fill_price = limit_price + spread  # Slight worsening
        else:
            fill_price = limit_price - spread
        
        fill_price = max(bar_low, min(bar_high, fill_price))
        trade_value = fill_price * position_size_units
        fees = self.calculate_fees(trade_value, "limit")
        
        return SimulatedFill(
            result=FillResult.FILLED,
            fill_price=round(fill_price, 8),
            slippage_applied=0.0,  # No slippage on limits
            spread_applied=round(spread, 8),
            fees_paid=round(fees, 8),
            fill_probability=prob,
            execution_bar_offset=0,
            reason=f"Limit {side} filled at {fill_price:.4f} (limit={limit_price:.4f}, "
                   f"prob={prob:.2%})"
        )
    
    def check_liquidity(
        self,
        position_size_units: float,
        avg_volume_20: float,
    ) -> Tuple[bool, str]:
        """
        Liquidity filter: reject if position > X% of 20-bar avg volume.
        """
        if avg_volume_20 <= 0:
            return False, "No volume data available"
        
        volume_pct = (position_size_units / avg_volume_20) * 100
        if volume_pct > self.max_position_volume_pct:
            return False, (
                f"Position ({position_size_units:.2f}) is {volume_pct:.1f}% of "
                f"20-bar avg volume ({avg_volume_20:.0f}), exceeds "
                f"{self.max_position_volume_pct}% limit"
            )
        return True, "OK"

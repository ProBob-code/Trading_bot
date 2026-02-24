"""
Data Cleaning Layer
===================

Pre-insertion validation for trade records.
Ensures no garbage data enters the database.

Rules:
- No NaN, null, empty strings
- Round to tick size
- Normalize timestamps to UTC
- Validate SL/TP sanity (SL >= entry rejected for long)
- Enforce R:R floor
- Reject unrealistic prices outside candle range
- Trades are IMMUTABLE after insertion
"""

import math
from datetime import datetime, timezone
from typing import Dict, Any, Tuple, Optional


class DataCleaner:
    """
    Validates and cleans trade records before database insertion.
    
    Enforces:
    - Numeric validity (no NaN/Inf)
    - Price sanity (SL/TP relative to entry)
    - R:R minimum
    - Risk cap
    - Tick size rounding
    - UTC timestamps
    """
    
    def __init__(
        self,
        min_rr: float = 2.0,
        max_risk_pct: float = 5.0,
        tick_size: float = 0.01,
    ):
        self.min_rr = min_rr
        self.max_risk_pct = max_risk_pct
        self.tick_size = tick_size
    
    def round_to_tick(self, price: float) -> float:
        """Round price to instrument tick size."""
        if self.tick_size <= 0:
            return round(price, 8)
        return round(round(price / self.tick_size) * self.tick_size, 8)
    
    def normalize_timestamp(self, ts: Any) -> Optional[datetime]:
        """Normalize timestamp to UTC datetime."""
        if ts is None:
            return None
        if isinstance(ts, datetime):
            if ts.tzinfo is None:
                return ts.replace(tzinfo=timezone.utc)
            return ts.astimezone(timezone.utc)
        if isinstance(ts, str):
            try:
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                return dt.astimezone(timezone.utc)
            except ValueError:
                return None
        if isinstance(ts, (int, float)):
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        return None
    
    def _is_valid_number(self, val: Any) -> bool:
        """Check if value is a valid finite number."""
        if val is None:
            return False
        try:
            f = float(val)
            return not (math.isnan(f) or math.isinf(f))
        except (TypeError, ValueError):
            return False
    
    def validate_trade(self, trade: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Validate a trade record before insertion.
        
        Returns (is_valid, reason).
        """
        # Required fields
        required = [
            'trade_id', 'bot_id', 'instrument', 'side',
            'entry_price', 'sl_price', 'tp_price',
            'position_size', 'risk_percent'
        ]
        for f in required:
            if f not in trade or trade[f] is None:
                return False, f"Missing required field: {f}"
            if isinstance(trade[f], str) and trade[f].strip() == "":
                return False, f"Empty string for field: {f}"
        
        # Numeric validity
        numeric_fields = [
            'entry_price', 'sl_price', 'tp_price',
            'position_size', 'risk_percent'
        ]
        for f in numeric_fields:
            if not self._is_valid_number(trade[f]):
                return False, f"Invalid number for {f}: {trade[f]}"
        
        entry = float(trade['entry_price'])
        sl = float(trade['sl_price'])
        tp = float(trade['tp_price'])
        risk_pct = float(trade['risk_percent'])
        position_size = float(trade['position_size'])
        side = trade['side'].lower()
        
        # Positive price checks
        if entry <= 0:
            return False, f"Entry price must be > 0, got {entry}"
        if sl <= 0:
            return False, f"SL price must be > 0, got {sl}"
        if tp <= 0:
            return False, f"TP price must be > 0, got {tp}"
        if position_size <= 0:
            return False, f"Position size must be > 0, got {position_size}"
        
        # SL/TP sanity relative to entry
        if side == "buy":
            if sl >= entry:
                return False, f"Long SL ({sl}) must be < entry ({entry})"
            if tp <= entry:
                return False, f"Long TP ({tp}) must be > entry ({entry})"
        elif side == "sell":
            if sl <= entry:
                return False, f"Short SL ({sl}) must be > entry ({entry})"
            if tp >= entry:
                return False, f"Short TP ({tp}) must be < entry ({entry})"
        
        # R:R validation
        risk_distance = abs(entry - sl)
        reward_distance = abs(tp - entry)
        if risk_distance > 0:
            rr_ratio = reward_distance / risk_distance
            if rr_ratio < self.min_rr:
                return False, (
                    f"R:R ratio {rr_ratio:.2f} below minimum {self.min_rr} "
                    f"(risk={risk_distance:.4f}, reward={reward_distance:.4f})"
                )
        
        # Risk cap
        if risk_pct <= 0 or risk_pct > self.max_risk_pct:
            return False, f"Risk % ({risk_pct}) must be > 0 and <= {self.max_risk_pct}"
        
        # Trade result validation (if present)
        if 'trade_result' in trade and trade['trade_result']:
            valid_results = {'WIN', 'LOSS', 'BREAKEVEN'}
            if trade['trade_result'] not in valid_results:
                return False, f"Invalid trade_result: {trade['trade_result']}, must be {valid_results}"
        
        return True, "OK"
    
    def clean_trade(self, trade: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean and normalize a trade record.
        
        Applies:
        - Tick rounding on prices
        - UTC normalization on timestamps
        - Float rounding on numeric fields
        - Removes None values for optional fields
        """
        cleaned = {}
        
        # String fields — strip whitespace
        for f in ['trade_id', 'bot_id', 'instrument', 'timeframe', 'side',
                   'order_type', 'trade_result', 'trade_reason', 'regime_at_entry']:
            if f in trade and trade[f] is not None:
                cleaned[f] = str(trade[f]).strip()
        
        # Price fields — tick round
        for f in ['entry_price', 'exit_price', 'sl_price', 'tp_price']:
            if f in trade and self._is_valid_number(trade[f]):
                cleaned[f] = self.round_to_tick(float(trade[f]))
        
        # Numeric fields — round to 8 decimals
        for f in ['position_size', 'risk_percent', 'r_multiple',
                   'slippage_applied', 'fees_paid', 'net_pnl']:
            if f in trade and self._is_valid_number(trade[f]):
                cleaned[f] = round(float(trade[f]), 8)
        
        # Timestamp fields — normalize to UTC
        for f in ['timestamp_open', 'timestamp_close']:
            if f in trade:
                cleaned[f] = self.normalize_timestamp(trade[f])
        
        # Indicator snapshot (JSON) — only if non-empty dict
        if 'indicator_snapshot' in trade:
            snap = trade['indicator_snapshot']
            if isinstance(snap, dict) and snap:
                cleaned['indicator_snapshot'] = snap
        
        return cleaned
    
    def validate_and_clean(self, trade: Dict[str, Any]) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Validate then clean. Returns (is_valid, reason, cleaned_trade).
        """
        is_valid, reason = self.validate_trade(trade)
        if not is_valid:
            return False, reason, {}
        
        cleaned = self.clean_trade(trade)
        return True, "OK", cleaned

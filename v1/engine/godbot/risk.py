"""
Risk Management Engine
======================

Position sizing, liquidity filter, correlation control,
capital decay ladder, daily loss limits, drawdown circuit breaker,
and regime drift detection.
"""

import math
import numpy as np
from typing import Dict, Any, List, Tuple, Optional
from collections import defaultdict


class RiskManager:
    """
    Multi-bot risk management.
    
    Enforces:
    - Position sizing: (capital × risk%) / SL_distance
    - Volatility-adjusted sizing (ATR scaling)
    - Liquidity filter: reject if size > X% of 20-bar avg volume
    - Correlation control: max instrument/directional exposure
    - Capital decay ladder: graduated risk reduction under drawdown
    - Max concurrent trades per bot
    - Global daily loss limit
    """
    
    def __init__(
        self,
        max_daily_loss_pct: float = 5.0,
        max_instrument_exposure_pct: float = 60.0,
        max_directional_exposure_pct: float = 80.0,
    ):
        self.max_daily_loss_pct = max_daily_loss_pct
        self.max_instrument_exposure_pct = max_instrument_exposure_pct
        self.max_directional_exposure_pct = max_directional_exposure_pct
        
        # Daily tracking
        self._daily_pnl: float = 0.0
        self._daily_initial_capital: float = 0.0
        self._is_daily_stopped: bool = False
        
        # Open position tracking for correlation
        self._open_positions: List[Dict[str, Any]] = []
    
    def calculate_position_size(
        self,
        capital: float,
        risk_pct: float,
        entry_price: float,
        sl_price: float,
        atr: float = 0,
        volatility_adjust: bool = False,
    ) -> Dict[str, Any]:
        """
        Calculate position size based on risk.
        
        size = (capital × risk%) / SL_distance
        
        If volatility_adjust=True, scales risk by ATR relative to price.
        """
        sl_distance = abs(entry_price - sl_price)
        if sl_distance <= 0 or entry_price <= 0:
            return {'units': 0, 'value': 0, 'risk_amount': 0, 'rejected': True, 'reason': 'Invalid SL distance'}
        
        effective_risk_pct = risk_pct
        
        # Volatility adjustment: reduce size in high vol
        if volatility_adjust and atr > 0 and entry_price > 0:
            atr_pct = atr / entry_price
            # Normal ATR ~1-2% for most assets
            if atr_pct > 0.03:  # High vol
                effective_risk_pct *= 0.5
            elif atr_pct > 0.02:
                effective_risk_pct *= 0.75
        
        risk_amount = capital * (effective_risk_pct / 100.0)
        units = risk_amount / sl_distance
        value = units * entry_price
        
        return {
            'units': round(units, 8),
            'value': round(value, 2),
            'risk_amount': round(risk_amount, 2),
            'effective_risk_pct': round(effective_risk_pct, 4),
            'sl_distance': round(sl_distance, 8),
            'rejected': False,
            'reason': 'OK',
        }
    
    def apply_drawdown_ladder(
        self,
        current_equity: float,
        peak_equity: float,
        base_risk_pct: float,
    ) -> Tuple[float, bool]:
        """
        Capital decay ladder: graduated risk reduction under drawdown.
        
        DD > 10% → risk × 0.75
        DD > 15% → risk × 0.50
        DD > 20% → risk × 0.25
        
        Returns (adjusted_risk_pct, circuit_breaker_active).
        """
        if peak_equity <= 0:
            return base_risk_pct, False
        
        dd_pct = ((peak_equity - current_equity) / peak_equity) * 100
        
        if dd_pct > 20:
            return base_risk_pct * 0.25, True
        elif dd_pct > 15:
            return base_risk_pct * 0.50, True
        elif dd_pct > 10:
            return base_risk_pct * 0.75, False
        
        return base_risk_pct, False
    
    def check_liquidity(
        self,
        position_size_units: float,
        avg_volume_20: float,
        max_volume_pct: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Reject if position > X% of 20-bar average volume.
        """
        if avg_volume_20 <= 0:
            return False, "No volume data"
        
        pct = (position_size_units / avg_volume_20) * 100
        if pct > max_volume_pct:
            return False, f"Position {pct:.1f}% of avg volume exceeds {max_volume_pct}% limit"
        return True, "OK"
    
    def check_correlation(
        self,
        instrument: str,
        side: str,
        position_value: float,
        total_capital: float,
    ) -> Tuple[bool, str]:
        """
        Correlation risk control.
        
        Limits:
        - Max instrument exposure: max_instrument_exposure_pct of total capital
        - Max directional exposure: max_directional_exposure_pct of total capital
        """
        if total_capital <= 0:
            return False, "No capital"
        
        # Current exposure by instrument
        instrument_exposure = sum(
            p['value'] for p in self._open_positions
            if p['instrument'] == instrument
        )
        
        # Current directional exposure
        directional_exposure = sum(
            p['value'] for p in self._open_positions
            if p['side'] == side
        )
        
        # Check instrument limit
        new_instr_exposure_pct = ((instrument_exposure + position_value) / total_capital) * 100
        if new_instr_exposure_pct > self.max_instrument_exposure_pct:
            return False, (
                f"Instrument exposure {new_instr_exposure_pct:.1f}% exceeds "
                f"{self.max_instrument_exposure_pct}% limit"
            )
        
        # Check directional limit
        new_dir_exposure_pct = ((directional_exposure + position_value) / total_capital) * 100
        if new_dir_exposure_pct > self.max_directional_exposure_pct:
            return False, (
                f"Directional ({side}) exposure {new_dir_exposure_pct:.1f}% exceeds "
                f"{self.max_directional_exposure_pct}% limit"
            )
        
        return True, "OK"
    
    def register_position(self, instrument: str, side: str, value: float, bot_id: str):
        """Register an open position for correlation tracking."""
        self._open_positions.append({
            'instrument': instrument,
            'side': side,
            'value': value,
            'bot_id': bot_id,
        })
    
    def remove_position(self, bot_id: str, instrument: str):
        """Remove a closed position."""
        self._open_positions = [
            p for p in self._open_positions
            if not (p['bot_id'] == bot_id and p['instrument'] == instrument)
        ]
    
    def check_concurrent_trades(
        self,
        bot_id: str,
        max_concurrent: int,
    ) -> Tuple[bool, str]:
        """Check if bot has too many open trades."""
        count = sum(1 for p in self._open_positions if p['bot_id'] == bot_id)
        if count >= max_concurrent:
            return False, f"Bot has {count} open trades (max {max_concurrent})"
        return True, "OK"
    
    def record_daily_pnl(self, pnl: float):
        """Record a trade PnL for daily loss tracking."""
        self._daily_pnl += pnl
    
    def check_daily_loss(self, total_capital: float) -> Tuple[bool, str]:
        """Check if daily loss limit is breached."""
        if total_capital <= 0:
            return False, "No capital"
        
        daily_loss_pct = abs(min(self._daily_pnl, 0)) / total_capital * 100
        if daily_loss_pct >= self.max_daily_loss_pct:
            self._is_daily_stopped = True
            return False, f"Daily loss {daily_loss_pct:.1f}% exceeds {self.max_daily_loss_pct}% limit"
        return True, "OK"
    
    def reset_daily(self):
        """Reset daily tracking (call at start of each day)."""
        self._daily_pnl = 0.0
        self._is_daily_stopped = False
    
    @property
    def is_daily_stopped(self) -> bool:
        return self._is_daily_stopped
    
    def full_risk_check(
        self,
        bot_id: str,
        instrument: str,
        side: str,
        entry_price: float,
        sl_price: float,
        position_size_units: float,
        position_value: float,
        current_equity: float,
        peak_equity: float,
        total_capital_all_bots: float,
        avg_volume_20: float,
        max_concurrent: int = 3,
        max_volume_pct: float = 5.0,
    ) -> Tuple[bool, str]:
        """
        Run all risk checks in sequence.
        Returns (passed, reason) where reason is the first failure.
        """
        # Daily loss
        ok, reason = self.check_daily_loss(total_capital_all_bots)
        if not ok:
            return False, reason
        
        # Concurrent trades
        ok, reason = self.check_concurrent_trades(bot_id, max_concurrent)
        if not ok:
            return False, reason
        
        # Liquidity
        ok, reason = self.check_liquidity(position_size_units, avg_volume_20, max_volume_pct)
        if not ok:
            return False, reason
        
        # Correlation
        ok, reason = self.check_correlation(instrument, side, position_value, total_capital_all_bots)
        if not ok:
            return False, reason
        
        return True, "OK"


class RegimeDriftDetector:
    """
    Rolling regime drift tracking.
    
    Tracks rolling expectancy per regime and computes z-score
    relative to historical regime mean. Early warning if
    z-score < -2.0 (bot underperforming its regime benchmark).
    """
    
    def __init__(self, rolling_window: int = 30):
        self.rolling_window = rolling_window
    
    def detect(
        self,
        trades: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Compute regime drift z-scores.
        
        Args:
            trades: List of trade dicts with regime_at_entry, net_pnl
            
        Returns:
            Dict with per-regime drift analysis and warnings
        """
        if len(trades) < self.rolling_window:
            return {
                'regimes': [],
                'warnings': [],
                'overall_drift': False,
            }
        
        # Group by regime
        regime_trades: Dict[str, List[float]] = defaultdict(list)
        for t in trades:
            regime = t.get('regime_at_entry', 'unknown')
            pnl = t.get('net_pnl', 0) or 0
            regime_trades[regime].append(pnl)
        
        results = []
        warnings = []
        
        for regime, pnls in regime_trades.items():
            if len(pnls) < 10:  # Need minimum data
                continue
            
            pnl_arr = np.array(pnls, dtype=float)
            
            # Historical stats
            historical_mean = float(np.mean(pnl_arr))
            historical_std = float(np.std(pnl_arr))
            
            # Rolling recent expectancy (last N trades)
            recent_window = min(self.rolling_window, len(pnl_arr))
            recent_pnls = pnl_arr[-recent_window:]
            recent_mean = float(np.mean(recent_pnls))
            
            # Z-score: how far recent performance is from historical
            if historical_std > 0:
                zscore = (recent_mean - historical_mean) / historical_std
            else:
                zscore = 0.0
            
            is_warning = zscore < -2.0
            is_deteriorating = zscore < -1.0
            
            regime_result = {
                'regime': regime,
                'historical_mean': round(historical_mean, 2),
                'historical_std': round(historical_std, 2),
                'recent_mean': round(recent_mean, 2),
                'zscore': round(zscore, 4),
                'n_trades': len(pnls),
                'n_recent': recent_window,
                'is_warning': is_warning,
                'is_deteriorating': is_deteriorating,
                'status': 'ALERT' if is_warning else ('WATCH' if is_deteriorating else 'OK'),
            }
            results.append(regime_result)
            
            if is_warning:
                warnings.append(
                    f"⚠ {regime}: z-score {zscore:.2f} — "
                    f"recent mean ${recent_mean:.0f} vs historical ${historical_mean:.0f}"
                )
        
        # Sort by z-score (worst first)
        results.sort(key=lambda x: x['zscore'])
        
        return {
            'regimes': results,
            'warnings': warnings,
            'overall_drift': len(warnings) > 0,
            'worst_zscore': round(min(r['zscore'] for r in results), 4) if results else 0,
        }


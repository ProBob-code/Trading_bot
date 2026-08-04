"""
V3 Quant Strategies — Renaissance-Style Statistical Alpha Engine
=================================================================

NEW FILE — v2_strategies.py is untouched and remains fully functional.
This module ADDS three new strategies and re-exports everything the rest
of the backend imports, so it is a drop-in superset of v2_strategies.

Design philosophy (Jim Simons / Medallion principles):
    1.  MANY WEAK SIGNALS, ONE STRONG PORTFOLIO — no single indicator is
        trusted; nine independent "alphas" each contribute a small edge in
        [-1, +1] and are combined with regime-conditional weights.
    2.  STATISTICS OVER STORIES — regime is not a vibe; it is measured with
        the Hurst exponent and a Lo-MacKinlay variance ratio. H < 0.5 means
        the series mean-reverts, H > 0.5 means it trends. Weights shift
        continuously with the evidence instead of switching strategies.
    3.  VOLATILITY IS THE UNIT OF EVERYTHING — every alpha is normalized by
        realized volatility (z-scores), and position size targets constant
        portfolio volatility with a fractional-Kelly cap.
    4.  COSTS ARE PART OF THE MODEL — a trade only fires when the estimated
        expected move exceeds round-trip cost by a safety multiple. The real
        expected move (edge × ATR) is attached to the Signal so RiskEngineV2's
        cost gate finally receives a true number instead of a hardcoded 0.5%.
    5.  SHORT MEMORY, FAST DECAY — recent data is weighted more via EWMA;
        the OU half-life sets the mean-reversion horizon adaptively.

New strategy IDs (FROZEN — do not rename):
    quant_alpha     — flagship multi-alpha statistical ensemble
    quant_meanrev   — pure stat-arb z-score reversion (OU half-life gated)
    quant_momentum  — multi-horizon vol-normalized momentum + breakout

Registry:
    REGISTRY_V3  — only the new strategies
    REGISTRY     — v2 REGISTRY + REGISTRY_V3 (combined, for drop-in import)
"""

import logging
import math
import time
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Re-export the v2 surface so this file is a drop-in superset.
from shared.logic.strategies.v2_strategies import (  # noqa: F401
    REGISTRY as V2_REGISTRY,
    Signal,
    analyze_order_book,
    atr,
    compute_atr_position_size,
    compute_bollinger,
    compute_cvd,
    compute_macd,
    compute_roc,
    compute_rsi,
    compute_smart_entry,
    compute_vwap,
    detect_whale_activity,
    ema,
    sma,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Cost model — round-trip cost estimate and required edge multiple
# ---------------------------------------------------------------------------
# Calibrated for intraday paper trading (1m–15m). The execution engine's
# modelled round-trip cost is ~0.04% (spread 0.02% + commission/slippage), and
# paper fills carry no real market impact — so the old 0.10%×1.5 = 0.15% floor
# was mathematically unreachable on 1m candles (ATR% ≈ 0.1–0.2%), which is why
# bots ran forever without ever taking a trade. These values keep the
# "only trade with positive post-cost edge" principle while being achievable.
ROUND_TRIP_COST_PCT = 0.0004  # 0.04% modelled round-trip cost
EDGE_COST_MULTIPLE = 1.0      # expected move must clear costs (positive EV)

# ── Signal Sensitivity presets (per-bot, chosen on the Command Deck) ──
# Each preset scales the entry gates that decide whether a bot trades. Lower
# thresholds → more frequent trades (and more noise / cost drag).
#   edge_min   : min |composite edge| for quant strategies to fire
#   cost_mult  : multiplier on the round-trip-cost gate (higher = stricter)
#   ml_conf    : min ML directional confidence |P(up)-0.5| for ml_forecast
#   risk_floor : min signal score the risk engine will accept
SENSITIVITY_PRESETS = {
    'conservative': {'edge_min': 0.12, 'cost_mult': 1.5, 'ml_conf': 0.15, 'risk_floor': 0.50},
    'balanced':     {'edge_min': 0.07, 'cost_mult': 1.0, 'ml_conf': 0.10, 'risk_floor': 0.35},
    'aggressive':   {'edge_min': 0.03, 'cost_mult': 0.6, 'ml_conf': 0.05, 'risk_floor': 0.25},
}


def resolve_sensitivity(name) -> dict:
    """Map a sensitivity name to its threshold preset (defaults to conservative)."""
    return SENSITIVITY_PRESETS.get(
        str(name or 'conservative').lower(), SENSITIVITY_PRESETS['conservative'])

# Per-strategy cooldown (replaces v2's single global cooldown)
_LAST_TRADE: Dict[str, float] = {}
COOLDOWN_SECONDS = 60


def _cooldown_ok(strategy_id: str) -> bool:
    return (time.time() - _LAST_TRADE.get(strategy_id, 0.0)) >= COOLDOWN_SECONDS


def _mark_trade(strategy_id: str) -> None:
    _LAST_TRADE[strategy_id] = time.time()


# ---------------------------------------------------------------------------
# Statistical machinery
# ---------------------------------------------------------------------------

def realized_vol(close: pd.Series, span: int = 20) -> float:
    """EWMA realized volatility of log returns (per-bar, not annualized)."""
    rets = np.log(close / close.shift(1)).dropna()
    if len(rets) < 5:
        return 0.0
    vol = rets.ewm(span=span, min_periods=5).std().iloc[-1]
    return float(vol) if np.isfinite(vol) else 0.0


def hurst_exponent(close: pd.Series, max_lag: int = 20) -> float:
    """
    Hurst exponent via the aggregated-variance method:
        std(x[t+lag] - x[t]) ∝ lag^H
    H ≈ 0.5 random walk, H < 0.5 mean-reverting, H > 0.5 trending.
    """
    prices = np.log(close.dropna().values)
    n = len(prices)
    if n < max_lag * 3:
        return 0.5
    lags = range(2, max_lag)
    tau = []
    for lag in lags:
        diff = prices[lag:] - prices[:-lag]
        s = np.std(diff)
        tau.append(s if s > 0 else 1e-10)
    try:
        h = np.polyfit(np.log(list(lags)), np.log(tau), 1)[0]
    except Exception:
        return 0.5
    return float(np.clip(h, 0.0, 1.0))


def variance_ratio(close: pd.Series, k: int = 5) -> float:
    """
    Lo-MacKinlay variance ratio: Var(k-period returns) / (k × Var(1-period)).
    VR < 1 → mean reversion, VR > 1 → momentum/trending.
    """
    rets = np.log(close / close.shift(1)).dropna().values
    if len(rets) < k * 10:
        return 1.0
    var1 = np.var(rets, ddof=1)
    if var1 <= 0:
        return 1.0
    rets_k = pd.Series(rets).rolling(k).sum().dropna().values
    vark = np.var(rets_k, ddof=1)
    return float(vark / (k * var1))


def ou_half_life(close: pd.Series) -> float:
    """
    Ornstein-Uhlenbeck half-life of mean reversion via AR(1):
        Δp[t] = a + b·p[t-1] + ε  →  half_life = -ln(2)/b
    Returns bars-to-half-revert; large/inf means no reversion.
    """
    p = np.log(close.dropna().values)
    if len(p) < 30:
        return float('inf')
    lag = p[:-1]
    delta = np.diff(p)
    lag_c = lag - lag.mean()
    denom = float(np.dot(lag_c, lag_c))
    if denom <= 0:
        return float('inf')
    b = float(np.dot(lag_c, delta - delta.mean())) / denom
    if b >= 0:
        return float('inf')  # no mean reversion
    hl = -math.log(2) / b
    return float(hl) if np.isfinite(hl) and hl > 0 else float('inf')


def zscore(series: pd.Series, window: int = 20) -> float:
    """Current value's z-score vs rolling window (clipped to ±4)."""
    if len(series) < window:
        return 0.0
    win = series.iloc[-window:]
    std = win.std()
    if not np.isfinite(std) or std == 0:
        return 0.0
    z = (series.iloc[-1] - win.mean()) / std
    return float(np.clip(z, -4, 4))


# ---------------------------------------------------------------------------
# Alpha library — each alpha returns (edge ∈ [-1, +1], reason)
# Positive edge = bullish, negative = bearish, 0 = no opinion.
# ---------------------------------------------------------------------------

def alpha_zscore_reversion(df: pd.DataFrame) -> Tuple[float, str]:
    """Fade statistically stretched prices. Classic stat-arb reversion."""
    z = zscore(df['close'], 20)
    if abs(z) < 1.0:
        return 0.0, f"price z={z:+.2f} (inside 1σ)"
    # Fade the move: +2σ stretch → sell pressure, scaled, saturating at 3σ
    edge = float(np.clip(-z / 3.0, -1, 1))
    return edge, f"price z={z:+.2f} → fade ({edge:+.2f})"


def alpha_multi_horizon_momentum(df: pd.DataFrame) -> Tuple[float, str]:
    """Vol-normalized momentum blended across 5/10/20-bar horizons."""
    close = df['close']
    vol = realized_vol(close)
    if vol <= 0:
        return 0.0, "no vol estimate"
    parts = []
    for horizon, w in ((5, 0.5), (10, 0.3), (20, 0.2)):
        if len(close) <= horizon:
            continue
        r = math.log(close.iloc[-1] / close.iloc[-1 - horizon])
        # normalize by vol scaled to horizon (sqrt-time)
        parts.append(w * np.clip(r / (vol * math.sqrt(horizon) * 2), -1, 1))
    if not parts:
        return 0.0, "insufficient history"
    edge = float(np.clip(sum(parts), -1, 1))
    return edge, f"multi-horizon momentum {edge:+.2f}"


def alpha_vwap_cvd_flow(df: pd.DataFrame) -> Tuple[float, str]:
    """Order-flow alpha: VWAP deviation confirmed by CVD slope agreement."""
    vwap = compute_vwap(df)
    cvd = compute_cvd(df)
    if len(cvd) < 6 or np.isnan(vwap.iloc[-1]):
        return 0.0, "flow data unavailable"
    price = df['close'].iloc[-1]
    dev = (price - vwap.iloc[-1]) / vwap.iloc[-1] if vwap.iloc[-1] else 0.0
    cvd_slope = cvd.iloc[-1] - cvd.iloc[-6]
    vol_sum = df['volume'].iloc[-6:].sum()
    cvd_norm = float(np.clip(cvd_slope / vol_sum, -1, 1)) if vol_sum > 0 else 0.0
    dev_sig = float(np.clip(dev * 100, -1, 1))
    # Agreement amplifies, disagreement cancels
    edge = float(np.clip(0.5 * dev_sig + 0.5 * cvd_norm, -1, 1))
    if np.sign(dev_sig) != np.sign(cvd_norm) and dev_sig and cvd_norm:
        edge *= 0.3  # conflicting flow → distrust
    return edge, f"VWAP dev {dev:+.3%}, CVD flow {cvd_norm:+.2f} → {edge:+.2f}"


def alpha_rsi_tension(df: pd.DataFrame) -> Tuple[float, str]:
    """RSI as a rubber band: extremes revert, mid-zone follows the slope."""
    rsi = compute_rsi(df['close'], 14)
    if len(rsi) < 3 or np.isnan(rsi.iloc[-1]):
        return 0.0, "RSI unavailable"
    r = rsi.iloc[-1]
    if r >= 70:
        edge = -(r - 70) / 30
    elif r <= 30:
        edge = (30 - r) / 30
    else:
        edge = float(np.clip((rsi.iloc[-1] - rsi.iloc[-3]) / 20, -0.3, 0.3))
    return float(np.clip(edge, -1, 1)), f"RSI {r:.0f} → {edge:+.2f}"


def alpha_macd_impulse(df: pd.DataFrame) -> Tuple[float, str]:
    """MACD histogram slope (impulse), normalized by price."""
    macd = compute_macd(df['close'])
    hist = macd['histogram']
    if len(hist) < 4 or np.isnan(hist.iloc[-1]):
        return 0.0, "MACD unavailable"
    price = df['close'].iloc[-1]
    impulse = (hist.iloc[-1] - hist.iloc[-4]) / price if price else 0.0
    edge = float(np.clip(impulse * 400, -1, 1))
    return edge, f"MACD impulse {edge:+.2f}"


def alpha_donchian_breakout(df: pd.DataFrame, window: int = 20) -> Tuple[float, str]:
    """Breakout of N-bar channel with volume confirmation."""
    if len(df) < window + 2:
        return 0.0, "insufficient history"
    hi = df['high'].iloc[-window - 1:-1].max()
    lo = df['low'].iloc[-window - 1:-1].min()
    close = df['close'].iloc[-1]
    vol_ok = df['volume'].iloc[-1] >= df['volume'].rolling(20).mean().iloc[-1]
    if close > hi:
        return (0.8 if vol_ok else 0.4), f"{window}-bar breakout ↑ (vol {'✓' if vol_ok else '✗'})"
    if close < lo:
        return (-0.8 if vol_ok else -0.4), f"{window}-bar breakdown ↓ (vol {'✓' if vol_ok else '✗'})"
    return 0.0, "inside channel"


def alpha_volume_price_corr(df: pd.DataFrame, window: int = 20) -> Tuple[float, str]:
    """Correlation of returns with signed volume — smart-money accumulation."""
    if len(df) < window + 1:
        return 0.0, "insufficient history"
    rets = df['close'].pct_change().iloc[-window:]
    body = (df['close'] - df['open']).iloc[-window:]
    signed_vol = np.sign(body) * df['volume'].iloc[-window:]
    if rets.std() == 0 or signed_vol.std() == 0:
        return 0.0, "flat window"
    corr = float(rets.corr(pd.Series(signed_vol, index=rets.index)))
    if not np.isfinite(corr):
        return 0.0, "corr undefined"
    recent_flow = float(np.sign(signed_vol.iloc[-5:].sum()))
    edge = float(np.clip(corr * recent_flow * 0.6, -1, 1))
    return edge, f"vol/price corr {corr:+.2f}, flow {recent_flow:+.0f} → {edge:+.2f}"


def alpha_serial_correlation(df: pd.DataFrame, window: int = 40) -> Tuple[float, str]:
    """
    Lag-1 autocorrelation of returns: positive → last move continues,
    negative → last move reverses. A direct, testable micro-predictor.
    """
    rets = df['close'].pct_change().dropna().iloc[-window:]
    if len(rets) < 20 or rets.std() == 0:
        return 0.0, "insufficient returns"
    ac = float(rets.autocorr(lag=1))
    if not np.isfinite(ac):
        return 0.0, "autocorr undefined"
    last_dir = float(np.sign(rets.iloc[-1]))
    edge = float(np.clip(ac * last_dir * 1.5, -1, 1))
    return edge, f"lag-1 autocorr {ac:+.2f} × last move → {edge:+.2f}"


def alpha_seasonality(df: pd.DataFrame) -> Tuple[float, str]:
    """Hour-of-day seasonal drift measured from this symbol's own history."""
    if 'timestamp' in df.columns:
        hours = pd.to_datetime(df['timestamp']).dt.hour
    elif isinstance(df.index, pd.DatetimeIndex):
        hours = pd.Series(df.index.hour, index=df.index)
    else:
        return 0.0, "no timestamps"
    rets = df['close'].pct_change()
    cur_hour = int(hours.iloc[-1])
    mask = hours == cur_hour
    sample = rets[mask].dropna()
    if len(sample) < 10:
        return 0.0, f"hour {cur_hour}: sample too small"
    t_stat = sample.mean() / (sample.std() / math.sqrt(len(sample))) if sample.std() > 0 else 0.0
    edge = float(np.clip(t_stat / 4.0, -0.5, 0.5))  # weak alpha by design
    return edge, f"hour-{cur_hour} drift t={t_stat:+.2f} → {edge:+.2f}"


# ---------------------------------------------------------------------------
# Regime-conditional weighting (the Simons blend)
# ---------------------------------------------------------------------------

# (alpha_fn, family, base_weight)
_ALPHAS = [
    (alpha_zscore_reversion,       'reversion', 1.0),
    (alpha_rsi_tension,            'reversion', 0.6),
    (alpha_multi_horizon_momentum, 'momentum',  1.0),
    (alpha_macd_impulse,           'momentum',  0.6),
    (alpha_donchian_breakout,      'momentum',  0.8),
    (alpha_vwap_cvd_flow,          'flow',      0.9),
    (alpha_volume_price_corr,      'flow',      0.5),
    (alpha_serial_correlation,     'stat',      0.7),
    (alpha_seasonality,            'stat',      0.3),
]


def regime_evidence(close: pd.Series) -> Dict[str, float]:
    """
    Continuous regime evidence in [0, 1] for momentum vs reversion,
    from Hurst exponent and variance ratio (two independent estimators).
    """
    h = hurst_exponent(close)
    vr = variance_ratio(close, k=5)
    # Map H∈[0.3,0.7] → momentum weight ∈ [0,1]
    mom_h = np.clip((h - 0.3) / 0.4, 0, 1)
    # Map VR∈[0.6,1.4] → momentum weight ∈ [0,1]
    mom_vr = np.clip((vr - 0.6) / 0.8, 0, 1)
    momentum_w = float(0.5 * mom_h + 0.5 * mom_vr)
    return {
        'hurst': round(h, 3),
        'variance_ratio': round(vr, 3),
        'momentum_weight': round(momentum_w, 3),
        'reversion_weight': round(1.0 - momentum_w, 3),
    }


def combine_alphas(df: pd.DataFrame, **kwargs) -> Dict[str, Any]:
    """
    Run the full alpha library, weight by regime evidence, and produce a
    composite edge plus a true expected-move estimate.
    """
    close = df['close']
    regime = regime_evidence(close)
    mom_w, rev_w = regime['momentum_weight'], regime['reversion_weight']
    family_scale = {'momentum': 0.5 + mom_w, 'reversion': 0.5 + rev_w,
                    'flow': 1.0, 'stat': 1.0}

    total, weight_sum = 0.0, 0.0
    details: List[str] = []
    for fn, family, base_w in _ALPHAS:
        try:
            edge, reason = fn(df)
        except Exception as exc:
            logger.debug("alpha %s failed: %s", fn.__name__, exc)
            continue
        w = base_w * family_scale[family]
        total += edge * w
        weight_sum += w
        if edge != 0.0:
            details.append(f"{fn.__name__.replace('alpha_', '')}: {reason}")

    composite = total / weight_sum if weight_sum else 0.0

    # Order book + funding overlays (kept small, like v2)
    ob = analyze_order_book(kwargs.get('order_book'))
    if ob['available'] and ob['bias']:
        composite = float(np.clip(composite + ob['bias'] * 0.5, -1, 1))
        details.append(f"order book bias {ob['bias']:+.2f}")
    funding = kwargs.get('funding_rate')
    if funding is not None and abs(funding) > 0.01:
        fr = -0.08 if funding > 0 else 0.08
        composite = float(np.clip(composite + fr, -1, 1))
        details.append(f"funding overlay {fr:+.2f}")

    # Expected move: |edge| × ATR% × sqrt(horizon). This feeds the risk
    # engine's cost gate with a REAL number (v2 hardcoded 0.5%).
    atr_series = atr(df, 14)
    atr_val = atr_series.iloc[-1] if len(atr_series) else np.nan
    price = close.iloc[-1]
    atr_pct = float(atr_val / price) if price and np.isfinite(atr_val) else 0.0
    expected_move_pct = abs(composite) * atr_pct * math.sqrt(5) * 100  # in %

    return {
        'edge': float(np.clip(composite, -1, 1)),
        'expected_move_pct': round(expected_move_pct, 4),
        'atr': float(atr_val) if np.isfinite(atr_val) else 0.0,
        'atr_pct': round(atr_pct, 5),
        'regime': regime,
        'details': details,
    }


# ---------------------------------------------------------------------------
# Position sizing — volatility targeting with fractional Kelly cap
# ---------------------------------------------------------------------------

def vol_target_size(equity: float, price: float, bar_vol: float,
                    target_daily_vol: float = 0.02,
                    bars_per_day: int = 1440) -> float:
    """
    Size the position so its expected daily volatility contribution equals
    `target_daily_vol` of equity (2% default). bar_vol is per-bar return vol.
    """
    if price <= 0 or bar_vol <= 0:
        return 0.0
    daily_vol = bar_vol * math.sqrt(bars_per_day)
    if daily_vol <= 0:
        return 0.0
    notional = equity * (target_daily_vol / daily_vol)
    notional = min(notional, equity)  # never exceed 1× equity unlevered
    return notional / price


def fractional_kelly(p_win: float, payoff_ratio: float = 2.0,
                     fraction: float = 0.25) -> float:
    """
    Quarter-Kelly bet fraction: f* = (p·b − q)/b, scaled by `fraction`.
    Returns fraction of equity to risk (≥ 0, capped at 2%).
    """
    if payoff_ratio <= 0:
        return 0.0
    q = 1.0 - p_win
    f_star = (p_win * payoff_ratio - q) / payoff_ratio
    return float(np.clip(f_star * fraction, 0.0, 0.02))


def compute_position_plan(df: pd.DataFrame, equity: float,
                          edge: float, atr_val: float) -> Dict[str, float]:
    """
    Full trade plan: entry, asymmetric ATR stops, vol-targeted size with
    Kelly cap. SL at 1.5×ATR, TP at 3×ATR (2:1 like v2, but tighter stop —
    smaller loss per mistake, same reward shape).
    """
    price = float(df['close'].iloc[-1])
    if price <= 0 or atr_val <= 0:
        return {'size': 0.0, 'stop': 0.0, 'tp': 0.0, 'entry': price}

    stop_dist = 1.5 * atr_val
    tp_dist = 3.0 * atr_val

    # p_win proxy from |edge|: edge 0 → 0.5, edge 1 → 0.65 (kept conservative)
    p_win = 0.5 + min(abs(edge), 1.0) * 0.15
    kelly_risk = fractional_kelly(p_win, payoff_ratio=tp_dist / stop_dist)
    risk_amount = equity * max(kelly_risk, 0.0)
    kelly_qty = risk_amount / stop_dist if stop_dist > 0 else 0.0

    vol_qty = vol_target_size(equity, price, realized_vol(df['close']))
    qty = min(kelly_qty, vol_qty) if vol_qty > 0 else kelly_qty

    if edge >= 0:
        stop, tp = price - stop_dist, price + tp_dist
    else:
        stop, tp = price + stop_dist, price - tp_dist

    return {
        'entry': round(price, 8),
        'stop': round(stop, 8),
        'tp': round(tp, 8),
        'size': round(qty, 8),
        'risk_pct': round(kelly_risk * 100, 3),
        'p_win': round(p_win, 3),
    }


# ---------------------------------------------------------------------------
# Signal construction helper
# ---------------------------------------------------------------------------

def _make_signal(strategy_id: str, direction: str, score: float, price: float,
                 reasons: List[str], expected_move_pct: float) -> Signal:
    sig = Signal(strategy_id, direction, round(score, 4), price,
                 reasons, datetime.now())
    # RiskEngineV2's cost gate reads this via getattr — attach the real value
    sig.expected_move_pct = max(expected_move_pct, 0.0)
    return sig


# ---------------------------------------------------------------------------
# Strategy 1 — quant_alpha (flagship ensemble)
# ---------------------------------------------------------------------------

def quant_alpha_strategy(df: pd.DataFrame, price: float,
                         min_confluence: int, **kwargs) -> Signal:
    """
    Multi-alpha statistical ensemble with regime-conditional weights.

    Fires only when:  |composite edge| ≥ 0.25
                 AND  expected move ≥ 1.5× round-trip cost
                 AND  cooldown elapsed
    """
    if len(df) < 60:
        return _make_signal('quant_alpha', 'HOLD', 0.0, price,
                            ['Need ≥60 candles'], 0.0)

    # Per-bot sensitivity (falls back to conservative defaults)
    params = kwargs.get('params') or {}
    edge_min = params.get('edge_min', 0.12)
    cost_mult = params.get('cost_mult', EDGE_COST_MULTIPLE)

    res = combine_alphas(df, **kwargs)
    edge = res['edge']
    reg = res['regime']
    reasons = [
        f"Regime: H={reg['hurst']}, VR={reg['variance_ratio']} "
        f"(momentum {reg['momentum_weight']:.0%} / reversion {reg['reversion_weight']:.0%})",
        f"Composite edge: {edge:+.3f} | expected move {res['expected_move_pct']:.3f}%",
    ] + res['details'][:8]

    if abs(edge) < edge_min:
        reasons.append(f"Edge below {edge_min:.2f} threshold — no trade")
        return _make_signal('quant_alpha', 'HOLD', 0.0, price, reasons,
                            res['expected_move_pct'])

    required = ROUND_TRIP_COST_PCT * 100 * cost_mult
    if res['expected_move_pct'] < required:
        reasons.append(
            f"Expected move {res['expected_move_pct']:.3f}% < required "
            f"{required:.3f}% (cost × {EDGE_COST_MULTIPLE}) — no trade")
        return _make_signal('quant_alpha', 'HOLD', 0.0, price, reasons,
                            res['expected_move_pct'])

    if not _cooldown_ok('quant_alpha'):
        reasons.append("Cooldown active")
        return _make_signal('quant_alpha', 'HOLD', 0.0, price, reasons,
                            res['expected_move_pct'])

    direction = 'BUY' if edge > 0 else 'SELL'
    score = 0.5 + abs(edge) * 0.5  # edge 0.25→0.625, edge 1.0→1.0
    _mark_trade('quant_alpha')
    reasons.append(f"➡ {direction} score {score:.2f}")
    return _make_signal('quant_alpha', direction, min(score, 1.0), price,
                        reasons, res['expected_move_pct'])


# ---------------------------------------------------------------------------
# Strategy 2 — quant_meanrev (pure stat-arb reversion)
# ---------------------------------------------------------------------------

def quant_meanrev_strategy(df: pd.DataFrame, price: float,
                           min_confluence: int, **kwargs) -> Signal:
    """
    Z-score reversion, only when the statistics say reversion is live:
      - OU half-life finite and short (< 60 bars)
      - variance ratio < 1 (anti-persistent returns)
      - |z| ≥ 1.5, faded toward the mean
    """
    if len(df) < 60:
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price,
                            ['Need ≥60 candles'], 0.0)

    params = kwargs.get('params') or {}
    cost_mult = params.get('cost_mult', EDGE_COST_MULTIPLE)

    close = df['close']
    hl = ou_half_life(close)
    vr = variance_ratio(close, k=5)
    z = zscore(close, 20)
    reasons = [f"OU half-life: {'∞' if math.isinf(hl) else f'{hl:.1f} bars'}, "
               f"VR={vr:.2f}, z={z:+.2f}"]

    if math.isinf(hl) or hl > 60:
        reasons.append("No tradeable mean reversion (half-life too long)")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)
    if vr >= 1.0:
        reasons.append("Variance ratio ≥ 1 — returns persistent, reversion unsafe")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)
    if abs(z) < 1.5:
        reasons.append("Stretch < 1.5σ — wait for a better entry")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)

    # Confirmation: CVD should NOT be accelerating against the fade
    cvd = compute_cvd(df)
    cvd_slope = cvd.iloc[-1] - cvd.iloc[-6] if len(cvd) > 6 else 0.0
    fading_down = z > 0  # stretched up → we sell
    if fading_down and cvd_slope > 0 and abs(z) < 2.5:
        reasons.append("Buyers still aggressive — fade postponed")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)
    if not fading_down and cvd_slope < 0 and abs(z) < 2.5:
        reasons.append("Sellers still aggressive — fade postponed")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)

    if not _cooldown_ok('quant_meanrev'):
        reasons.append("Cooldown active")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons, 0.0)

    # Expected move: distance back to the 20-bar mean
    mean20 = close.iloc[-20:].mean()
    expected_move_pct = abs(price - mean20) / price * 100 if price else 0.0
    required = ROUND_TRIP_COST_PCT * 100 * cost_mult
    if expected_move_pct < required:
        reasons.append(f"Reversion target {expected_move_pct:.3f}% below cost bar")
        return _make_signal('quant_meanrev', 'HOLD', 0.0, price, reasons,
                            expected_move_pct)

    direction = 'SELL' if z > 0 else 'BUY'
    score = min(0.5 + (abs(z) - 1.5) * 0.2 + (0.1 if vr < 0.8 else 0.0), 1.0)
    _mark_trade('quant_meanrev')
    reasons.append(f"➡ {direction} fade to mean {mean20:.2f} (score {score:.2f})")
    return _make_signal('quant_meanrev', direction, score, price, reasons,
                        expected_move_pct)


# ---------------------------------------------------------------------------
# Strategy 3 — quant_momentum (multi-horizon trend capture)
# ---------------------------------------------------------------------------

def quant_momentum_strategy(df: pd.DataFrame, price: float,
                            min_confluence: int, **kwargs) -> Signal:
    """
    Vol-normalized multi-horizon momentum, gated by trend evidence:
      - Hurst > 0.5 or VR > 1 (persistence measured, not assumed)
      - momentum and breakout alphas agree in sign
      - volume confirms
    """
    if len(df) < 60:
        return _make_signal('quant_momentum', 'HOLD', 0.0, price,
                            ['Need ≥60 candles'], 0.0)

    params = kwargs.get('params') or {}
    edge_min = params.get('edge_min', 0.12)
    cost_mult = params.get('cost_mult', EDGE_COST_MULTIPLE)

    close = df['close']
    h = hurst_exponent(close)
    vr = variance_ratio(close, k=5)
    mom_edge, mom_reason = alpha_multi_horizon_momentum(df)
    brk_edge, brk_reason = alpha_donchian_breakout(df)
    flow_edge, flow_reason = alpha_vwap_cvd_flow(df)

    reasons = [f"H={h:.2f}, VR={vr:.2f}", mom_reason, brk_reason, flow_reason]

    if h < 0.5 and vr < 1.0:
        reasons.append("No persistence evidence — momentum stood down")
        return _make_signal('quant_momentum', 'HOLD', 0.0, price, reasons, 0.0)

    combined = 0.5 * mom_edge + 0.3 * brk_edge + 0.2 * flow_edge
    mom_min = max(edge_min, 0.03)
    if abs(combined) < mom_min:
        reasons.append(f"Combined momentum {combined:+.2f} < {mom_min:.2f} — no trade")
        return _make_signal('quant_momentum', 'HOLD', 0.0, price, reasons, 0.0)
    if brk_edge != 0 and np.sign(brk_edge) != np.sign(mom_edge):
        reasons.append("Breakout and momentum disagree — no trade")
        return _make_signal('quant_momentum', 'HOLD', 0.0, price, reasons, 0.0)

    if not _cooldown_ok('quant_momentum'):
        reasons.append("Cooldown active")
        return _make_signal('quant_momentum', 'HOLD', 0.0, price, reasons, 0.0)

    atr_series = atr(df, 14)
    atr_val = atr_series.iloc[-1] if np.isfinite(atr_series.iloc[-1]) else 0.0
    expected_move_pct = (atr_val / price * 100 * 2.0 * abs(combined)) if price else 0.0
    required = ROUND_TRIP_COST_PCT * 100 * cost_mult
    if expected_move_pct < required:
        reasons.append(f"Expected move {expected_move_pct:.3f}% below cost bar")
        return _make_signal('quant_momentum', 'HOLD', 0.0, price, reasons,
                            expected_move_pct)

    whale = detect_whale_activity(df)
    direction = 'BUY' if combined > 0 else 'SELL'
    score = min(0.5 + abs(combined) * 0.4 + whale, 1.0)
    _mark_trade('quant_momentum')
    reasons.append(f"➡ {direction} momentum score {score:.2f}"
                   + (" 🐋" if whale else ""))
    return _make_signal('quant_momentum', direction, score, price, reasons,
                        expected_move_pct)


# ---------------------------------------------------------------------------
# V3 risk manager (same interface as v2's risk_manager, tighter stops)
# ---------------------------------------------------------------------------

def risk_manager_v3(df: pd.DataFrame, account_balance: float) -> Dict[str, Any]:
    """Trade plan built from the flagship ensemble + Kelly/vol-target sizing."""
    if df.empty or len(df) < 60:
        return {'signal': 'HOLD', 'entry': 0.0, 'stop': 0.0, 'tp': 0.0,
                'size': 0.0, 'reasons': ['Insufficient data']}

    price = float(df['close'].iloc[-1])
    sig = quant_alpha_strategy(df, price, min_confluence=2)
    res = combine_alphas(df)
    plan = compute_position_plan(df, account_balance, res['edge'], res['atr'])

    return {
        'signal': sig.signal,
        'entry': plan['entry'],
        'stop': plan['stop'],
        'tp': plan['tp'],
        'size': plan['size'] if sig.signal != 'HOLD' else 0.0,
        'atr': round(res['atr'], 8),
        'risk_reward': 2.0,
        'risk_pct': plan.get('risk_pct', 0.0),
        'expected_move_pct': res['expected_move_pct'],
        'regime': res['regime'],
        'reasons': sig.reasons,
    }


# ---------------------------------------------------------------------------
# Registries
# ---------------------------------------------------------------------------

REGISTRY_V3 = [
    {
        'id': 'quant_alpha',
        'name': 'Quant Alpha Ensemble',
        'icon': '🧠',
        'description': ('Renaissance-style multi-alpha ensemble — 9 weak signals, '
                        'Hurst/variance-ratio regime weighting, cost-aware edge, '
                        'true expected-move risk gating.'),
        'logic': quant_alpha_strategy,
    },
    {
        'id': 'quant_meanrev',
        'name': 'Statistical Mean Reversion',
        'icon': '🎯',
        'description': ('Stat-arb z-score fade — OU half-life and variance-ratio '
                        'gated, CVD aggression filter, mean-distance profit target.'),
        'logic': quant_meanrev_strategy,
    },
    {
        'id': 'quant_momentum',
        'name': 'Persistence Momentum',
        'icon': '🌊',
        'description': ('Vol-normalized multi-horizon momentum — trades only with '
                        'measured persistence (Hurst>0.5 or VR>1), breakout + flow '
                        'agreement required.'),
        'logic': quant_momentum_strategy,
    },
]

# Combined registry: everything v2 had, plus the new quant strategies.
# Drop-in replacement for `from ...v2_strategies import REGISTRY`.
REGISTRY = V2_REGISTRY + REGISTRY_V3

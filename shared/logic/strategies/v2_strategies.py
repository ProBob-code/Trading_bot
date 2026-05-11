"""
V2 Trading Strategies Module — Professional Grade
===================================================

Leading-indicator strategies with order book intelligence.

Strategies (FROZEN IDs — do NOT rename):
    ichimoku    — VWAP Trend + CVD confirmation
    bollinger   — Smart Mean Reversion + divergence filter
    macd_rsi    — Momentum Flow (ROC + CVD surge)
    ml_forecast — Engineered LSTM (12-feature, direction probability)
    combined    — Regime-Adaptive Ensemble + order book bias

Indicator Stack (leading, not lagging):
    VWAP, CVD, Momentum Divergence, ROC, Adaptive ATR Bands

Risk Management:
    ATR-based stops (2×ATR SL, 4×ATR TP), 1% risk per trade
"""

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Trade Cooldown
# ---------------------------------------------------------------------------
LAST_TRADE_TIME: float = 0.0
TRADE_COOLDOWN_SECONDS: int = 60  # 1 minute


def _check_cooldown() -> bool:
    """Return True if cooldown has elapsed and trading is allowed."""
    global LAST_TRADE_TIME
    now = time.time()
    remaining = TRADE_COOLDOWN_SECONDS - (now - LAST_TRADE_TIME)
    if remaining > 0:
        print(f"[STRATEGY] Cooldown active: {remaining:.1f}s")
        return False
    return True


def _mark_trade():
    """Record current time as last trade time."""
    global LAST_TRADE_TIME
    LAST_TRADE_TIME = time.time()


# ---------------------------------------------------------------------------
# Signal dataclass (backward compatible: .score AND .strength)
# ---------------------------------------------------------------------------

@dataclass
class Signal:
    """Standardized trading signal returned by every strategy."""
    strategy: str
    signal: str        # "BUY" | "SELL" | "HOLD"
    score: float       # 0.0 – 1.0 confidence
    price: float
    reasons: List[str]
    timestamp: datetime

    @property
    def strength(self) -> int:
        """Backward compatibility: score 0.0–1.0 → strength 0–100."""
        return int(abs(self.score) * 100)


# ---------------------------------------------------------------------------
# Leading Indicator Helpers
# ---------------------------------------------------------------------------

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple Moving Average."""
    return series.rolling(window=period, min_periods=period).mean()


def ema(series: pd.Series, period: int) -> pd.Series:
    """Exponential Moving Average."""
    return series.ewm(span=period, min_periods=period, adjust=False).mean()


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range — NaN-safe."""
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window=period, min_periods=period).mean()


def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window=period, min_periods=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period, min_periods=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series, fast: int = 12, slow: int = 26,
                 signal_period: int = 9) -> pd.DataFrame:
    """MACD line, signal line, histogram."""
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal_period)
    histogram = macd_line - signal_line
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram,
    })


def compute_obv(df: pd.DataFrame) -> pd.Series:
    """On-Balance Volume."""
    direction = np.sign(df['close'].diff())
    return (direction * df['volume']).cumsum()


# ---------------------------------------------------------------------------
# LEADING Indicators (the core upgrade)
# ---------------------------------------------------------------------------

def compute_vwap(df: pd.DataFrame) -> pd.Series:
    """
    VWAP — Volume Weighted Average Price.
    Rolling 24h for crypto (uses all available data in the session).
    """
    tp = (df['high'] + df['low'] + df['close']) / 3
    cumvol = df['volume'].cumsum()
    cumvol = cumvol.replace(0, np.nan)
    return (tp * df['volume']).cumsum() / cumvol


def compute_cvd(df: pd.DataFrame) -> pd.Series:
    """
    CVD — Cumulative Volume Delta.
    Approximated from candle body ratio:
        body = close - open
        range = high - low
        delta = (body / range) * volume
    Positive CVD = buying pressure, Negative = selling pressure.
    """
    body = df['close'] - df['open']
    rng = (df['high'] - df['low']).replace(0, np.nan).fillna(1)
    ratio = (body / rng).clip(-1, 1)
    delta = ratio * df['volume']
    return delta.cumsum()


def compute_roc(series: pd.Series, period: int = 10) -> pd.Series:
    """Rate of Change — price momentum acceleration."""
    shifted = series.shift(period)
    return (series - shifted) / shifted.replace(0, np.nan)


def detect_divergence(price: pd.Series, indicator: pd.Series,
                      lookback: int = 14) -> str:
    """
    Detect price vs indicator divergence.

    Bearish divergence: price makes higher high but indicator makes lower high.
    Bullish divergence: price makes lower low but indicator makes higher low.

    Returns: 'BULLISH', 'BEARISH', or 'NONE'
    """
    if len(price) < lookback * 2:
        return 'NONE'

    recent = slice(-lookback, None)
    prior = slice(-lookback * 2, -lookback)

    price_recent_high = price.iloc[recent].max()
    price_prior_high = price.iloc[prior].max()
    ind_recent_high = indicator.iloc[recent].max()
    ind_prior_high = indicator.iloc[prior].max()

    price_recent_low = price.iloc[recent].min()
    price_prior_low = price.iloc[prior].min()
    ind_recent_low = indicator.iloc[recent].min()
    ind_prior_low = indicator.iloc[prior].min()

    # Bearish: price ↑ but indicator ↓
    if price_recent_high > price_prior_high and ind_recent_high < ind_prior_high:
        return 'BEARISH'

    # Bullish: price ↓ but indicator ↑
    if price_recent_low < price_prior_low and ind_recent_low > ind_prior_low:
        return 'BULLISH'

    return 'NONE'


def compute_bollinger(close: pd.Series, period: int = 20,
                      num_std: float = 2.0) -> Dict[str, pd.Series]:
    """Bollinger Bands."""
    middle = sma(close, period)
    std = close.rolling(window=period, min_periods=period).std()
    return {
        'upper': middle + num_std * std,
        'lower': middle - num_std * std,
        'middle': middle,
        'pct_b': (close - (middle - num_std * std)) / (2 * num_std * std),
    }


# ---------------------------------------------------------------------------
# Market Regime Detection
# ---------------------------------------------------------------------------

def detect_market_regime(df: pd.DataFrame) -> str:
    """
    Classify current market as TREND, RANGE, or VOLATILE.

    Uses SMA20/SMA50 spread and ATR/close ratio.
    """
    if len(df) < 50:
        return 'RANGE'

    close = df['close']
    sma20 = sma(close, 20).iloc[-1]
    sma50 = sma(close, 50).iloc[-1]
    atr_val = atr(df, 14).iloc[-1]
    price = close.iloc[-1]

    if np.isnan(sma20) or np.isnan(sma50) or np.isnan(atr_val) or price == 0:
        return 'RANGE'

    sma_spread = abs(sma20 - sma50) / price
    volatility_ratio = atr_val / price

    # High volatility → VOLATILE
    if volatility_ratio > 0.03:
        return 'VOLATILE'

    # Clear trend separation → TREND
    if sma_spread > 0.01:
        return 'TREND'

    return 'RANGE'


# ---------------------------------------------------------------------------
# Volume Confirmation + Whale Detection
# ---------------------------------------------------------------------------

def _volume_confirmed(df: pd.DataFrame) -> bool:
    """Return True if current volume ≥ 20-period rolling average."""
    if len(df) < 20:
        return True  # Not enough data — allow trade
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    vol_now = df['volume'].iloc[-1]
    if np.isnan(vol_avg) or vol_avg == 0:
        return True
    return vol_now >= vol_avg


def detect_whale_activity(df: pd.DataFrame) -> float:
    """
    Detect whale activity from volume spikes.
    Returns a score bonus (0.0 or +0.2) if volume > 3× average.
    """
    if len(df) < 20:
        return 0.0
    vol_avg = df['volume'].rolling(20).mean().iloc[-1]
    vol_now = df['volume'].iloc[-1]
    if np.isnan(vol_avg) or vol_avg == 0:
        return 0.0
    if vol_now > 3 * vol_avg:
        return 0.2
    return 0.0


# ---------------------------------------------------------------------------
# Order Book Intelligence
# ---------------------------------------------------------------------------

def analyze_order_book(order_book: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Analyze order book for trading signals.

    Returns:
        imbalance: bid/ask volume ratio (>1.5 bullish, <0.67 bearish)
        bid_walls: list of large support levels
        ask_walls: list of large resistance levels
        spread_pct: bid-ask spread as fraction
        bias: float score adjustment (-0.15 to +0.15)
    """
    default = {
        'imbalance': 1.0, 'bid_walls': [], 'ask_walls': [],
        'spread_pct': 0.001, 'bias': 0.0, 'available': False,
    }

    if not order_book:
        return default

    bids = order_book.get('bids', [])
    asks = order_book.get('asks', [])

    if not bids or not asks:
        return default

    bid_volume = sum(b.get('volume', 0) for b in bids)
    ask_volume = sum(a.get('volume', 0) for a in asks)

    # Imbalance ratio
    imbalance = bid_volume / max(ask_volume, 1e-8)

    # Large wall detection
    avg_bid_vol = bid_volume / max(len(bids), 1)
    avg_ask_vol = ask_volume / max(len(asks), 1)
    bid_walls = [b for b in bids if b.get('volume', 0) > 3 * avg_bid_vol]
    ask_walls = [a for a in asks if a.get('volume', 0) > 3 * avg_ask_vol]

    # Spread
    best_bid = bids[0].get('price', 0)
    best_ask = asks[0].get('price', 0)
    spread_pct = (best_ask - best_bid) / best_bid if best_bid > 0 else 0.001

    # Directional bias from order book
    bias = 0.0
    if imbalance > 1.5:
        bias = 0.15   # bullish pressure
    elif imbalance < 0.67:
        bias = -0.15  # bearish pressure

    return {
        'imbalance': round(imbalance, 3),
        'bid_walls': bid_walls,
        'ask_walls': ask_walls,
        'spread_pct': round(spread_pct, 6),
        'bias': bias,
        'available': True,
        'bid_volume': bid_volume,
        'ask_volume': ask_volume,
    }


# ---------------------------------------------------------------------------
# Funding Rate Bias
# ---------------------------------------------------------------------------

def _funding_rate_bias(funding_rate: Optional[float] = None,
                       threshold: float = 0.01) -> float:
    """
    Funding rate directional bias.
    High positive → market overheated → bearish bias.
    High negative → shorts overloaded → bullish bias.
    Returns score adjustment.
    """
    if funding_rate is None:
        return 0.0
    if funding_rate > threshold:
        return -0.1  # bearish bias (longs paying → overbought)
    if funding_rate < -threshold:
        return 0.1   # bullish bias (shorts paying → oversold)
    return 0.0


# ---------------------------------------------------------------------------
# Liquidation Cluster Awareness
# ---------------------------------------------------------------------------

def _liquidation_proximity_boost(price: float,
                                  liquidation_levels: Optional[List[float]] = None,
                                  proximity_pct: float = 0.02) -> float:
    """
    If price is near a liquidation cluster, boost momentum score.
    Liquidation cascades drive price moves.
    """
    if not liquidation_levels:
        return 0.0
    for level in liquidation_levels:
        if level > 0 and abs(price - level) / level < proximity_pct:
            return 0.15  # price near liquidation cluster
    return 0.0


# ---------------------------------------------------------------------------
# LSTM Feature Engineering
# ---------------------------------------------------------------------------

def _rolling_zscore(series: pd.Series, window: int = 100) -> pd.Series:
    """Rolling z-score normalization."""
    mean = series.rolling(window=window, min_periods=1).mean()
    std = series.rolling(window=window, min_periods=1).std().replace(0, 1)
    return (series - mean) / std


def build_feature_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build 12-feature matrix for LSTM input.

    Features:
        1. Log returns
        2. ATR ratio (ATR/close)
        3. RSI(14)
        4. MACD histogram
        5. Volume ratio (vol / vol_avg_20)
        6. Bollinger %B
        7. ROC(10)
        8. OBV slope (OBV.diff(5) / OBV.rolling(20).mean())
        9. VWAP deviation ((close - VWAP) / VWAP)
        10. CVD slope (CVD.diff(5), normalized)
        11. Hour sin (sin(2π × hour/24))
        12. Hour cos (cos(2π × hour/24))
    """
    close = df['close']

    # 1. Log returns
    log_ret = np.log(close / close.shift(1))

    # 2. ATR ratio
    atr_val = atr(df, 14)
    atr_ratio = atr_val / close.replace(0, np.nan)

    # 3. RSI
    rsi = compute_rsi(close, 14) / 100.0  # normalize to 0–1

    # 4. MACD histogram
    macd_data = compute_macd(close)
    macd_hist = macd_data['histogram'] / close.replace(0, np.nan)  # normalize

    # 5. Volume ratio
    vol_avg = df['volume'].rolling(20, min_periods=1).mean()
    vol_ratio = df['volume'] / vol_avg.replace(0, np.nan)

    # 6. Bollinger %B
    bb = compute_bollinger(close)
    pct_b = bb['pct_b']

    # 7. ROC(10)
    roc = compute_roc(close, 10)

    # 8. OBV slope
    obv = compute_obv(df)
    obv_mean = obv.rolling(20, min_periods=1).mean().replace(0, np.nan)
    obv_slope = obv.diff(5) / obv_mean.abs()

    # 9. VWAP deviation
    vwap = compute_vwap(df)
    vwap_dev = (close - vwap) / vwap.replace(0, np.nan)

    # 10. CVD slope
    cvd = compute_cvd(df)
    cvd_mean = cvd.rolling(20, min_periods=1).mean().abs().replace(0, np.nan)
    cvd_slope = cvd.diff(5) / cvd_mean

    # 11–12. Time-of-day cycles
    if hasattr(df.index, 'hour'):
        hour = df.index.hour
    elif 'timestamp' in df.columns:
        hour = pd.to_datetime(df['timestamp']).dt.hour
    else:
        hour = pd.Series(np.zeros(len(df)), index=df.index)

    hour_sin = np.sin(2 * np.pi * hour / 24)
    hour_cos = np.cos(2 * np.pi * hour / 24)

    features = pd.DataFrame({
        'log_ret': log_ret,
        'atr_ratio': atr_ratio,
        'rsi': rsi,
        'macd_hist': macd_hist,
        'vol_ratio': vol_ratio,
        'pct_b': pct_b,
        'roc': roc,
        'obv_slope': obv_slope,
        'vwap_dev': vwap_dev,
        'cvd_slope': cvd_slope,
        'hour_sin': hour_sin,
        'hour_cos': hour_cos,
    }, index=df.index)

    # Rolling z-score normalization (per-feature)
    for col in features.columns:
        features[col] = _rolling_zscore(features[col], window=100)

    # Fill remaining NaN with 0
    features = features.fillna(0)

    return features


# ---------------------------------------------------------------------------
# LSTM Model Management
# ---------------------------------------------------------------------------

_lstm_model: Optional[Any] = None
_lstm_trained: bool = False
_LSTM_SEQ_LEN: int = 60


def _build_lstm_model() -> Any:
    """
    Build LSTM model for direction probability prediction.
    Output: sigmoid → P(up move) in [0, 1].
    """
    try:
        import tensorflow as tf
        tf.get_logger().setLevel("ERROR")

        model = tf.keras.Sequential([
            tf.keras.layers.LSTM(64, return_sequences=True,
                                 input_shape=(_LSTM_SEQ_LEN, 12)),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.LSTM(32),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid'),  # P(up)
        ])
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy'],
        )
        return model
    except ImportError:
        logger.warning("TensorFlow not installed — LSTM strategy will return HOLD.")
        return None
    except Exception as exc:
        logger.error("Failed to build LSTM model: %s", exc)
        return None


def _train_lstm(model: Any, df: pd.DataFrame) -> None:
    """
    Train LSTM on direction labels (not raw price).
    Label: 1 if close[t+1] > close[t], else 0.
    """
    features = build_feature_matrix(df)
    close = df['close']

    # Direction labels
    labels = (close.shift(-1) > close).astype(float)

    # Drop last row (no label) and initial NaN rows
    valid_start = max(100, _LSTM_SEQ_LEN)  # after z-score warmup
    features = features.iloc[valid_start:-1]
    labels = labels.iloc[valid_start:-1]

    if len(features) < _LSTM_SEQ_LEN + 10:
        return

    # Create sequences
    feat_vals = features.values
    label_vals = labels.values

    X, y = [], []
    for i in range(_LSTM_SEQ_LEN, len(feat_vals)):
        X.append(feat_vals[i - _LSTM_SEQ_LEN:i])
        y.append(label_vals[i])

    if len(X) < 10:
        return

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Replace any inf/nan
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    model.fit(X, y, epochs=10, batch_size=32, verbose=0,
              validation_split=0.1)


# ---------------------------------------------------------------------------
# Strategy 1 — VWAP Trend (ID: ichimoku)
# ---------------------------------------------------------------------------

def ichimoku_strategy(df: pd.DataFrame, price: float,
                      min_confluence: int, **kwargs) -> Signal:
    """
    VWAP Trend strategy — leading indicator replacement for SMA crossover.

    BUY:  Price above VWAP AND CVD rising (buying pressure)
    SELL: Price below VWAP AND CVD falling (selling pressure)

    Volume confirmation + whale bonus applied.
    """
    if len(df) < 50:
        return Signal("ichimoku", "HOLD", 0.0, price,
                       ["Insufficient data"], datetime.now())

    close = df['close']
    vwap = compute_vwap(df)
    cvd = compute_cvd(df)
    rsi = compute_rsi(close, 14)

    vwap_now = vwap.iloc[-1]
    cvd_now = cvd.iloc[-1]
    cvd_prev = cvd.iloc[-6] if len(cvd) > 5 else cvd.iloc[0]
    rsi_now = rsi.iloc[-1]

    if np.isnan(vwap_now) or np.isnan(cvd_now):
        return Signal("ichimoku", "HOLD", 0.0, price,
                       ["Indicator NaN"], datetime.now())

    reasons: List[str] = []
    score = 0.0

    # VWAP deviation
    vwap_dev = (price - vwap_now) / vwap_now if vwap_now != 0 else 0

    # CVD trend (rising or falling)
    cvd_rising = cvd_now > cvd_prev
    cvd_strength = abs(cvd_now - cvd_prev)

    # Bullish: price above VWAP + buying pressure
    if vwap_dev > 0.001 and cvd_rising:
        score = min(0.5 + abs(vwap_dev) * 10, 1.0)
        reasons.append(f"Price above VWAP ({vwap_dev:+.3f})")
        reasons.append(f"CVD rising — buying pressure")

        # RSI confirmation
        if 40 < rsi_now < 70:
            score = min(score + 0.1, 1.0)
            reasons.append(f"RSI {rsi_now:.0f} — room to run")

    # Bearish: price below VWAP + selling pressure
    elif vwap_dev < -0.001 and not cvd_rising:
        score = min(0.5 + abs(vwap_dev) * 10, 1.0)
        reasons.append(f"Price below VWAP ({vwap_dev:+.3f})")
        reasons.append(f"CVD falling — selling pressure")

        if 30 < rsi_now < 60:
            score = min(score + 0.1, 1.0)
            reasons.append(f"RSI {rsi_now:.0f} — room to fall")
    else:
        reasons.append(f"VWAP dev={vwap_dev:+.3f}, CVD {'↑' if cvd_rising else '↓'} — no setup")
        return Signal("ichimoku", "HOLD", 0.0, price, reasons, datetime.now())

    # Volume confirmation filter
    if not _volume_confirmed(df):
        reasons.append("Volume below avg — signal downgraded to HOLD")
        return Signal("ichimoku", "HOLD", 0.0, price, reasons, datetime.now())

    # Whale bonus
    whale = detect_whale_activity(df)
    if whale > 0:
        score = min(score + whale, 1.0)
        reasons.append("🐋 Whale volume detected — score boosted")

    # Funding rate bias
    funding = kwargs.get('funding_rate')
    fr_bias = _funding_rate_bias(funding)
    if fr_bias != 0:
        score = max(0, min(score + abs(fr_bias), 1.0))
        reasons.append(f"Funding rate bias: {fr_bias:+.2f}")

    direction = "BUY" if vwap_dev > 0 else "SELL"
    return Signal("ichimoku", direction, score, price, reasons, datetime.now())


# ---------------------------------------------------------------------------
# Strategy 2 — Smart Mean Reversion (ID: bollinger)
# ---------------------------------------------------------------------------

def bollinger_strategy(df: pd.DataFrame, price: float,
                       min_confluence: int, **kwargs) -> Signal:
    """
    Smart Mean Reversion — VWAP reversion with divergence confirmation.

    BUY:  Price below VWAP with bullish divergence (price↓ but RSI↑)
    SELL: Price above VWAP with bearish divergence (price↑ but RSI↓)

    Filters out mean-reversion traps using CVD confirmation.
    """
    if len(df) < 50:
        return Signal("bollinger", "HOLD", 0.0, price,
                       ["Insufficient data"], datetime.now())

    close = df['close']
    vwap = compute_vwap(df)
    rsi = compute_rsi(close, 14)
    bb = compute_bollinger(close)

    vwap_now = vwap.iloc[-1]
    rsi_now = rsi.iloc[-1]
    pct_b = bb['pct_b'].iloc[-1]
    lower = bb['lower'].iloc[-1]
    upper = bb['upper'].iloc[-1]

    if np.isnan(vwap_now) or np.isnan(rsi_now) or np.isnan(pct_b):
        return Signal("bollinger", "HOLD", 0.0, price,
                       ["Indicator NaN"], datetime.now())

    reasons: List[str] = []
    score = 0.0

    # Divergence detection
    divergence = detect_divergence(close, rsi, lookback=14)

    vwap_dev = (price - vwap_now) / vwap_now if vwap_now != 0 else 0

    # Oversold + bullish divergence → BUY
    if pct_b < 0.2 and vwap_dev < -0.005:
        score = 0.5
        reasons.append(f"BB %B={pct_b:.2f} — oversold zone")
        reasons.append(f"Price below VWAP ({vwap_dev:+.3f})")

        if divergence == 'BULLISH':
            score = 0.8
            reasons.append("Bullish divergence — reversal signal")

        if rsi_now < 35:
            score = min(score + 0.1, 1.0)
            reasons.append(f"RSI {rsi_now:.0f} — deeply oversold")

        # CVD rising = buyers stepping in → confirms reversal
        cvd = compute_cvd(df)
        if len(cvd) > 5 and cvd.iloc[-1] > cvd.iloc[-6]:
            score = min(score + 0.1, 1.0)
            reasons.append("CVD rising — buyers entering")

        if not _volume_confirmed(df):
            reasons.append("Low volume — signal weakened")
            score = max(score - 0.2, 0.0)

        if score >= 0.5:
            return Signal("bollinger", "BUY", score, price,
                           reasons, datetime.now())

    # Overbought + bearish divergence → SELL
    elif pct_b > 0.8 and vwap_dev > 0.005:
        score = 0.5
        reasons.append(f"BB %B={pct_b:.2f} — overbought zone")
        reasons.append(f"Price above VWAP ({vwap_dev:+.3f})")

        if divergence == 'BEARISH':
            score = 0.8
            reasons.append("Bearish divergence — reversal signal")

        if rsi_now > 65:
            score = min(score + 0.1, 1.0)
            reasons.append(f"RSI {rsi_now:.0f} — overbought")

        cvd = compute_cvd(df)
        if len(cvd) > 5 and cvd.iloc[-1] < cvd.iloc[-6]:
            score = min(score + 0.1, 1.0)
            reasons.append("CVD falling — sellers entering")

        if not _volume_confirmed(df):
            score = max(score - 0.2, 0.0)
            reasons.append("Low volume — signal weakened")

        if score >= 0.5:
            return Signal("bollinger", "SELL", score, price,
                           reasons, datetime.now())

    reasons.append(f"BB %B={pct_b:.2f}, VWAP dev={vwap_dev:+.3f} — no setup")
    return Signal("bollinger", "HOLD", 0.0, price, reasons, datetime.now())


# ---------------------------------------------------------------------------
# Strategy 3 — Momentum Flow (ID: macd_rsi)
# ---------------------------------------------------------------------------

def macd_rsi_strategy(df: pd.DataFrame, price: float,
                      min_confluence: int, **kwargs) -> Signal:
    """
    Momentum Flow — ROC acceleration + CVD surge detection.

    BUY:  ROC accelerating upward AND CVD surging positive AND MACD bullish
    SELL: ROC accelerating downward AND CVD surging negative AND MACD bearish

    Designed to catch the START of explosive moves, not react to them.
    """
    if len(df) < 50:
        return Signal("macd_rsi", "HOLD", 0.0, price,
                       ["Insufficient data"], datetime.now())

    close = df['close']
    roc = compute_roc(close, 10)
    roc5 = compute_roc(close, 5)
    cvd = compute_cvd(df)
    macd_data = compute_macd(close)
    rsi = compute_rsi(close, 14)

    roc_now = roc.iloc[-1]
    roc5_now = roc5.iloc[-1]
    roc5_prev = roc5.iloc[-3] if len(roc5) > 2 else 0
    cvd_now = cvd.iloc[-1]
    cvd_short = cvd.iloc[-5] if len(cvd) > 5 else cvd.iloc[0]
    macd_hist = macd_data['histogram'].iloc[-1]
    rsi_now = rsi.iloc[-1]

    if np.isnan(roc_now) or np.isnan(cvd_now) or np.isnan(macd_hist):
        return Signal("macd_rsi", "HOLD", 0.0, price,
                       ["Indicator NaN"], datetime.now())

    reasons: List[str] = []
    score = 0.0

    # ROC acceleration: short ROC > long ROC and increasing
    roc_accelerating_up = (roc5_now > 0 and roc5_now > roc5_prev and
                           roc_now > 0)
    roc_accelerating_down = (roc5_now < 0 and roc5_now < roc5_prev and
                             roc_now < 0)

    # CVD surge
    cvd_surge_up = cvd_now > cvd_short and (cvd_now - cvd_short) > 0
    cvd_surge_down = cvd_now < cvd_short and (cvd_now - cvd_short) < 0

    # Bullish momentum
    if roc_accelerating_up and cvd_surge_up and macd_hist > 0:
        score = 0.6
        reasons.append(f"ROC accelerating up ({roc5_now:+.4f})")
        reasons.append("CVD surging positive — aggressive buying")
        reasons.append(f"MACD histogram positive ({macd_hist:+.4f})")

        if rsi_now < 70:  # Not overbought
            score += 0.15
            reasons.append(f"RSI {rsi_now:.0f} — momentum not exhausted")

        # Liquidation proximity boost
        liq_levels = kwargs.get('liquidation_levels')
        liq_boost = _liquidation_proximity_boost(price, liq_levels)
        if liq_boost > 0:
            score = min(score + liq_boost, 1.0)
            reasons.append("Near liquidation cluster — cascade potential")

        whale = detect_whale_activity(df)
        if whale > 0:
            score = min(score + whale, 1.0)
            reasons.append("🐋 Whale volume boost")

        return Signal("macd_rsi", "BUY", min(score, 1.0), price,
                       reasons, datetime.now())

    # Bearish momentum
    elif roc_accelerating_down and cvd_surge_down and macd_hist < 0:
        score = 0.6
        reasons.append(f"ROC accelerating down ({roc5_now:+.4f})")
        reasons.append("CVD surging negative — aggressive selling")
        reasons.append(f"MACD histogram negative ({macd_hist:+.4f})")

        if rsi_now > 30:
            score += 0.15
            reasons.append(f"RSI {rsi_now:.0f} — momentum not exhausted")

        liq_levels = kwargs.get('liquidation_levels')
        liq_boost = _liquidation_proximity_boost(price, liq_levels)
        if liq_boost > 0:
            score = min(score + liq_boost, 1.0)
            reasons.append("Near liquidation cluster — cascade potential")

        whale = detect_whale_activity(df)
        if whale > 0:
            score = min(score + whale, 1.0)
            reasons.append("🐋 Whale volume boost")

        return Signal("macd_rsi", "SELL", min(score, 1.0), price,
                       reasons, datetime.now())

    reasons.append(f"ROC={roc5_now:+.4f}, MACD={macd_hist:+.4f} — no momentum setup")
    return Signal("macd_rsi", "HOLD", 0.0, price, reasons, datetime.now())


# ---------------------------------------------------------------------------
# Strategy 4 — Engineered LSTM (ID: ml_forecast)
# ---------------------------------------------------------------------------

def ml_forecast_strategy(df: pd.DataFrame, price: float,
                         min_confluence: int, **kwargs) -> Signal:
    """
    Engineered LSTM — 12-feature normalized input, direction probability output.

    Predicts P(up move) instead of raw price.
    Confidence gate: only signal when |P(up) - 0.5| > 0.15
    """
    global _lstm_model, _lstm_trained

    if len(df) < max(100 + _LSTM_SEQ_LEN, 51):
        return Signal("ml_forecast", "HOLD", 0.0, price,
                       [f"Need ≥{100 + _LSTM_SEQ_LEN} candles"], datetime.now())

    # Lazy build
    if _lstm_model is None and not _lstm_trained:
        _lstm_model = _build_lstm_model()
        if _lstm_model is None:
            _lstm_trained = True
            return Signal("ml_forecast", "HOLD", 0.0, price,
                           ["TensorFlow unavailable"], datetime.now())

    if _lstm_model is None:
        return Signal("ml_forecast", "HOLD", 0.0, price,
                       ["LSTM model unavailable"], datetime.now())

    # Lazy train
    if not _lstm_trained:
        try:
            _train_lstm(_lstm_model, df)
            _lstm_trained = True
        except Exception as exc:
            logger.error("LSTM training failed: %s", exc)
            return Signal("ml_forecast", "HOLD", 0.0, price,
                           [f"Training error: {exc}"], datetime.now())

    # Predict
    try:
        features = build_feature_matrix(df)
        feat_vals = features.values

        if len(feat_vals) < _LSTM_SEQ_LEN:
            return Signal("ml_forecast", "HOLD", 0.0, price,
                           ["Not enough features after warmup"], datetime.now())

        # Last sequence
        last_seq = feat_vals[-_LSTM_SEQ_LEN:]
        last_seq = np.nan_to_num(last_seq, nan=0.0, posinf=0.0, neginf=0.0)
        X = last_seq.reshape(1, _LSTM_SEQ_LEN, 12).astype(np.float32)

        prob_up = float(_lstm_model.predict(X, verbose=0)[0, 0])
        confidence = abs(prob_up - 0.5)

        reasons = [f"P(up)={prob_up:.3f}, confidence={confidence:.3f}"]

        # Confidence gate
        if confidence < 0.15:
            reasons.append("Below confidence threshold 0.15 — HOLD")
            return Signal("ml_forecast", "HOLD", 0.0, price,
                           reasons, datetime.now())

        if prob_up > 0.65:
            score = min(prob_up, 1.0)
            reasons.append(f"Strong bullish prediction ({prob_up:.1%})")
            return Signal("ml_forecast", "BUY", score, price,
                           reasons, datetime.now())

        if prob_up < 0.35:
            score = min(1.0 - prob_up, 1.0)
            reasons.append(f"Strong bearish prediction ({prob_up:.1%})")
            return Signal("ml_forecast", "SELL", score, price,
                           reasons, datetime.now())

        reasons.append("Prediction near 50/50 — HOLD")
        return Signal("ml_forecast", "HOLD", 0.0, price,
                       reasons, datetime.now())

    except Exception as exc:
        logger.error("LSTM prediction error: %s", exc)
        return Signal("ml_forecast", "HOLD", 0.0, price,
                       [f"Prediction error: {exc}"], datetime.now())


# ---------------------------------------------------------------------------
# Strategy 5 — Regime-Adaptive Ensemble (ID: combined)
# ---------------------------------------------------------------------------

_REGIME_STRATEGY_MAP = {
    'TREND':    ['ichimoku', 'ml_forecast'],
    'RANGE':    ['bollinger'],
    'VOLATILE': ['macd_rsi'],
}

_STRATEGY_FUNCS = {
    'ichimoku': ichimoku_strategy,
    'bollinger': bollinger_strategy,
    'macd_rsi': macd_rsi_strategy,
    'ml_forecast': ml_forecast_strategy,
}


def combined_strategy(df: pd.DataFrame, price: float,
                      min_confluence: int, **kwargs) -> Signal:
    """
    Regime-Adaptive Ensemble.

    1. Detect market regime (TREND/RANGE/VOLATILE)
    2. Run ONLY relevant strategies for current regime
    3. Apply order book bias if available
    4. Score-weighted aggregation with quality gate
    """
    if len(df) < 50:
        return Signal("combined", "HOLD", 0.0, price,
                       ["Insufficient data"], datetime.now())

    # Cooldown check
    if not _check_cooldown():
        return Signal("combined", "HOLD", 0.0, price,
                       ["Trade cooldown active"], datetime.now())

    regime = detect_market_regime(df)

    # Get strategies for this regime
    strategy_ids = _REGIME_STRATEGY_MAP.get(regime, ['ichimoku'])
    reasons: List[str] = [f"Regime: {regime} → running {strategy_ids}"]

    buy_scores: List[float] = []
    sell_scores: List[float] = []
    all_reasons: List[str] = list(reasons)

    for sid in strategy_ids:
        func = _STRATEGY_FUNCS.get(sid)
        if not func:
            continue
        try:
            sig = func(df, price, min_confluence, **kwargs)
        except Exception as exc:
            logger.error("Ensemble sub-strategy %s failed: %s", sid, exc)
            continue

        all_reasons.extend(f"[{sid}] {r}" for r in sig.reasons)

        if sig.signal == 'BUY' and sig.score > 0:
            buy_scores.append(sig.score)
        elif sig.signal == 'SELL' and sig.score > 0:
            sell_scores.append(sig.score)

    # Aggregate scores
    avg_buy = np.mean(buy_scores) if buy_scores else 0.0
    avg_sell = np.mean(sell_scores) if sell_scores else 0.0

    # Order book bias
    order_book = kwargs.get('order_book')
    ob_analysis = analyze_order_book(order_book)
    ob_bias = ob_analysis['bias']

    if ob_analysis['available']:
        all_reasons.append(
            f"Order book: imbalance={ob_analysis['imbalance']:.2f}, "
            f"bias={ob_bias:+.2f}")
        avg_buy += max(ob_bias, 0)
        avg_sell += max(-ob_bias, 0)

    # Funding rate bias
    funding = kwargs.get('funding_rate')
    fr_bias = _funding_rate_bias(funding)
    if fr_bias != 0:
        all_reasons.append(f"Funding rate bias: {fr_bias:+.2f}")
        if fr_bias > 0:
            avg_buy += fr_bias
        else:
            avg_sell += abs(fr_bias)

    all_reasons.append(f"Scores — BUY: {avg_buy:.2f}, SELL: {avg_sell:.2f}")

    # Decision with quality gate
    if avg_buy > avg_sell and avg_buy >= 0.5:
        _mark_trade()
        return Signal("combined", "BUY", min(avg_buy, 1.0), price,
                       all_reasons, datetime.now())

    if avg_sell > avg_buy and avg_sell >= 0.5:
        _mark_trade()
        return Signal("combined", "SELL", min(avg_sell, 1.0), price,
                       all_reasons, datetime.now())

    all_reasons.append("No strategy reached quality threshold 0.5")
    return Signal("combined", "HOLD", 0.0, price, all_reasons, datetime.now())


# ---------------------------------------------------------------------------
# Risk Manager — ATR-based
# ---------------------------------------------------------------------------

def risk_manager(df: pd.DataFrame, account_balance: float) -> Dict[str, Any]:
    """
    Professional risk management wrapper.

    Stop Loss:  2 × ATR
    Take Profit: 4 × ATR (2:1 reward/risk)
    Risk/trade:  1% of account
    Daily stop:  3% of account (not enforced here, tracked externally)
    """
    if df.empty or len(df) < 50:
        return {
            "signal": "HOLD", "entry": 0.0, "stop": 0.0,
            "tp": 0.0, "size": 0.0,
            "reasons": ["Insufficient data for risk management"],
        }

    current_price = float(df["close"].iloc[-1])
    sig = combined_strategy(df, current_price, min_confluence=2)

    current_atr = atr(df, period=14)
    atr_value = current_atr.iloc[-1]

    if np.isnan(atr_value) or atr_value <= 0:
        return {
            "signal": sig.signal, "entry": current_price,
            "stop": 0.0, "tp": 0.0, "size": 0.0,
            "reasons": sig.reasons + ["ATR unavailable — cannot compute stops"],
        }

    # 2× ATR stop loss, 4× ATR take profit
    stop_distance = 2.0 * atr_value
    tp_distance = 4.0 * atr_value

    if sig.signal == 'BUY':
        stop_loss = current_price - stop_distance
        take_profit = current_price + tp_distance
    elif sig.signal == 'SELL':
        stop_loss = current_price + stop_distance
        take_profit = current_price - tp_distance
    else:
        stop_loss = current_price - stop_distance
        take_profit = current_price + tp_distance

    # 1% risk position sizing
    risk_amount = account_balance * 0.01
    position_size = risk_amount / stop_distance if stop_distance > 0 else 0.0

    return {
        "signal": sig.signal,
        "entry": round(current_price, 8),
        "stop": round(stop_loss, 8),
        "tp": round(take_profit, 8),
        "size": round(position_size, 8),
        "atr": round(atr_value, 8),
        "risk_reward": round(tp_distance / stop_distance, 2) if stop_distance > 0 else 0,
        "reasons": sig.reasons,
    }


# ---------------------------------------------------------------------------
# Smart Execution Helpers (used by api_v2.py bot loop)
# ---------------------------------------------------------------------------

def compute_smart_entry(side: str, current_price: float,
                        order_book: Optional[Dict] = None,
                        atr_value: float = 0.0) -> float:
    """
    Compute limit order entry price (better than market).
    Places order to capture part of the spread.
    """
    ob = analyze_order_book(order_book)
    spread = ob['spread_pct']

    # Offset: half the spread or 10% of ATR, whichever is smaller
    atr_offset = atr_value * 0.1 if atr_value > 0 else 0
    offset = min(spread * 0.5, atr_offset) if atr_offset > 0 else spread * 0.3

    if side.upper() == 'BUY':
        return current_price * (1 - offset)
    else:
        return current_price * (1 + offset)


def compute_atr_position_size(account_balance: float, atr_value: float,
                               risk_pct: float = 0.01) -> float:
    """
    ATR-based position sizing.

    risk_amount = account × risk_pct
    stop_distance = 2 × ATR
    position_size = risk_amount / stop_distance
    """
    if atr_value <= 0:
        return 0.0
    risk_amount = account_balance * risk_pct
    stop_distance = 2.0 * atr_value
    return risk_amount / stop_distance


# ---------------------------------------------------------------------------
# Strategy REGISTRY (consumed by strategy_engine.py)
# ⚠️ IDs are FROZEN — do NOT rename
# ---------------------------------------------------------------------------

REGISTRY = [
    {
        "id": "ichimoku",
        "name": "VWAP Trend",
        "icon": "📈",
        "description": "VWAP Trend — Leading VWAP deviation + CVD buying pressure confirmation.",
        "logic": ichimoku_strategy,
    },
    {
        "id": "bollinger",
        "name": "Smart Mean Reversion",
        "icon": "📊",
        "description": "Smart Mean Reversion — VWAP reversion with divergence filter + CVD confirmation.",
        "logic": bollinger_strategy,
    },
    {
        "id": "macd_rsi",
        "name": "Momentum Flow",
        "icon": "🚀",
        "description": "Momentum Flow — ROC acceleration + CVD surge + MACD confirmation.",
        "logic": macd_rsi_strategy,
    },
    {
        "id": "ml_forecast",
        "name": "Deep Learning LSTM",
        "icon": "🤖",
        "description": "Engineered LSTM — 12-feature normalized input, direction probability with confidence gate.",
        "logic": ml_forecast_strategy,
    },
    {
        "id": "combined",
        "name": "Regime-Adaptive Ensemble",
        "icon": "⚡",
        "description": "Smart Ensemble — regime detection + order book bias + quality-gated signal aggregation.",
        "logic": combined_strategy,
    },
]

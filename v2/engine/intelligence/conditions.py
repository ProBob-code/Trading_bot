"""
Market Conditions Read
======================

Replaces the old "AI Momentum Pulse", which picked a random sentence from a
hardcoded list and showed a score derived from three if-statements. It looked
intelligent and told the trader nothing — the same headline could appear in
opposite markets, and several templates named the internal strategies.

This computes four independent, honest measurements from real candles:

  trend        direction and how committed the move is
  volatility   today's range against its own recent normal
  participation volume now against its own recent normal
  stretch      how far price sits from its own mean

and turns them into the one thing a trader actually needs from a panel like
this: **is now a good time to be trading this instrument, and why not.**

Every number shown is derived, reproducible and explainable. Nothing is
randomised, and no strategy internals are ever referenced.
"""

from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from loguru import logger


def _pct_rank(series: pd.Series, value: float) -> float:
    """Where `value` sits within `series`, 0-100."""
    clean = series.dropna()
    if clean.empty:
        return 50.0
    return float((clean < value).sum()) / len(clean) * 100.0


def _safe_last(series: pd.Series, default: float = 0.0) -> float:
    try:
        val = float(series.iloc[-1])
        return default if np.isnan(val) else val
    except Exception:
        return default


def analyse(df: pd.DataFrame) -> Optional[Dict]:
    """
    Measure current conditions from OHLCV candles.

    Returns None when there is not enough history to say anything honest —
    the caller shows "not enough data" rather than inventing a number.
    """
    if df is None or df.empty or len(df) < 60:
        return None

    close = df['close'].astype(float)
    high = df['high'].astype(float)
    low = df['low'].astype(float)
    volume = df['volume'].astype(float) if 'volume' in df else pd.Series([0.0] * len(df))

    price = _safe_last(close)
    if price <= 0:
        return None

    # ── Trend: where price sits relative to its own fast/slow averages, and
    #    whether those averages agree with each other. ──
    fast = close.rolling(20).mean()
    slow = close.rolling(50).mean()
    fast_v, slow_v = _safe_last(fast, price), _safe_last(slow, price)

    spread_pct = ((fast_v - slow_v) / slow_v * 100.0) if slow_v else 0.0
    above_fast = price > fast_v
    above_slow = price > slow_v

    # Agreement of three independent facts, each worth a third.
    agree = sum([above_fast, above_slow, fast_v > slow_v])
    trend_dir = 'up' if agree >= 2 else 'down'
    trend_strength = abs(agree - 1.5) / 1.5 * 100.0    # 0 = conflicted, 100 = aligned

    # ── Volatility: today's true range against its own 60-bar distribution. ──
    prev_close = close.shift(1)
    true_range = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = true_range.rolling(14).mean()
    atr_now = _safe_last(atr)
    atr_pct = (atr_now / price * 100.0) if price else 0.0
    vol_rank = _pct_rank(atr.tail(120), atr_now)

    # ── Participation: current volume against its own recent normal. ──
    vol_avg = _safe_last(volume.rolling(20).mean(), 0.0)
    vol_now = _safe_last(volume, 0.0)
    rel_volume = (vol_now / vol_avg) if vol_avg > 0 else 1.0
    participation = min(100.0, rel_volume * 50.0)      # 1.0x normal -> 50

    # ── Stretch: distance from the mean in units of its own deviation. ──
    mean20 = _safe_last(fast, price)
    std20 = _safe_last(close.rolling(20).std(), 0.0)
    stretch_z = ((price - mean20) / std20) if std20 > 0 else 0.0

    return {
        'price': price,
        'trend': {
            'direction': trend_dir,
            'strength': round(trend_strength, 1),
            'spread_pct': round(spread_pct, 3),
            'aligned': agree == 3 or agree == 0,
        },
        'volatility': {
            'atr_pct': round(atr_pct, 3),
            'rank': round(vol_rank, 1),
        },
        'participation': {
            'relative': round(rel_volume, 2),
            'score': round(participation, 1),
        },
        'stretch': {
            'z': round(stretch_z, 2),
        },
    }


def plain_bands(m: Dict) -> Dict:
    """
    Turn each measurement into a word a trader reads at a glance.

    The raw figures (a z-score, a percentile, a volume multiple) are accurate
    but they are not what someone wants to read mid-session. Each band keeps
    the exact number alongside for the tooltip.
    """
    trend = m['trend']
    vol = m['volatility']
    part = m['participation']
    z = m['stretch']['z']

    ts = trend['strength']
    if ts >= 80:
        trend_word, trend_tone = 'Strong', 'good'
    elif ts >= 50:
        trend_word, trend_tone = 'Building', 'good'
    elif ts >= 25:
        trend_word, trend_tone = 'Weak', 'warn'
    else:
        trend_word, trend_tone = 'None', 'bad'

    vr = vol['rank']
    if vr < 20:
        vol_word, vol_tone = 'Quiet', 'bad'
    elif vr < 70:
        vol_word, vol_tone = 'Normal', 'good'
    elif vr < 88:
        vol_word, vol_tone = 'Lively', 'good'
    else:
        vol_word, vol_tone = 'Wild', 'warn'

    rv = part['relative']
    if rv < 0.6:
        vol2_word, vol2_tone = 'Thin', 'bad'
    elif rv < 1.4:
        vol2_word, vol2_tone = 'Normal', 'none'
    elif rv < 2.5:
        vol2_word, vol2_tone = 'Busy', 'good'
    else:
        vol2_word, vol2_tone = 'Heavy', 'good'

    az = abs(z)
    if az < 0.8:
        st_word, st_tone = 'Fair value', 'good'
    elif az < 1.6:
        st_word, st_tone = 'Drifting', 'none'
    elif az < 2.2:
        st_word, st_tone = 'Stretched', 'warn'
    else:
        st_word, st_tone = 'Overstretched', 'bad'

    arrow = '\u2191' if trend['direction'] == 'up' else '\u2193'

    return {
        'trend': {
            'word': '%s %s' % (arrow, trend_word),
            'tone': trend_tone if trend_word != 'None' else 'bad',
            'detail': 'Price and its two trend references agree %d%% of the way.' % round(ts),
        },
        'volatility': {
            'word': vol_word,
            'tone': vol_tone,
            'detail': 'Today\u2019s range sits above %d%% of recent sessions (%.2f%% per bar).'
                      % (round(vr), vol['atr_pct']),
        },
        'volume': {
            'word': vol2_word,
            'tone': vol2_tone,
            'detail': 'Trading %.2f\u00d7 the recent average volume.' % rv,
        },
        'stretch': {
            'word': st_word,
            'tone': st_tone,
            'detail': 'Price is %.1f standard deviations from its recent average.' % z,
        },
    }


def verdict(m: Dict) -> Dict:
    """
    Turn measurements into a tradeability call plus the reasoning.

    The score answers one question: how favourable are conditions for taking a
    position right now? It is NOT a direction forecast — a market can be
    strongly trending down and still score well.
    """
    trend = m['trend']
    vol = m['volatility']
    part = m['participation']
    stretch = m['stretch']

    reasons: List[Dict[str, str]] = []
    score = 50.0

    # Directional conviction helps — chop does not.
    if trend['strength'] >= 66:
        score += 18
        reasons.append({
            'kind': 'good',
            'text': 'Everything points the same way \u2014 the market is heading %s.' % trend['direction'],
        })
    elif trend['strength'] <= 33:
        score -= 18
        reasons.append({
            'kind': 'bad',
            'text': 'The market keeps changing its mind — there is no clear direction to trade.',
        })

    # Volatility: too little means nothing to capture, too much means stops get hit.
    if vol['rank'] < 20:
        score -= 15
        reasons.append({
            'kind': 'bad',
            'text': 'Barely moving — quieter than %d%% of recent sessions, so gains may not cover costs.'
                    % round(vol['rank']),
        })
    elif vol['rank'] > 85:
        score -= 10
        reasons.append({
            'kind': 'warn',
            'text': 'Moving unusually hard — wilder than %d%% of recent sessions. Trade smaller and give stops room.'
                    % round(vol['rank']),
        })
    else:
        score += 12
        reasons.append({
            'kind': 'good',
            'text': 'Moving a healthy amount — enough to trade, not so much it whipsaws you.',
        })

    # Participation: a move without volume behind it is easily reversed.
    if part['relative'] < 0.6:
        score -= 12
        reasons.append({
            'kind': 'bad',
            'text': 'Few people trading right now — expect worse prices getting in and out.',
        })
    elif part['relative'] > 1.4:
        score += 10
        reasons.append({
            'kind': 'good',
            'text': 'Plenty of people trading — there is real weight behind this move.',
        })

    # Stretch matters differently depending on what you intend to do.
    z = stretch['z']
    if abs(z) > 2:
        reasons.append({
            'kind': 'warn',
            'text': 'Price has run a long way from normal — late to join, and it may snap back.',
        })
        score -= 8
    elif abs(z) < 0.5 and trend['strength'] >= 66:
        reasons.append({
            'kind': 'good',
            'text': 'Price is near its normal level while the trend holds — a lower-risk place to join.',
        })
        score += 8

    score = max(0.0, min(100.0, score))

    if score >= 70:
        label, headline = 'GOOD TO TRADE', 'The market is behaving \u2014 conditions favour taking a position.'
    elif score >= 45:
        label, headline = 'MIXED', 'Conditions are workable but not clean — be selective.'
    else:
        label, headline = 'SIT THIS OUT', 'The market is not offering much right now. Patience pays here.'

    return {
        'score': round(score),
        'label': label,
        'headline': headline,
        'reasons': reasons,
    }


def read(df: pd.DataFrame, symbol: str = '') -> Dict:
    """Full conditions read for one instrument."""
    try:
        m = analyse(df)
        if not m:
            return {
                'available': False,
                'symbol': symbol,
                'bands': {},
                'label': 'NO DATA',
                'score': 0,
                'headline': 'Not enough history for this instrument yet.',
                'reasons': [],
                'metrics': {},
            }

        v = verdict(m)
        return {
            'available': True,
            'symbol': symbol,
            'score': v['score'],
            'label': v['label'],
            'headline': v['headline'],
            'reasons': v['reasons'],
            'bands': plain_bands(m),
            'metrics': {
                'trend_direction': m['trend']['direction'],
                'trend_strength': m['trend']['strength'],
                'volatility_rank': m['volatility']['rank'],
                'volatility_pct': m['volatility']['atr_pct'],
                'relative_volume': m['participation']['relative'],
                'stretch_z': m['stretch']['z'],
            },
        }
    except Exception as e:
        logger.error(f"[CONDITIONS] read failed for {symbol}: {e}")
        return {
            'available': False, 'symbol': symbol, 'bands': {}, 'label': 'UNAVAILABLE',
            'score': 0, 'headline': 'Could not read conditions.', 'reasons': [], 'metrics': {},
        }

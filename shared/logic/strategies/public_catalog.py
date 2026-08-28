"""
Public Strategy Catalog
=======================

The internal strategy ids and names describe the actual technique
("ichimoku", "bollinger", "macd_rsi", "Statistical Mean Reversion — OU
half-life and variance-ratio gated..."). Those are the firm's IP and must not
reach the browser, the API, or anything a user can inspect.

This module is the translation layer at the edge of the system:

    internal id  ──to_public()──▶  opaque code (GBX-04)
    opaque code  ──to_internal()─▶ internal id

Everything the client sees is the opaque code plus a deliberately vague
description of *what the strategy does for the trader*, never *how it works*.
Route handlers translate on the way out and back on the way in, so the engine
keeps using its real ids internally and nothing else has to change.

Adding a strategy: give it an entry here. Anything missing from the catalog is
masked behind a generated code rather than falling through to its real name —
failing closed is the whole point.
"""

import hashlib
from typing import Dict, List, Optional

# ── internal id → public code ──
# Codes are arbitrary and carry no meaning. Do not renumber them: they appear
# in persisted bot ids and in the user's saved preferences.
PUBLIC_CODES: Dict[str, str] = {
    'combined':       'GBX-01',
    'ichimoku':       'GBX-02',
    'bollinger':      'GBX-03',
    'macd_rsi':       'GBX-04',
    'ml_forecast':    'GBX-05',
    'quant_alpha':    'GBX-06',
    'quant_meanrev':  'GBX-07',
    'quant_momentum': 'GBX-08',
}

INTERNAL_IDS: Dict[str, str] = {code: internal for internal, code in PUBLIC_CODES.items()}

# ── public presentation ──
# Describe the BEHAVIOUR and when it suits the trader. Never name an indicator,
# a formula, a parameter, or the academic technique behind it.
PUBLIC_META: Dict[str, Dict[str, str]] = {
    'GBX-01': {
        'name': 'Adaptive Core',
        'icon': '⚡',
        'description': (
            'Reads what kind of market it is in and adjusts its approach as '
            'conditions change. The generalist — a sensible default when you '
            'are not sure which profile fits.'
        ),
        'best_for': 'Any market · a good starting point',
    },
    'GBX-02': {
        'name': 'Trend Rider',
        'icon': '📈',
        'description': (
            'Waits for a market to commit to a direction, then stays with the '
            'move. Deliberately inactive while price is going nowhere.'
        ),
        'best_for': 'Markets that are trending',
    },
    'GBX-03': {
        'name': 'Range Fader',
        'icon': '🎯',
        'description': (
            'Looks for prices that have stretched unusually far from their '
            'recent normal and positions for a return toward it.'
        ),
        'best_for': 'Calm, range-bound markets',
    },
    'GBX-04': {
        'name': 'Flow Momentum',
        'icon': '🌊',
        'description': (
            'Acts when the strength behind a move agrees with the direction of '
            'the move itself, and stands down when the two disagree.'
        ),
        'best_for': 'Actively moving markets',
    },
    'GBX-05': {
        'name': 'Predictive Model',
        'icon': '🧠',
        'description': (
            'A learned model that estimates where price is headed next and only '
            'acts when its own confidence clears a threshold.'
        ),
        'best_for': 'Liquid instruments with long history',
    },
    'GBX-06': {
        'name': 'Multi-Signal Ensemble',
        'icon': '🔀',
        'description': (
            'Combines many independent, individually-weak opinions into one '
            'decision, re-weighting them as conditions shift. Trades less often '
            'but with more agreement behind each entry.'
        ),
        'best_for': 'Traders who prefer fewer, higher-conviction trades',
    },
    'GBX-07': {
        'name': 'Statistical Reversion',
        'icon': '📊',
        'description': (
            'Positions against moves it measures as statistically overstretched, '
            'with checks that the stretch is the kind that historically corrects.'
        ),
        'best_for': 'Choppy markets without a strong trend',
    },
    'GBX-08': {
        'name': 'Persistence Engine',
        'icon': '🚀',
        'description': (
            'Rides moves that measurement suggests are likely to keep going, and '
            'ignores drift that looks random.'
        ),
        'best_for': 'Markets with sustained directional pressure',
    },
}

# ── Timeframe each strategy is designed to run on ──
# The interval is a property of the strategy, not a user preference: a trend
# strategy asked to decide every minute is reading noise, and a mean-reversion
# strategy on a daily candle never sees the stretch it exists to fade. The
# Command Deck therefore shows this rather than offering a dropdown.
#
# Keyed by PUBLIC code so this file stays the single edge-facing description of
# a strategy, alongside its name and behaviour.
DEFAULT_INTERVALS: Dict[str, str] = {
    'GBX-01': '15m',   # Adaptive Core — general purpose, mid timeframe
    'GBX-02': '1h',    # Trend Rider — needs a committed move to follow
    'GBX-03': '15m',   # Range Fader — fades intraday stretches
    'GBX-04': '15m',   # Flow Momentum — momentum/direction agreement
    'GBX-05': '1h',    # Predictive Model — forecasts over a longer horizon
    'GBX-06': '1h',    # Multi-Signal Ensemble — fewer, higher-conviction entries
    'GBX-07': '15m',   # Statistical Reversion — intraday z-score fades
    'GBX-08': '1h',    # Persistence Engine — rides sustained pressure
}

FALLBACK_INTERVAL = '15m'


def interval_for(internal_id: Optional[str]) -> str:
    """The timeframe a strategy is meant to trade on.

    Server-side callers use this instead of trusting a client-supplied
    interval, so removing the dropdown actually changes what runs rather than
    just hiding a field that still gets posted.
    """
    return DEFAULT_INTERVALS.get(to_public(internal_id), FALLBACK_INTERVAL)


# What the client is told when a strategy has no catalog entry. Unknown
# strategies are masked, never passed through under their real name.
_FALLBACK_META = {
    'name': 'Private Strategy',
    'icon': '🔒',
    'description': 'A proprietary profile. Details are not published.',
    'best_for': 'Specialist use',
}


def to_public(internal_id: Optional[str]) -> str:
    """Internal strategy id → opaque public code (fails closed)."""
    if not internal_id:
        return 'GBX-00'
    key = str(internal_id).strip().lower()
    if key in PUBLIC_CODES:
        return PUBLIC_CODES[key]
    # A custom user strategy is the user's own; it keeps its identity.
    if key.startswith('custom:'):
        return key
    # Anything unrecognised is masked rather than leaked. The digest keeps the
    # generated code stable across restarts (builtin hash() is salted).
    digest = hashlib.md5(key.encode('utf-8')).hexdigest()
    return 'GBX-%02d' % (int(digest[:8], 16) % 89 + 10)


def to_internal(public_code: Optional[str], default: str = 'combined') -> str:
    """Opaque public code → internal strategy id."""
    if not public_code:
        return default
    key = str(public_code).strip()
    if key in INTERNAL_IDS:
        return INTERNAL_IDS[key]
    if key.lower().startswith('custom:'):
        return key
    # Accept an internal id too, so server-side callers and older persisted
    # rows keep working. Never accept an unknown value.
    if key.lower() in PUBLIC_CODES:
        return key.lower()
    return default


def public_meta(internal_id: Optional[str]) -> Dict[str, str]:
    """The full public description for an internal strategy id."""
    code = to_public(internal_id)
    meta = PUBLIC_META.get(code, _FALLBACK_META)
    return {'id': code, **meta,
            'interval': DEFAULT_INTERVALS.get(code, FALLBACK_INTERVAL)}


def public_catalog(internal_ids: List[str] = None) -> List[Dict[str, str]]:
    """The catalog as the client should see it."""
    ids = internal_ids if internal_ids is not None else list(PUBLIC_CODES.keys())
    seen, out = set(), []
    for internal in ids:
        meta = public_meta(internal)
        if meta['id'] in seen:
            continue
        seen.add(meta['id'])
        out.append(meta)
    return out


def mask_bot_id(bot_id: Optional[str]) -> str:
    """
    Replace the strategy segment of a bot id with its public code.

    Bot ids are `u{uid}_{market}_{symbol}_{strategy}_{hash}`, so an unmasked id
    published the strategy name in the DOM. Symbols can themselves contain
    underscores, so this swaps the known token rather than splitting on '_'.
    """
    if not bot_id:
        return ''
    out = str(bot_id)
    for internal, code in PUBLIC_CODES.items():
        token = '_%s_' % internal
        if token in out:
            return out.replace(token, '_%s_' % code, 1)
    return out


def unmask_bot_id(bot_id: Optional[str]) -> str:
    """Inverse of mask_bot_id — public code back to the internal id."""
    if not bot_id:
        return ''
    out = str(bot_id)
    for code, internal in INTERNAL_IDS.items():
        token = '_%s_' % code
        if token in out:
            return out.replace(token, '_%s_' % internal, 1)
    return out

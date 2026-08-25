"""
Discord Daily Reports
=====================

Posts an end-of-day trading report per user into a shared Discord guild.

Design
------
* **Opt-in.** Nothing is posted for a user until they enable reports on the
  website. Trading activity is personal data; it is not published by default.
* **One channel per user.** On first approval the bot creates
  `#{username}_report` in the configured guild and remembers the channel id.
* **Two sections per post.** "Today" covers the session just ended; "Overall"
  covers the account since inception, so a reader sees both the day and the
  trajectory.
* **Strategy names are masked.** Reports go through the public catalog, so a
  Discord channel never publishes the firm's internal strategy ids.

Configuration (environment)
---------------------------
    DISCORD_BOT_TOKEN   bot token — REQUIRED, never commit this
    DISCORD_GUILD_ID    target server id
    DISCORD_REPORT_HOUR_UTC  hour to post (default 21)

The bot must be invited to the guild with the `bot` scope and needs
**Manage Channels** (to create the per-user channel) and **Send Messages**.
Without a token the module stays dormant and logs a single notice — the app
runs normally, reports simply do not send.
"""

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

import requests
from loguru import logger

from shared.logic.strategies.public_catalog import public_meta

DISCORD_API = 'https://discord.com/api/v10'

# Discord rejects channel names that are not lowercase-kebab.
_NAME_SAFE = 'abcdefghijklmnopqrstuvwxyz0123456789-_'


def _token() -> Optional[str]:
    return os.getenv('DISCORD_BOT_TOKEN') or None


def _guild_id() -> Optional[str]:
    return os.getenv('DISCORD_GUILD_ID') or None


def is_configured() -> bool:
    return bool(_token() and _guild_id())


def _headers() -> Dict[str, str]:
    return {
        'Authorization': 'Bot %s' % _token(),
        'Content-Type': 'application/json',
        'User-Agent': 'GoatBotTrade (https://goatbot.trade, 2.4)',
    }


def _request(method: str, path: str, payload: dict = None, retries: int = 2):
    """Call the Discord API, honouring rate limits."""
    url = DISCORD_API + path
    for attempt in range(retries + 1):
        try:
            resp = requests.request(method, url, headers=_headers(),
                                    json=payload, timeout=15)
        except Exception as e:
            logger.warning(f"[DISCORD] {method} {path} failed: {e}")
            return None

        if resp.status_code == 429:
            # Respect the retry window rather than hammering the endpoint.
            wait = float((resp.json() or {}).get('retry_after', 1.0))
            logger.warning(f"[DISCORD] rate limited, waiting {wait:.1f}s")
            time.sleep(min(wait, 10.0))
            continue

        if resp.status_code >= 400:
            logger.error(f"[DISCORD] {method} {path} -> {resp.status_code}: {resp.text[:300]}")
            return None

        return resp.json() if resp.text else {}

    return None


def channel_name_for(username: str) -> str:
    """`bobj` -> `bobj_report`, sanitised for Discord."""
    base = ''.join(c for c in (username or 'trader').lower().replace(' ', '-')
                   if c in _NAME_SAFE) or 'trader'
    return '%s_report' % base[:80]


def ensure_channel(username: str, existing_channel_id: str = None) -> Optional[str]:
    """
    Return the user's report channel id, creating it if needed.

    Passing a known id revalidates it — a channel deleted in Discord is
    recreated rather than silently swallowing every future report.
    """
    if not is_configured():
        return None

    if existing_channel_id:
        found = _request('GET', '/channels/%s' % existing_channel_id)
        if found and found.get('id'):
            return found['id']
        logger.info(f"[DISCORD] stored channel {existing_channel_id} is gone — recreating")

    name = channel_name_for(username)

    # Reuse a same-named channel if one already exists in the guild.
    channels = _request('GET', '/guilds/%s/channels' % _guild_id()) or []
    for ch in channels:
        if ch.get('name') == name and ch.get('type') == 0:
            return ch.get('id')

    created = _request('POST', '/guilds/%s/channels' % _guild_id(), {
        'name': name,
        'type': 0,                       # text channel
        'topic': 'Automated GoatBot trading reports for %s' % username,
    })
    if created and created.get('id'):
        logger.info(f"[DISCORD] created #{name} ({created['id']})")
        return created['id']

    logger.error(f"[DISCORD] could not create #{name} — check the bot's Manage Channels permission")
    return None


# ── Report building ────────────────────────────────────────────

def _money(v) -> str:
    v = float(v or 0)
    return ('+${:,.2f}'.format(v)) if v >= 0 else ('-${:,.2f}'.format(abs(v)))


def _pct(v) -> str:
    v = float(v or 0)
    return '%+.2f%%' % v


def summarise(trades: List[Dict]) -> Dict:
    """Aggregate a set of closed trades into report figures."""
    closed = [t for t in trades
              if t.get('action') in ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')]
    pnls = [float(t.get('pnl') or 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]

    gross_win, gross_loss = sum(wins), abs(sum(losses))

    by_strategy: Dict[str, Dict] = {}
    for t in closed:
        # Masked here so a Discord post never names an internal strategy.
        name = public_meta(t.get('strategy'))['name']
        row = by_strategy.setdefault(name, {'trades': 0, 'pnl': 0.0, 'wins': 0})
        row['trades'] += 1
        row['pnl'] += float(t.get('pnl') or 0)
        if float(t.get('pnl') or 0) > 0:
            row['wins'] += 1

    return {
        'total_trades': len(closed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': (len(wins) / len(closed) * 100.0) if closed else 0.0,
        'net_pnl': sum(pnls),
        'gross_win': gross_win,
        'gross_loss': gross_loss,
        'profit_factor': (gross_win / gross_loss) if gross_loss > 0 else (gross_win and 999.0),
        'best': max(pnls) if pnls else 0.0,
        'worst': min(pnls) if pnls else 0.0,
        'avg': (sum(pnls) / len(pnls)) if pnls else 0.0,
        'by_strategy': by_strategy,
    }


def build_embed(username: str, today: Dict, overall: Dict,
                balance: float = None, open_positions: int = 0) -> Dict:
    """Format the report as a Discord embed."""
    day_pnl = today['net_pnl']
    colour = 0x22C55E if day_pnl > 0 else (0xEF4444 if day_pnl < 0 else 0x64748B)

    def block(s: Dict) -> str:
        if not s['total_trades']:
            return '_No closed trades._'
        pf = s['profit_factor']
        pf_txt = ('%.2f' % pf) if pf and pf < 999 else ('∞' if pf else '—')
        return (
            '**Net P&L** %s\n'
            '**Trades** %d  ·  **Win rate** %.1f%% (%dW / %dL)\n'
            '**Profit factor** %s  ·  **Avg** %s\n'
            '**Best** %s  ·  **Worst** %s'
            % (_money(s['net_pnl']), s['total_trades'], s['win_rate'],
               s['wins'], s['losses'], pf_txt, _money(s['avg']),
               _money(s['best']), _money(s['worst']))
        )

    fields = [
        {'name': '📅 Today', 'value': block(today), 'inline': False},
        {'name': '📈 Overall', 'value': block(overall), 'inline': False},
    ]

    if today['by_strategy']:
        lines = []
        for name, row in sorted(today['by_strategy'].items(),
                                key=lambda kv: kv[1]['pnl'], reverse=True):
            wr = (row['wins'] / row['trades'] * 100.0) if row['trades'] else 0.0
            lines.append('`%-22s` %s · %d trades · %.0f%% win'
                         % (name[:22], _money(row['pnl']), row['trades'], wr))
        fields.append({'name': '🤖 By strategy (today)',
                       'value': '\n'.join(lines[:8]), 'inline': False})

    footer_bits = []
    if balance is not None:
        footer_bits.append('Balance ${:,.2f}'.format(float(balance)))
    if open_positions:
        footer_bits.append('%d position%s still open' % (open_positions, '' if open_positions == 1 else 's'))

    return {
        'title': '%s — Daily Trading Report' % username,
        'description': '%s\n_%s UTC_' % (
            ('🟢 Up on the day.' if day_pnl > 0 else
             '🔴 Down on the day.' if day_pnl < 0 else '⚪ Flat on the day.'),
            datetime.now(timezone.utc).strftime('%d %b %Y, %H:%M')),
        'color': colour,
        'fields': fields,
        'footer': {'text': ' · '.join(footer_bits) or 'GoatBot Trade'},
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }


def post_report(channel_id: str, embed: Dict) -> bool:
    """Send one report embed to a channel."""
    if not is_configured() or not channel_id:
        return False
    res = _request('POST', '/channels/%s/messages' % channel_id, {'embeds': [embed]})
    return bool(res)


def post_text(channel_id: str, content: str) -> bool:
    if not is_configured() or not channel_id:
        return False
    return bool(_request('POST', '/channels/%s/messages' % channel_id,
                         {'content': content[:1900]}))

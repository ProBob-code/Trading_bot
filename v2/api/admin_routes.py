"""
Admin Console API
=================

Operator-only views across every account: who is trading, what they are
running, how their bots are performing, and what the evolution engine has
learned for them — plus downloads of any of it, per user or all at once.

Access control
--------------
Every route here is behind `@admin_required`, which is `@login_required` plus a
check of `users.is_admin`. There is no query parameter, header or session value
a normal user can set to reach these; the flag lives in the database and is
re-read from it on each request.

An account becomes an admin either by being listed in the ADMIN_USERNAMES /
ADMIN_MOBILES environment variables (applied at boot) or by an existing admin
granting it. Nothing promotes itself.

Strategy identity
-----------------
These routes report the public catalog code and name (GBX-02 / "Trend Rider").
That is the same identity the user chose in the UI, so it answers "what is this
account running?" without moving the internal strategy ids — the firm's IP —
into another surface that has to be kept from leaking.
"""

import csv
import io
import json
import zipfile
from datetime import datetime, timezone
from functools import wraps

from flask import Blueprint, Response, jsonify, request
from flask_login import current_user, login_required
from loguru import logger

from shared.database.db_manager import db_manager
from shared.logic.strategies.public_catalog import (
    public_meta, to_public as strat_to_public)

admin_bp = Blueprint('admin', __name__)


# ---------------------------------------------------------------------------
# Access control
# ---------------------------------------------------------------------------

def is_admin_user(user) -> bool:
    """True only if the database says so.

    Read through to `users.is_admin` rather than trusting whatever the session
    object was built with, so revoking admin takes effect on the next request
    instead of the next login.
    """
    try:
        if not user or not getattr(user, 'is_authenticated', False):
            return False
        row = db_manager.get_user_by_id(int(user.id))
        return bool(row and int(row.get('is_admin') or 0))
    except Exception as e:
        logger.warning(f"[ADMIN] admin check failed: {e}")
        return False


def admin_required(fn):
    @wraps(fn)
    @login_required
    def wrapper(*args, **kwargs):
        if not is_admin_user(current_user):
            logger.warning(f"[ADMIN] denied non-admin access to {request.path} "
                           f"by user {getattr(current_user, 'id', '?')}")
            return jsonify({'success': False, 'error': 'Administrator access required.'}), 403
        return fn(*args, **kwargs)
    return wrapper


# ---------------------------------------------------------------------------
# Shared shaping helpers
# ---------------------------------------------------------------------------

CLOSING_ACTIONS = ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')


def _label(internal_strategy: str) -> dict:
    """Public code + display name for an internal strategy id."""
    code = strat_to_public(internal_strategy)
    return {'code': code, 'name': public_meta(internal_strategy)['name']}


def _iso(value):
    return value.isoformat() if hasattr(value, 'isoformat') else value


def _summarise(trades):
    """Win/loss/P&L over a list of ledger rows."""
    closed = [t for t in trades if t.get('action') in CLOSING_ACTIONS]
    pnls = [float(t.get('pnl') or 0) for t in closed]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p < 0]
    gross_profit, gross_loss = sum(wins), abs(sum(losses))
    return {
        'events': len(trades),
        'closed_trades': len(closed),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': (len(wins) / len(closed) * 100) if closed else 0.0,
        'total_pnl': sum(pnls),
        'avg_win': (gross_profit / len(wins)) if wins else 0.0,
        'avg_loss': (sum(losses) / len(losses)) if losses else 0.0,
        # No losses at all leaves profit factor undefined; report 0 rather than
        # infinity so the number stays sortable and JSON-safe.
        'profit_factor': (gross_profit / gross_loss) if gross_loss else 0.0,
        'commission': sum(float(t.get('commission') or 0) for t in trades),
    }


def _user_strategy_rows(user_id, usage_rows, evo_by_user):
    """What one account is running, folded into per-strategy rows.

    `usage_rows` is this user's slice, already grouped by the caller — filtering
    the whole platform's rows once per user turned the page into an O(users x
    rows) scan.
    """
    evo = {(e.get('strategy'), e.get('symbol')): e
           for e in evo_by_user.get(user_id, [])}

    rows = []
    for u in usage_rows:
        internal = u.get('strategy')
        state = evo.get((internal, u.get('symbol'))) or evo.get((internal, 'ALL')) or {}
        params = {}
        try:
            params = json.loads(state.get('params_json') or '{}')
        except Exception:
            params = {}
        rows.append({
            'strategy': _label(internal)['code'],
            'strategy_name': _label(internal)['name'],
            'symbol': u.get('symbol'),
            'closed_trades': u.get('closed_trades', 0),
            'wins': u.get('wins', 0),
            'win_rate': round(u.get('win_rate', 0.0), 2),
            'total_pnl': round(u.get('total_pnl', 0.0), 2),
            'last_trade': u.get('last_trade'),
            'generation': state.get('generation', 0),
            'learning_status': state.get('status') or 'active',
            'autopilot': True if state.get('auto_apply') is None
                         else bool(state.get('auto_apply')),
            'baseline_win_rate': state.get('baseline_win_rate'),
            'params': params,
        })
    return rows


def _build_admin_snapshot():
    """One pass over the database that every admin view is derived from."""
    users = db_manager.get_all_users()
    stats = db_manager.admin_user_trade_stats()
    usage = db_manager.admin_strategy_usage()

    evo_by_user = {}
    for e in db_manager.v2_get_all_evolution_states():
        evo_by_user.setdefault(int(e.get('user_id') or 0), []).append(e)

    usage_by_user = {}
    for u in usage:
        usage_by_user.setdefault(u['user_id'], []).append(u)

    custom_by_user = {}
    for c in db_manager.get_all_custom_strategies():
        custom_by_user.setdefault(int(c.get('user_id') or 0), []).append(c)

    rows = []
    for u in users:
        uid = int(u['id'])
        st = stats.get(uid, {})
        rows.append({
            'user_id': uid,
            'username': u.get('username') or f'trader-{uid}',
            'mobile': u.get('mobile'),
            'is_admin': bool(u.get('is_admin')),
            'is_verified': bool(u.get('is_verified')),
            'joined': u.get('created_at'),
            'closed_trades': st.get('closed_trades', 0),
            'total_events': st.get('total_events', 0),
            'wins': st.get('wins', 0),
            'losses': st.get('losses', 0),
            'win_rate': round(st.get('win_rate', 0.0), 2),
            'total_pnl': round(st.get('total_pnl', 0.0), 2),
            'commission': round(st.get('total_commission', 0.0), 2),
            'first_trade': st.get('first_trade'),
            'last_trade': st.get('last_trade'),
            'strategies': _user_strategy_rows(uid, usage_by_user.get(uid, []), evo_by_user),
            'custom_strategies': [
                {'id': c.get('strategy_id'), 'name': c.get('name'),
                 'market': c.get('market'), 'times_used': c.get('times_used', 0),
                 'updated_at': c.get('updated_at')}
                for c in custom_by_user.get(uid, [])
            ],
        })
    return rows, usage, evo_by_user


# ---------------------------------------------------------------------------
# Views
# ---------------------------------------------------------------------------

@admin_bp.route('/api/v2/admin/whoami', methods=['GET'])
@login_required
def admin_whoami():
    """Lets the frontend decide whether to show the Admin entry at all."""
    return jsonify({'success': True, 'is_admin': is_admin_user(current_user),
                    'user_id': getattr(current_user, 'id', None),
                    'username': getattr(current_user, 'username', None)})


@admin_bp.route('/api/v2/admin/overview', methods=['GET'])
@admin_required
def admin_overview():
    """Every account with its headline numbers and the strategies it runs."""
    rows, _usage, _evo = _build_admin_snapshot()

    totals = {
        'users': len(rows),
        'active_users': sum(1 for r in rows if r['closed_trades'] > 0),
        'closed_trades': sum(r['closed_trades'] for r in rows),
        'total_pnl': round(sum(r['total_pnl'] for r in rows), 2),
        'profitable_users': sum(1 for r in rows if r['total_pnl'] > 0),
    }
    closed = totals['closed_trades']
    totals['win_rate'] = round(
        (sum(r['wins'] for r in rows) / closed * 100) if closed else 0.0, 2)

    return jsonify({'success': True, 'generated_at': datetime.now(timezone.utc).isoformat(),
                    'totals': totals, 'users': rows})


@admin_bp.route('/api/v2/admin/strategies', methods=['GET'])
@admin_required
def admin_strategies():
    """Strategy adoption across the whole platform, and who is running each."""
    rows, usage, evo_by_user = _build_admin_snapshot()
    names = {r['user_id']: r['username'] for r in rows}

    by_strategy = {}
    for u in usage:
        internal = u.get('strategy')
        label = _label(internal)
        entry = by_strategy.setdefault(label['code'], {
            'strategy': label['code'], 'strategy_name': label['name'],
            'users': [], 'symbols': set(), 'closed_trades': 0, 'wins': 0,
            'total_pnl': 0.0,
        })
        uid = u['user_id']
        if uid not in [x['user_id'] for x in entry['users']]:
            entry['users'].append({'user_id': uid, 'username': names.get(uid, f'trader-{uid}')})
        if u.get('symbol'):
            entry['symbols'].add(u['symbol'])
        entry['closed_trades'] += u.get('closed_trades', 0)
        entry['wins'] += u.get('wins', 0)
        entry['total_pnl'] += u.get('total_pnl', 0.0)

    out = []
    for entry in by_strategy.values():
        closed = entry['closed_trades']
        out.append({**entry,
                    'symbols': sorted(entry['symbols']),
                    'user_count': len(entry['users']),
                    'win_rate': round((entry['wins'] / closed * 100) if closed else 0.0, 2),
                    'total_pnl': round(entry['total_pnl'], 2)})
    out.sort(key=lambda e: (-e['user_count'], -e['closed_trades']))

    # Custom, user-authored strategies are separate from the catalog ones.
    custom = []
    for r in rows:
        for c in r['custom_strategies']:
            custom.append({**c, 'user_id': r['user_id'], 'username': r['username']})

    return jsonify({'success': True, 'catalog': out, 'custom': custom})


@admin_bp.route('/api/v2/admin/users/<int:user_id>/report', methods=['GET'])
@admin_required
def admin_user_report(user_id):
    """The full report for one account — the same figures the user sees."""
    from v2.engine.evolution.evolution_engine import evolution_engine

    user = db_manager.get_user_by_id(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'No such user.'}), 404

    limit = min(int(request.args.get('limit', 1000)), 5000)
    trades = db_manager.v2_get_user_trades(user_id=user_id, limit=limit) or []

    by_strategy = {}
    for t in trades:
        by_strategy.setdefault((t.get('strategy'), t.get('symbol')), []).append(t)

    strategies = []
    for (internal, symbol), rows in sorted(
            by_strategy.items(), key=lambda kv: (str(kv[0][0]), str(kv[0][1]))):
        if not internal:
            continue
        state = db_manager.v2_get_evolution_state(user_id, internal, symbol) or {}
        label = _label(internal)
        strategies.append({
            'strategy': label['code'], 'strategy_name': label['name'], 'symbol': symbol,
            'summary': _summarise(rows),
            'generation': state.get('generation', 0),
            'learning_status': state.get('status') or 'active',
            'autopilot': True if state.get('auto_apply') is None
                         else bool(state.get('auto_apply')),
            'journey': evolution_engine.win_rate_journey(user_id, internal, symbol, state),
        })

    return jsonify({
        'success': True,
        'user': {'user_id': user_id,
                 'username': user.get('username') or f'trader-{user_id}',
                 'mobile': user.get('mobile'),
                 'joined': _iso(user.get('created_at'))},
        'summary': _summarise(trades),
        'strategies': strategies,
        'custom_strategies': db_manager.get_user_strategies(user_id),
        'trades': [{
            'trade_id': t.get('trade_id'), 'timestamp': _iso(t.get('timestamp')),
            'symbol': t.get('symbol'), 'side': t.get('side'), 'action': t.get('action'),
            'quantity': float(t.get('quantity') or 0), 'price': float(t.get('price') or 0),
            'pnl': float(t.get('pnl') or 0), 'commission': float(t.get('commission') or 0),
            'strategy': _label(t.get('strategy'))['code'],
            'strategy_name': _label(t.get('strategy'))['name'],
            'session_id': t.get('session_id'),
        } for t in trades],
    })


@admin_bp.route('/api/v2/admin/users/<int:user_id>/admin', methods=['POST'])
@admin_required
def admin_set_admin(user_id):
    """Grant or revoke admin on another account.

    The last admin cannot be demoted — an install with no administrator has no
    way back into this console.
    """
    data = request.json or {}
    grant = bool(data.get('is_admin'))

    if not grant and db_manager.count_admins() <= 1:
        return jsonify({'success': False,
                        'error': 'This is the only administrator — promote '
                                 'someone else before revoking it.'}), 400

    if not db_manager.get_user_by_id(user_id):
        return jsonify({'success': False, 'error': 'No such user.'}), 404

    ok = db_manager.set_user_admin(user_id, grant)
    if ok:
        logger.info(f"[ADMIN] user {user_id} admin={grant} "
                    f"(by {getattr(current_user, 'id', '?')})")
    return jsonify({'success': ok, 'user_id': user_id, 'is_admin': grant})


# ---------------------------------------------------------------------------
# Downloads
# ---------------------------------------------------------------------------

TRADE_COLUMNS = ['user_id', 'username', 'timestamp', 'session_id', 'symbol',
                 'strategy', 'strategy_name', 'side', 'action', 'quantity',
                 'price', 'pnl', 'commission']

SUMMARY_COLUMNS = ['user_id', 'username', 'mobile', 'joined', 'closed_trades',
                   'wins', 'losses', 'win_rate', 'total_pnl', 'commission',
                   'first_trade', 'last_trade']

STRATEGY_COLUMNS = ['user_id', 'username', 'strategy', 'strategy_name', 'symbol',
                    'closed_trades', 'wins', 'win_rate', 'total_pnl',
                    'generation', 'learning_status', 'autopilot',
                    'baseline_win_rate', 'last_trade']


def _trade_rows(user_id, username, limit=20000):
    for t in db_manager.v2_get_user_trades(user_id=user_id, limit=limit) or []:
        label = _label(t.get('strategy'))
        yield {
            'user_id': user_id, 'username': username,
            'timestamp': _iso(t.get('timestamp')), 'session_id': t.get('session_id'),
            'symbol': t.get('symbol'), 'strategy': label['code'],
            'strategy_name': label['name'], 'side': t.get('side'),
            'action': t.get('action'), 'quantity': float(t.get('quantity') or 0),
            'price': float(t.get('price') or 0), 'pnl': float(t.get('pnl') or 0),
            'commission': float(t.get('commission') or 0),
        }


def _csv_bytes(columns, rows) -> bytes:
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=columns, extrasaction='ignore')
    writer.writeheader()
    for r in rows:
        writer.writerow(r)
    return buf.getvalue().encode('utf-8')


def _stamp():
    return datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')


def _safe_name(name: str) -> str:
    """A filename fragment that cannot escape the archive or break a header."""
    keep = [c if (c.isalnum() or c in '-_') else '-' for c in str(name or '')]
    return (''.join(keep).strip('-') or 'user')[:48]


@admin_bp.route('/api/v2/admin/export/user/<int:user_id>', methods=['GET'])
@admin_required
def admin_export_user(user_id):
    """Download one account's report as CSV or JSON."""
    fmt = (request.args.get('format') or 'csv').lower()
    user = db_manager.get_user_by_id(user_id)
    if not user:
        return jsonify({'success': False, 'error': 'No such user.'}), 404

    username = user.get('username') or f'trader-{user_id}'
    rows = list(_trade_rows(user_id, username))
    base = f"goatbot-{_safe_name(username)}-{_stamp()}"

    if fmt == 'json':
        payload = {'user': {'user_id': user_id, 'username': username},
                   'generated_at': datetime.now(timezone.utc).isoformat(),
                   'summary': _summarise(rows), 'trades': rows}
        return Response(
            json.dumps(payload, indent=2, default=str),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="{base}.json"'})

    return Response(
        _csv_bytes(TRADE_COLUMNS, rows), mimetype='text/csv',
        headers={'Content-Disposition': f'attachment; filename="{base}.csv"'})


@admin_bp.route('/api/v2/admin/export/all', methods=['GET'])
@admin_required
def admin_export_all():
    """Download every account at once.

    `zip`  — a summary CSV, a strategy-usage CSV, and one CSV per user
    `csv`  — every trade from every account in a single flat file
    `json` — the full structured snapshot
    """
    fmt = (request.args.get('format') or 'zip').lower()
    rows, _usage, _evo = _build_admin_snapshot()
    base = f"goatbot-all-users-{_stamp()}"

    strategy_rows = [
        {**s, 'user_id': r['user_id'], 'username': r['username']}
        for r in rows for s in r['strategies']
    ]

    if fmt == 'json':
        payload = {'generated_at': datetime.now(timezone.utc).isoformat(),
                   'users': rows,
                   'trades': [t for r in rows
                              for t in _trade_rows(r['user_id'], r['username'])]}
        return Response(
            json.dumps(payload, indent=2, default=str),
            mimetype='application/json',
            headers={'Content-Disposition': f'attachment; filename="{base}.json"'})

    if fmt == 'csv':
        flat = [t for r in rows for t in _trade_rows(r['user_id'], r['username'])]
        return Response(
            _csv_bytes(TRADE_COLUMNS, flat), mimetype='text/csv',
            headers={'Content-Disposition': f'attachment; filename="{base}.csv"'})

    # Default: a zip so per-user files stay separable after download.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('summary.csv', _csv_bytes(SUMMARY_COLUMNS, rows))
        zf.writestr('strategies.csv', _csv_bytes(STRATEGY_COLUMNS, strategy_rows))
        for r in rows:
            trades = list(_trade_rows(r['user_id'], r['username']))
            if not trades:
                continue
            name = f"users/{r['user_id']}-{_safe_name(r['username'])}.csv"
            zf.writestr(name, _csv_bytes(TRADE_COLUMNS, trades))
    buf.seek(0)

    return Response(
        buf.getvalue(), mimetype='application/zip',
        headers={'Content-Disposition': f'attachment; filename="{base}.zip"'})

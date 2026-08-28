"""
V2 Evolution Engine — self-tuning strategy parameters
=====================================================

Learns from CLOSED paper trades and adjusts each strategy's parameters so the
next generation has better expectancy than the last.

Design principles (deliberate, not black-box):

1. **Interpretable.** Every change carries a plain-English reason the user can
   read. No opaque weights — you can always answer "why did it do that?".
2. **Evidence-gated.** Nothing changes until there are enough closed trades
   (MIN_TRADES). Small samples are noise, and over-fitting noise is how these
   systems destroy themselves.
3. **Bounded.** Every parameter has a hard floor/ceiling, and each step is a
   small nudge, never a jump.
4. **Reversible.** The best-performing parameter set is remembered. If a
   generation makes things worse, it rolls back instead of drifting.
5. **Never off.** Learning has no terminal state. A strategy that keeps losing
   is *de-risked* — tighter stop, more conviction demanded, a cool-down between
   entries — but it keeps trading, keeps closing trades, and therefore keeps
   learning. Pausing a losing strategy used to be the fail-safe; it was also a
   dead end, because a bot that never trades never produces the evidence it
   needs to recover.

Loss comes first. Every rule below is ordered so that reducing the damage of
losing trades outranks chasing a larger win, and no rule may widen a stop while
the strategy is losing money — that is the one change that makes a bad run
strictly worse.

What it detects (the failure modes seen in real ledger data):
  • take-profit never hit  → target is unreachable for this instrument
  • stop-loss hit constantly → stop is inside normal noise
  • exits dominated by signal flips → whipsaw / over-trading
  • a run of consecutive losses → the regime turned against the strategy
  • losers far larger than winners → payoff geometry is upside-down
"""

import json
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from shared.database.db_manager import db_manager


# ── Tunable parameter space (hard bounds — evolution can never escape these) ──
PARAM_BOUNDS = {
    'take_profit': (0.25, 10.0),   # %
    'stop_loss':   (0.15, 5.0),    # %
    'sensitivity_rank': (0, 2),    # 0 conservative, 1 balanced, 2 aggressive
    'cooldown_bars': (0, 30),      # bars to wait after an exit before re-entry
}

SENSITIVITY_BY_RANK = ['conservative', 'balanced', 'aggressive']

DEFAULT_PARAMS = {
    'take_profit': 6.0,
    'stop_loss': 2.0,
    'sensitivity_rank': 0,
    'cooldown_bars': 0,
}

# Evidence gates
MIN_TRADES = 8           # never adapt on fewer closed trades than this
EVAL_EVERY = 5           # re-evaluate after this many new closed trades
PATIENCE_GENERATIONS = 3  # unprofitable generations before hard de-risking
LOSS_STREAK_TRIGGER = 3   # consecutive losers that trigger a protective step

# Win-rate journey windows
BASELINE_WINDOW = 10     # trades that define "where it started"
RECENT_WINDOW = 20       # trades that define "where it is now"
CLOSING_ACTIONS = ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')


def _clamp(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


def _ts_key(trade: Dict):
    """Sortable timestamp for a ledger row, tolerant of str/datetime/None."""
    ts = trade.get('timestamp') or trade.get('trade_time')
    if ts is None:
        return ''
    return ts.isoformat() if hasattr(ts, 'isoformat') else str(ts)


class EvolutionEngine:
    """Analyses closed trades and evolves per-strategy parameters."""

    def __init__(self, db=None):
        self.db = db or db_manager

    # ── Metrics ──────────────────────────────────────────────

    def compute_metrics(self, trades: List[Dict]) -> Dict:
        """Turn raw closing events into the signals evolution reacts to."""
        closed = [t for t in trades if t.get('action') in CLOSING_ACTIONS]
        n = len(closed)
        if n == 0:
            return {'closed_trades': 0, 'win_rate': 0.0, 'expectancy': 0.0,
                    'total_pnl': 0.0, 'tp_hit_rate': 0.0, 'sl_hit_rate': 0.0,
                    'reversal_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0,
                    'gross_profit': 0.0, 'gross_loss': 0.0, 'profit_factor': 0.0,
                    'worst_loss': 0.0, 'loss_streak': 0, 'max_loss_streak': 0}

        # Streaks are meaningless out of order, and the eval query hands back
        # newest-first. Sort before anything sequential is measured.
        chronological = sorted(closed, key=_ts_key)

        pnls = [float(t.get('pnl') or 0) for t in chronological]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        actions = [t.get('action') for t in chronological]

        streak = max_streak = 0
        for p in pnls:
            streak = streak + 1 if p < 0 else 0
            max_streak = max(max_streak, streak)

        gross_profit = sum(wins)
        gross_loss = abs(sum(losses))

        return {
            'closed_trades': n,
            'win_rate': len(wins) / n * 100,
            'expectancy': sum(pnls) / n,
            'total_pnl': sum(pnls),
            'tp_hit_rate': actions.count('TAKE_PROFIT') / n * 100,
            'sl_hit_rate': actions.count('STOP_LOSS') / n * 100,
            'reversal_rate': (actions.count('REVERSAL') + actions.count('CLOSE')) / n * 100,
            'avg_win': gross_profit / len(wins) if wins else 0.0,
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
            'gross_profit': gross_profit,
            'gross_loss': gross_loss,
            'profit_factor': (gross_profit / gross_loss) if gross_loss > 0 else
                             (float('inf') if gross_profit > 0 else 0.0),
            'worst_loss': min(pnls) if pnls else 0.0,
            'loss_streak': streak,          # losers running right now
            'max_loss_streak': max_streak,
        }

    # ── Win-rate journey ─────────────────────────────────────

    def win_rate_journey(self, user_id: int, strategy: str, symbol: str = 'ALL',
                         state: Optional[Dict] = None) -> Dict:
        """Where the win rate started, where it is now, and the path between.

        This is the trust story: a bot that has learned from its mistakes should
        be able to show it, not assert it. Everything here is derived from the
        ledger, so it cannot drift out of step with the trades themselves.

        `from_pct`  — win rate over the first BASELINE_WINDOW closed trades
        `to_pct`    — win rate over the most recent RECENT_WINDOW closed trades
        `overall`   — win rate across every closed trade
        `points`    — a rolling win-rate curve for the sparkline
        `generations` — win rate recorded at each applied generation
        """
        series = self.db.v2_get_closed_pnl_series(user_id, strategy, symbol)
        pnls = [float(t.get('pnl') or 0) for t in series]
        n = len(pnls)

        def rate(chunk):
            return (sum(1 for p in chunk if p > 0) / len(chunk) * 100) if chunk else 0.0

        if n == 0:
            return {'available': False, 'closed_trades': 0, 'from_pct': 0.0,
                    'to_pct': 0.0, 'overall': 0.0, 'delta': 0.0, 'best': 0.0,
                    'points': [], 'generations': [], 'from_trades': 0, 'to_trades': 0,
                    'pnl_from': 0.0, 'pnl_to': 0.0}

        # With few trades the two windows would overlap and the journey would
        # compare a sample against itself. Split down the middle instead.
        window = min(BASELINE_WINDOW, max(1, n // 2))
        recent = min(RECENT_WINDOW, max(1, n // 2)) if n >= 2 else n

        first_chunk, last_chunk = pnls[:window], pnls[-recent:]
        from_pct, to_pct = rate(first_chunk), rate(last_chunk)

        # Rolling curve — window grows with the sample so early points are not
        # pure noise, capped so the line stays responsive later on.
        roll = max(5, min(RECENT_WINDOW, n // 4 or 1))
        points, best = [], 0.0
        step = max(1, n // 40)          # ≤ ~40 points keeps the sparkline light
        for i in range(roll, n + 1, step):
            r = rate(pnls[max(0, i - roll):i])
            points.append(round(r, 2))
            best = max(best, r)
        if not points:
            points = [round(rate(pnls), 2)]
            best = points[0]

        gens = []
        try:
            for h in self.db.v2_get_evolution_history(user_id, strategy, limit=100):
                if h.get('symbol') and symbol and symbol != 'ALL' and h['symbol'] != symbol:
                    continue
                gens.append({
                    'generation': h.get('generation'),
                    'win_rate': round(float(h.get('win_rate') or 0), 2),
                    'closed_trades': h.get('closed_trades'),
                    'verdict': h.get('verdict'),
                    'at': h.get('created_at'),
                })
            gens.sort(key=lambda g: (g['generation'] or 0))
        except Exception as e:
            logger.debug(f"[V2-EVO] journey history unavailable: {e}")

        # A persisted origin survives ledger retention trimming old rows.
        pinned = (state or {}).get('baseline_win_rate')
        if pinned is not None and n >= MIN_TRADES:
            from_pct = float(pinned)

        return {
            'available': n >= max(4, window + 1),
            'closed_trades': n,
            'from_pct': round(from_pct, 2),
            'to_pct': round(to_pct, 2),
            'overall': round(rate(pnls), 2),
            'delta': round(to_pct - from_pct, 2),
            'best': round(max(best, to_pct), 2),
            'from_trades': len(first_chunk),
            'to_trades': len(last_chunk),
            'pnl_from': round(sum(first_chunk), 2),
            'pnl_to': round(sum(last_chunk), 2),
            'points': points,
            'generations': gens,
        }

    # ── Diagnosis → parameter changes ────────────────────────

    def propose_changes(self, params: Dict, m: Dict) -> List[Dict]:
        """Diagnose what went wrong and propose bounded, explained adjustments.

        Ordered by priority: stop the bleeding, then fix the payoff geometry,
        then chase a better win. A stop is never widened while the strategy is
        losing — widening under a losing run is the one adjustment that makes
        the drawdown strictly deeper.
        """
        changes: List[Dict] = []
        n = m['closed_trades']
        if n < MIN_TRADES:
            return []

        losing = m['expectancy'] < 0

        def change(param, new_val, reason):
            old = params[param]
            # Round on assignment — otherwise repeated multiplicative nudges
            # accumulate float noise (6.0 * 0.6 -> 3.5999999999999996).
            new_val = round(_clamp(param, new_val), 3)
            if abs(new_val - old) < 1e-9:
                return
            changes.append({'param': param, 'from': round(old, 3),
                            'to': new_val, 'reason': reason})
            params[param] = new_val

        # 1. A run of losers right now → the regime has turned. Shrink the
        #    damage before anything else, and slow the bot down.
        if m['loss_streak'] >= LOSS_STREAK_TRIGGER:
            change('stop_loss', params['stop_loss'] * 0.8,
                   f"{m['loss_streak']} losing trades in a row — cutting the stop "
                   f"distance so the current run costs less per attempt.")
            change('sensitivity_rank', params['sensitivity_rank'] - 1,
                   f"{m['loss_streak']} consecutive losses — demanding stronger "
                   f"confirmation before taking the next entry.")
            change('cooldown_bars', max(params['cooldown_bars'], 3) + 2,
                   "Waiting longer after each exit so a bad stretch is not "
                   "compounded by immediately re-entering into it.")

        # 2. Losers much larger than winners → payoff geometry is upside-down.
        if m['avg_win'] > 0 and m['avg_loss'] < 0:
            if abs(m['avg_loss']) > 2.5 * m['avg_win']:
                change('stop_loss', params['stop_loss'] * 0.8,
                       f"Average loss (${abs(m['avg_loss']):.2f}) is more than 2.5x the "
                       f"average win (${m['avg_win']:.2f}). Tightening the stop to cut "
                       f"the tail off losing trades.")

        # 3. Take-profit never reached → the target is unreachable for this market.
        #    Pull it toward the size of moves actually being captured, but never
        #    below the stop: a target tighter than the stop inverts the payoff,
        #    so the bot would need to win most trades just to break even.
        if m['tp_hit_rate'] == 0:
            floor = params['stop_loss']
            proposed = max(params['take_profit'] * 0.6, floor)
            if proposed < params['take_profit'] - 1e-9:
                change('take_profit', proposed,
                       f"Take-profit never hit in {n} closed trades — target unreachable "
                       f"for this instrument's volatility. Lowering it so winners can "
                       f"actually be banked, while keeping it at least as wide as the "
                       f"stop so a win still pays for a loss.")

        # 4. Stop-loss firing constantly. The fix depends on whether the strategy
        #    is otherwise working:
        #      • profitable → the stop really is inside the noise; widen it.
        #      • losing     → widening would only enlarge each loss. Take fewer,
        #                     better trades instead.
        if m['sl_hit_rate'] > 40:
            if not losing:
                change('stop_loss', params['stop_loss'] * 1.25,
                       f"Stop-loss hit on {m['sl_hit_rate']:.0f}% of exits while the "
                       f"strategy is still profitable — the stop sits inside normal "
                       f"price noise. Widening it so good ideas get room to work.")
            else:
                change('sensitivity_rank', params['sensitivity_rank'] - 1,
                       f"Stop-loss hit on {m['sl_hit_rate']:.0f}% of exits and the run is "
                       f"unprofitable. Widening the stop would just enlarge each loss, "
                       f"so the bar for entering is being raised instead.")
                change('cooldown_bars', max(params['cooldown_bars'], 2) + 2,
                       "Pausing briefly after each stop-out rather than re-entering "
                       "straight back into the move that just stopped us.")

        # 5. Exits dominated by signal flips AND losing → whipsaw / over-trading.
        if m['reversal_rate'] > 60 and losing:
            change('sensitivity_rank', params['sensitivity_rank'] - 1,
                   f"{m['reversal_rate']:.0f}% of exits came from the signal flipping, "
                   f"and expectancy is negative — the strategy is being whipsawed in a "
                   f"ranging market. Demanding stronger conviction before entering.")
            change('cooldown_bars', max(params['cooldown_bars'], 3) + 2,
                   "Adding a cool-down after each exit so one choppy stretch can't "
                   "trigger a chain of reversals (each one pays commission twice).")

        # 6. Working, and banking winners easily → give the winners slightly more
        #    room. Small step, and only while the payoff geometry is healthy.
        if (not losing and not changes and m['tp_hit_rate'] > 55
                and m['profit_factor'] > 1.3 and m['win_rate'] > 55):
            change('take_profit', params['take_profit'] * 1.15,
                   f"Take-profit is being reached on {m['tp_hit_rate']:.0f}% of exits with "
                   f"a profit factor of {m['profit_factor']:.2f} — the target is being hit "
                   f"comfortably, so letting winners run a little further.")

        # 7. Working well → stop fiddling. Over-tuning a winner is how edges die.
        return changes

    def protective_changes(self, params: Dict, m: Dict, generation: int) -> List[Dict]:
        """Last-resort de-risking for a strategy that keeps losing.

        This replaces what used to be an automatic pause. Pausing stopped the
        losses but also stopped the trades, and with no trades there is no
        evidence, so the strategy could never learn its way back out. Here it
        keeps trading at the smallest risk the bounds allow.
        """
        changes: List[Dict] = []

        def change(param, new_val, reason):
            old = params[param]
            new_val = round(_clamp(param, new_val), 3)
            if abs(new_val - old) < 1e-9:
                return
            changes.append({'param': param, 'from': round(old, 3),
                            'to': new_val, 'reason': reason})
            params[param] = new_val

        change('stop_loss', params['stop_loss'] * 0.75,
               f"Still unprofitable after {generation} generations "
               f"(total ${m['total_pnl']:.2f}). Cutting the stop hard so each further "
               f"attempt risks as little as possible while it recovers.")
        change('sensitivity_rank', 0,
               "Dropping to the most conservative entry filter — far fewer trades, "
               "only the clearest setups, until expectancy turns positive again.")
        change('cooldown_bars', max(params['cooldown_bars'], 4) + 2,
               "Extending the wait between trades so the strategy stops paying "
               "commission on a setup that is not currently working.")
        return changes

    # ── Market regime ────────────────────────────────────────

    def detect_regime(self, symbol: str = 'BTCUSDT', interval: str = '15m') -> str:
        """Classify the current market so each strategy can carry per-regime memory.

        Uses the same statistical evidence as the v3 quant engine (Hurst
        exponent + variance ratio) rather than eyeballing price direction.
        Returns: trending | ranging | volatile | unknown
        """
        try:
            from shared.providers.crypto_provider import BinanceCryptoProvider
            from shared.logic.strategies.v3_quant_strategies import regime_evidence, atr

            df = BinanceCryptoProvider().get_historical_klines(
                symbol=symbol, interval=interval, limit=200)
            if df is None or len(df) < 60:
                return 'unknown'

            ev = regime_evidence(df['close'])
            atr_pct = float(atr(df, 14).iloc[-1]) / float(df['close'].iloc[-1]) * 100

            # High volatility dominates the classification — position sizing and
            # stop distance matter more than direction when the tape is wild.
            if atr_pct > 1.0:
                return 'volatile'
            if ev.get('hurst', 0.5) > 0.55 or ev.get('variance_ratio', 1.0) > 1.15:
                return 'trending'
            return 'ranging'
        except Exception as e:
            logger.debug(f"[V2-EVO] regime detection unavailable: {e}")
            return 'unknown'

    # ── Readiness meter ──────────────────────────────────────

    def readiness(self, user_id: int, strategy: str, symbol: str = 'ALL') -> Dict:
        """How full the 'evolve meter' is — drives the UI progress bar."""
        trades = self.db.v2_get_closed_trades_for_eval(user_id, strategy, symbol)
        m = self.compute_metrics(trades)
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol) or {}
        seen = int(state.get('trades_at_last_eval') or 0)
        gen = int(state.get('generation') or 0)

        if gen == 0:
            need, have = MIN_TRADES, m['closed_trades']
        else:
            need, have = EVAL_EVERY, max(0, m['closed_trades'] - seen)

        pct = 0 if need <= 0 else min(100, have / need * 100)
        return {
            'progress': round(pct, 1),
            'have': have,
            'need': need,
            'ready': have >= need,
            'closed_trades': m['closed_trades'],
        }

    # ── Evolution step ───────────────────────────────────────

    def evolve(self, user_id: int, strategy: str, symbol: str = 'ALL',
               force: bool = False) -> Optional[Dict]:
        """Evaluate a strategy and act on what it finds.

        With autopilot on (the default) a lesson is applied the moment the
        evidence supports it, so a bot keeps improving while nobody is watching
        the dashboard. With autopilot off the lesson is stored as a pending
        proposal for the user to review and approve.

        Either way this call always *observes*: metrics, the readiness meter and
        the win-rate journey are refreshed on every close, including while a
        pair is held. Learning has no off switch.
        """
        trades = self.db.v2_get_closed_trades_for_eval(user_id, strategy, symbol)
        m = self.compute_metrics(trades)

        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if state:
            params = json.loads(state.get('params_json') or '{}') or dict(DEFAULT_PARAMS)
            generation = int(state.get('generation') or 0)
            best_params = json.loads(state.get('best_params_json') or 'null') or dict(params)
            best_exp = state.get('best_expectancy')
            seen = int(state.get('trades_at_last_eval') or 0)
            status = state.get('status') or 'active'
            profiles = json.loads(state.get('profiles_json') or '{}')
            prev_regime = state.get('regime')
            auto_apply = state.get('auto_apply')
            auto_apply = True if auto_apply is None else bool(auto_apply)
        else:
            params, generation, best_params = dict(DEFAULT_PARAMS), 0, dict(DEFAULT_PARAMS)
            best_exp, seen, status = None, 0, 'active'
            profiles, prev_regime, auto_apply = {}, None, True

        meter = self.readiness(user_id, strategy, symbol)
        # Regime is measured on the instrument actually being traded.
        regime = self.detect_regime(symbol if symbol and symbol != 'ALL' else 'BTCUSDT')

        # Pin the origin of the win-rate journey the first time there is enough
        # evidence for it to mean something. It never moves again.
        baseline_win_rate = state.get('baseline_win_rate') if state else None
        baseline_trades = int((state or {}).get('baseline_trades') or 0)
        baseline_at = (state or {}).get('baseline_at')
        new_baseline = None
        if baseline_win_rate is None and m['closed_trades'] >= MIN_TRADES:
            origin = self.win_rate_journey(user_id, strategy, symbol)
            baseline_win_rate = new_baseline = origin['from_pct']
            baseline_trades = origin['from_trades']
            baseline_at = datetime.now(timezone.utc)
            logger.info(f"🧬 [V2-EVO] {strategy}/{symbol} journey origin pinned at "
                        f"{baseline_win_rate:.1f}% over {baseline_trades} trades")

        journey = self.win_rate_journey(user_id, strategy, symbol, state)

        def persist(pending_json, gen=None, prm=None, st=None, seen_count=None):
            self.db.v2_save_evolution_state(
                user_id, strategy,
                generation if gen is None else gen,
                json.dumps(prm if prm is not None else params),
                json.dumps(best_params), best_exp,
                seen if seen_count is None else seen_count,
                status if st is None else st,
                pending_json, regime, json.dumps(profiles), symbol,
                auto_apply=auto_apply,
                baseline_win_rate=baseline_win_rate,
                baseline_trades=baseline_trades,
                baseline_at=baseline_at)

        # The user put this pair on hold. Changes are withheld — but the meter,
        # the metrics and the journey keep updating, so nothing is lost and
        # resuming picks up exactly where it left off.
        # Note: no `force` escape here. "Evaluate all" must not quietly override
        # a hold the user set on one pair.
        if status == 'paused':
            persist(state.get('pending_json') if state else None)
            return {'skipped': True, 'paused': True, 'observing': True,
                    'strategy': strategy, 'symbol': symbol, 'meter': meter,
                    'regime': regime, 'journey': journey, 'auto_apply': auto_apply,
                    'reason': 'Changes are on hold for this pair. It is still '
                              'watching every closed trade — resume to let it act again.',
                    'metrics': m, 'params': params, 'generation': generation}

        # Meter not full yet → keep gathering evidence, nothing to review.
        if not meter['ready'] and not force:
            persist(state.get('pending_json') if state else None)
            return {'skipped': True, 'strategy': strategy, 'symbol': symbol,
                    'meter': meter, 'regime': regime, 'journey': journey,
                    'auto_apply': auto_apply,
                    'reason': f"Gathering evidence — {meter['have']}/{meter['need']} "
                              f"closed trades until the next lesson is ready.",
                    'metrics': m, 'params': params, 'generation': generation}

        # ── Build the proposal ──
        candidate = dict(params)
        verdict = 'adapted'

        if best_exp is None or m['expectancy'] > best_exp:
            best_exp, best_params = m['expectancy'], dict(params)

        changes = self.propose_changes(candidate, m)

        # Market flipped and we have memory of this regime → recall it.
        if regime != 'unknown' and prev_regime and regime != prev_regime and regime in profiles:
            remembered = profiles[regime]
            if remembered != params:
                candidate = dict(remembered)
                changes = [{'param': 'ALL', 'from': 'current', 'to': f'{regime} profile',
                            'reason': f"Market regime changed from {prev_regime} to {regime}. "
                                      f"Recalling the parameters this strategy already "
                                      f"learned for {regime} conditions."}]
                verdict = 'regime_switch'

        # Drifted worse than the best-known set → roll back.
        if not changes and best_exp is not None and m['expectancy'] < best_exp and params != best_params:
            candidate = dict(best_params)
            changes = [{'param': 'ALL', 'from': 'current', 'to': 'best-known',
                        'reason': f"Recent expectancy (${m['expectancy']:.2f}) is below the best "
                                  f"set seen (${best_exp:.2f}). Rolling back to what worked."}]
            verdict = 'revert'

        # Persistently negative → de-risk hard, but keep trading and keep
        # learning. This is where the engine used to switch itself off.
        if m['expectancy'] < 0 and generation >= PATIENCE_GENERATIONS and not changes:
            changes = self.protective_changes(candidate, m, generation)
            verdict = 'protect' if changes else 'adapted'

        pending = None
        if changes:
            pending = {
                'proposed_at': datetime.now(timezone.utc).isoformat(),
                'regime': regime,
                'verdict': verdict,
                'changes': changes,
                'params_before': params,
                'params_after': candidate,
                'metrics': {k: (round(v, 4) if isinstance(v, float) and v not in (float('inf'), float('-inf')) else v)
                            for k, v in m.items()},
                'next_generation': generation + 1,
            }

        persist(json.dumps(pending) if pending else None, seen_count=m['closed_trades'])

        result = {'strategy': strategy, 'symbol': symbol, 'generation': generation,
                  'meter': meter, 'regime': regime, 'metrics': m, 'params': params,
                  'journey': journey, 'auto_apply': auto_apply,
                  'pending': pending, 'changes': changes, 'verdict': verdict,
                  'awaiting_approval': bool(pending) and not auto_apply}

        if pending and auto_apply:
            applied = self.approve(user_id, strategy, symbol, automatic=True)
            if applied.get('success'):
                result.update({
                    'auto_applied': True,
                    'awaiting_approval': False,
                    'pending': None,
                    'generation': applied['generation'],
                    'params': applied['params'],
                    'journey': self.win_rate_journey(user_id, strategy, symbol, state),
                })
                logger.info(f"🧬 [V2-EVO] {strategy}/{symbol} autopilot applied gen "
                            f"{applied['generation']} ({verdict}, {len(changes)} change(s))")
            else:
                result['auto_apply_error'] = applied.get('error')
        elif pending:
            logger.info(f"🧬 [V2-EVO] {strategy}: lesson ready ({verdict}, {len(changes)} change(s)) "
                        f"— awaiting user approval")

        return result

    def approve(self, user_id: int, strategy: str, symbol: str = 'ALL',
                automatic: bool = False) -> Dict:
        """Apply the pending proposal — the Proceed button, or autopilot.

        Commits the proposed params, banks them into the per-regime profile so
        the strategy remembers what worked in this market, bumps the generation,
        and writes the history row.
        """
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if not state or not state.get('pending_json'):
            return {'success': False, 'error': 'No pending lesson to apply.'}

        pending = json.loads(state['pending_json'])
        params_before = json.loads(state.get('params_json') or '{}') or dict(DEFAULT_PARAMS)
        new_params = pending.get('params_after') or params_before
        generation = int(state.get('generation') or 0) + 1
        # Applying a lesson never switches a strategy off. Legacy rows may still
        # carry the retired 'pause' verdict; they are applied as de-risking.
        status = state.get('status') or 'active'
        regime = pending.get('regime') or state.get('regime') or 'unknown'
        profiles = json.loads(state.get('profiles_json') or '{}')

        # Remember what we settled on for this regime.
        if regime != 'unknown':
            profiles[regime] = dict(new_params)

        best_params = json.loads(state.get('best_params_json') or 'null') or dict(new_params)
        best_exp = state.get('best_expectancy')

        self.db.v2_save_evolution_state(
            user_id, strategy, generation, json.dumps(new_params), json.dumps(best_params),
            best_exp, int(state.get('trades_at_last_eval') or 0), status,
            None, regime, json.dumps(profiles), symbol)   # pending cleared

        met = pending.get('metrics', {})
        verdict = pending.get('verdict', 'adapted')
        self.db.v2_add_evolution_generation({
            'user_id': user_id, 'strategy': strategy, 'symbol': symbol, 'generation': generation,
            'closed_trades': met.get('closed_trades'), 'win_rate': met.get('win_rate'),
            'expectancy': met.get('expectancy'), 'total_pnl': met.get('total_pnl'),
            'tp_hit_rate': met.get('tp_hit_rate'), 'sl_hit_rate': met.get('sl_hit_rate'),
            'reversal_rate': met.get('reversal_rate'),
            'params_before': json.dumps(params_before), 'params_after': json.dumps(new_params),
            'changes_json': json.dumps(pending.get('changes', [])),
            'verdict': verdict,
        })

        logger.info(f"🧬 [V2-EVO] {strategy} gen {generation} "
                    f"{'AUTOPILOT' if automatic else 'APPROVED by user'} "
                    f"({verdict}) regime={regime}")

        return {'success': True, 'strategy': strategy, 'symbol': symbol,
                'generation': generation, 'automatic': automatic,
                'params': new_params, 'status': status, 'regime': regime,
                'verdict': verdict, 'changes': pending.get('changes', [])}

    def dismiss(self, user_id: int, strategy: str, symbol: str = 'ALL') -> Dict:
        """Discard a pending lesson without applying it."""
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if not state:
            return {'success': False, 'error': 'No evolution state.'}
        self.db.v2_save_evolution_state(
            user_id, strategy, int(state.get('generation') or 0),
            state.get('params_json'), state.get('best_params_json'),
            state.get('best_expectancy'), int(state.get('trades_at_last_eval') or 0),
            state.get('status') or 'active', None, state.get('regime'),
            state.get('profiles_json'), symbol)
        return {'success': True, 'strategy': strategy, 'symbol': symbol}

    def set_status(self, user_id: int, strategy: str, symbol: str = 'ALL',
                   status: str = 'paused') -> Dict:
        """
        Hold ('paused') or resume ('active') acting on lessons for one pair.

        A held pair still watches every closed trade — metrics, the meter and
        the win-rate journey keep updating, and every generation already learned
        is kept. Only the *acting* stops, so resuming picks up where it left off
        rather than starting from generation zero. Any pending proposal is
        cleared on hold so a stale lesson can't be applied later.
        """
        if status not in ('active', 'paused'):
            return {'success': False, 'error': f"Unknown status '{status}'"}

        symbol = symbol or 'ALL'
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if not state:
            # Nothing learned yet — record the intent so a later evolve() honours it.
            self.db.v2_save_evolution_state(
                user_id, strategy, 0, json.dumps(dict(DEFAULT_PARAMS)),
                json.dumps(dict(DEFAULT_PARAMS)), None, 0, status, None, None,
                json.dumps({}), symbol)
        else:
            self.db.v2_save_evolution_state(
                user_id, strategy, int(state.get('generation') or 0),
                state.get('params_json'), state.get('best_params_json'),
                state.get('best_expectancy'), int(state.get('trades_at_last_eval') or 0),
                status,
                None if status == 'paused' else state.get('pending_json'),
                state.get('regime'), state.get('profiles_json'), symbol)

        logger.info(f"🧬 [V2-EVO] {strategy}/{symbol} lesson-acting {status} by user")
        return {'success': True, 'strategy': strategy, 'symbol': symbol, 'status': status}

    def set_auto_apply(self, user_id: int, strategy: str, symbol: str = 'ALL',
                       enabled: bool = True) -> Dict:
        """Turn autopilot on or off for one pair.

        On (the default): lessons are applied as soon as the evidence supports
        them. Off: each lesson waits for Proceed. Learning itself runs either
        way — this only decides who pulls the trigger.
        """
        symbol = symbol or 'ALL'
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if not state:
            self.db.v2_save_evolution_state(
                user_id, strategy, 0, json.dumps(dict(DEFAULT_PARAMS)),
                json.dumps(dict(DEFAULT_PARAMS)), None, 0, 'active', None, None,
                json.dumps({}), symbol, auto_apply=enabled)
        else:
            self.db.v2_save_evolution_state(
                user_id, strategy, int(state.get('generation') or 0),
                state.get('params_json'), state.get('best_params_json'),
                state.get('best_expectancy'), int(state.get('trades_at_last_eval') or 0),
                state.get('status') or 'active', state.get('pending_json'),
                state.get('regime'), state.get('profiles_json'), symbol,
                auto_apply=enabled)

        logger.info(f"🧬 [V2-EVO] {strategy}/{symbol} autopilot "
                    f"{'ON' if enabled else 'OFF'}")
        return {'success': True, 'strategy': strategy, 'symbol': symbol,
                'auto_apply': bool(enabled)}

    def evolve_all(self, user_id: int, strategies: List[str] = None) -> List[Dict]:
        """Evolve every strategy that has closing history."""
        if strategies is None:
            strategies = []
            try:
                trades = self.db.v2_get_user_trades(user_id=user_id, limit=5000)
                strategies = sorted({t.get('strategy') for t in trades if t.get('strategy')})
            except Exception:
                pass
        return [self.evolve(user_id, s) for s in strategies]

    def live_params(self, user_id: int, strategy: str, symbol: str = 'ALL') -> Dict:
        """
        Parameters a bot should trade with right now (evolved or default).

        `_evolved_keys` lists the parameters evolution has actually stored a
        value for. Callers must consult it before overriding a user's setting:
        a defaulted key is an absence of evidence, not a decision, and treating
        the two the same is how an explicit sensitivity choice used to get
        silently reset to conservative on every start.
        """
        state = self.db.v2_get_evolution_state(user_id, strategy, symbol)
        if not state:
            return {**DEFAULT_PARAMS, '_evolved_keys': []}
        try:
            p = json.loads(state.get('params_json') or '{}')
            return {**DEFAULT_PARAMS, **p, '_status': state.get('status', 'active'),
                    '_generation': state.get('generation', 0),
                    '_evolved_keys': [k for k in p if k in PARAM_BOUNDS]}
        except Exception:
            return {**DEFAULT_PARAMS, '_evolved_keys': []}


evolution_engine = EvolutionEngine()

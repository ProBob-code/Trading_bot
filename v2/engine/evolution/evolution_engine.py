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
5. **Fail-safe.** A strategy that stays unprofitable across generations is
   paused rather than left bleeding.

What it detects (the failure modes seen in real ledger data):
  • take-profit never hit  → target is unreachable for this instrument
  • stop-loss hit constantly → stop is inside normal noise
  • exits dominated by signal flips → whipsaw / over-trading
  • persistently negative expectancy → strategy unfit for current regime
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
PATIENCE_GENERATIONS = 3  # unprofitable generations tolerated before pausing


def _clamp(name: str, value: float) -> float:
    lo, hi = PARAM_BOUNDS[name]
    return max(lo, min(hi, value))


class EvolutionEngine:
    """Analyses closed trades and evolves per-strategy parameters."""

    def __init__(self, db=None):
        self.db = db or db_manager

    # ── Metrics ──────────────────────────────────────────────

    def compute_metrics(self, trades: List[Dict]) -> Dict:
        """Turn raw closing events into the signals evolution reacts to."""
        closed = [t for t in trades if t.get('action') in
                  ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL')]
        n = len(closed)
        if n == 0:
            return {'closed_trades': 0, 'win_rate': 0.0, 'expectancy': 0.0,
                    'total_pnl': 0.0, 'tp_hit_rate': 0.0, 'sl_hit_rate': 0.0,
                    'reversal_rate': 0.0, 'avg_win': 0.0, 'avg_loss': 0.0}

        pnls = [float(t.get('pnl') or 0) for t in closed]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p < 0]
        actions = [t.get('action') for t in closed]

        return {
            'closed_trades': n,
            'win_rate': len(wins) / n * 100,
            'expectancy': sum(pnls) / n,
            'total_pnl': sum(pnls),
            'tp_hit_rate': actions.count('TAKE_PROFIT') / n * 100,
            'sl_hit_rate': actions.count('STOP_LOSS') / n * 100,
            'reversal_rate': (actions.count('REVERSAL') + actions.count('CLOSE')) / n * 100,
            'avg_win': sum(wins) / len(wins) if wins else 0.0,
            'avg_loss': sum(losses) / len(losses) if losses else 0.0,
        }

    # ── Diagnosis → parameter changes ────────────────────────

    def propose_changes(self, params: Dict, m: Dict) -> List[Dict]:
        """Diagnose what went wrong and propose bounded, explained adjustments."""
        changes: List[Dict] = []
        n = m['closed_trades']

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

        # 1. Take-profit never reached → the target is unreachable for this market.
        #    Pull it toward the size of moves actually being captured.
        if m['tp_hit_rate'] == 0 and n >= MIN_TRADES:
            change('take_profit', params['take_profit'] * 0.6,
                   f"Take-profit never hit in {n} closed trades — target unreachable "
                   f"for this instrument's volatility. Lowering it so winners can "
                   f"actually be banked.")

        # 2. Stop-loss firing constantly → it sits inside normal noise.
        if m['sl_hit_rate'] > 40 and n >= MIN_TRADES:
            change('stop_loss', params['stop_loss'] * 1.25,
                   f"Stop-loss hit on {m['sl_hit_rate']:.0f}% of exits — the stop is "
                   f"inside normal price noise and is being triggered before the idea "
                   f"can play out. Widening it.")

        # 3. Exits dominated by signal flips AND losing → whipsaw / over-trading.
        if m['reversal_rate'] > 60 and m['expectancy'] < 0 and n >= MIN_TRADES:
            change('sensitivity_rank', params['sensitivity_rank'] - 1,
                   f"{m['reversal_rate']:.0f}% of exits came from the signal flipping, "
                   f"and expectancy is negative — the strategy is being whipsawed in a "
                   f"ranging market. Demanding stronger conviction before entering.")
            change('cooldown_bars', max(params['cooldown_bars'], 3) + 2,
                   "Adding a cool-down after each exit so one choppy stretch can't "
                   "trigger a chain of reversals (each one pays commission twice).")

        # 4. Losers much larger than winners → payoff geometry is upside-down.
        if m['avg_win'] > 0 and m['avg_loss'] < 0 and n >= MIN_TRADES:
            if abs(m['avg_loss']) > 2.5 * m['avg_win']:
                change('stop_loss', params['stop_loss'] * 0.8,
                       f"Average loss (${abs(m['avg_loss']):.2f}) is more than 2.5x the "
                       f"average win (${m['avg_win']:.2f}). Tightening the stop to cut "
                       f"the tail off losing trades.")

        # 5. Working well → stop fiddling. Over-tuning a winner is how edges die.
        if m['expectancy'] > 0 and not changes:
            return []

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
        """Evaluate a strategy and PROPOSE changes.

        This never mutates live trading parameters — it stores a pending
        proposal ("the lesson learned") for the user to review and approve.
        Applying it is `approve()`, triggered by the Proceed button.
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
        else:
            params, generation, best_params = dict(DEFAULT_PARAMS), 0, dict(DEFAULT_PARAMS)
            best_exp, seen, status = None, 0, 'active'
            profiles, prev_regime = {}, None

        meter = self.readiness(user_id, strategy, symbol)
        # Regime is measured on the instrument actually being traded.
        regime = self.detect_regime(symbol if symbol and symbol != 'ALL' else 'BTCUSDT')

        # Meter not full yet → keep gathering evidence, nothing to review.
        if not meter['ready'] and not force:
            self.db.v2_save_evolution_state(
                user_id, strategy, generation, json.dumps(params), json.dumps(best_params),
                best_exp, seen, status, state.get('pending_json') if state else None,
                regime, json.dumps(profiles), symbol)
            return {'skipped': True, 'strategy': strategy, 'symbol': symbol,
                    'meter': meter, 'regime': regime,
                    'reason': f"Gathering evidence — {meter['have']}/{meter['need']} "
                              f"closed trades until the next lesson is ready.",
                    'metrics': m, 'params': params, 'generation': generation}

        # ── Build the proposal (does NOT touch live params) ──
        candidate = dict(params)
        verdict = 'adapted'

        if best_exp is not None and m['expectancy'] > best_exp:
            best_exp, best_params = m['expectancy'], dict(params)
        elif best_exp is None:
            best_exp, best_params = m['expectancy'], dict(params)

        changes = self.propose_changes(candidate, m)

        # Market flipped and we have memory of this regime → propose recalling it.
        if regime != 'unknown' and prev_regime and regime != prev_regime and regime in profiles:
            remembered = profiles[regime]
            if remembered != params:
                candidate = dict(remembered)
                changes = [{'param': 'ALL', 'from': 'current', 'to': f'{regime} profile',
                            'reason': f"Market regime changed from {prev_regime} to {regime}. "
                                      f"Recalling the parameters this strategy already "
                                      f"learned for {regime} conditions."}]
                verdict = 'regime_switch'

        # Drifted worse than the best-known set → propose rolling back.
        if not changes and best_exp is not None and m['expectancy'] < best_exp and params != best_params:
            candidate = dict(best_params)
            changes = [{'param': 'ALL', 'from': 'current', 'to': 'best-known',
                        'reason': f"Recent expectancy (${m['expectancy']:.2f}) is below the best "
                                  f"set seen (${best_exp:.2f}). Proposing a rollback."}]
            verdict = 'revert'

        # Persistent losses → propose pausing rather than bleeding.
        if m['expectancy'] < 0 and generation >= PATIENCE_GENERATIONS and not changes:
            changes = [{'param': 'status', 'from': 'active', 'to': 'paused',
                        'reason': f"Still negative after {generation} generations — proposing "
                                  f"to pause this strategy so it stops losing money."}]
            verdict = 'pause'

        pending = None
        if changes:
            pending = {
                'proposed_at': datetime.now(timezone.utc).isoformat(),
                'regime': regime,
                'verdict': verdict,
                'changes': changes,
                'params_before': params,
                'params_after': candidate,
                'metrics': {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in m.items()},
                'next_generation': generation + 1,
            }

        self.db.v2_save_evolution_state(
            user_id, strategy, generation, json.dumps(params), json.dumps(best_params),
            best_exp, m['closed_trades'], status,
            json.dumps(pending) if pending else None, regime, json.dumps(profiles), symbol)

        if pending:
            logger.info(f"🧬 [V2-EVO] {strategy}: lesson ready ({verdict}, {len(changes)} change(s)) "
                        f"— awaiting user approval")

        return {'strategy': strategy, 'symbol': symbol, 'generation': generation, 'meter': meter,
                'regime': regime, 'metrics': m, 'params': params,
                'pending': pending, 'changes': changes, 'verdict': verdict,
                'awaiting_approval': bool(pending)}

    def approve(self, user_id: int, strategy: str, symbol: str = 'ALL') -> Dict:
        """Apply the pending proposal — this is the Proceed button.

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
        status = state.get('status') or 'active'
        regime = pending.get('regime') or state.get('regime') or 'unknown'
        profiles = json.loads(state.get('profiles_json') or '{}')

        if pending.get('verdict') == 'pause':
            status = 'paused'

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
        self.db.v2_add_evolution_generation({
            'user_id': user_id, 'strategy': strategy, 'symbol': symbol, 'generation': generation,
            'closed_trades': met.get('closed_trades'), 'win_rate': met.get('win_rate'),
            'expectancy': met.get('expectancy'), 'total_pnl': met.get('total_pnl'),
            'tp_hit_rate': met.get('tp_hit_rate'), 'sl_hit_rate': met.get('sl_hit_rate'),
            'reversal_rate': met.get('reversal_rate'),
            'params_before': json.dumps(params_before), 'params_after': json.dumps(new_params),
            'changes_json': json.dumps(pending.get('changes', [])),
            'verdict': pending.get('verdict', 'adapted'),
        })

        logger.info(f"🧬 [V2-EVO] {strategy} gen {generation} APPROVED by user "
                    f"({pending.get('verdict')}) regime={regime}")

        return {'success': True, 'strategy': strategy, 'symbol': symbol,
                'generation': generation,
                'params': new_params, 'status': status, 'regime': regime,
                'changes': pending.get('changes', [])}

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

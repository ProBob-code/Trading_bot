"""
V2 accounting invariants
========================

These lock down the arithmetic behind every performance number the product
shows. Each one corresponds to a bug that shipped and was visible to users, so
none of them should be relaxed without a very good reason.

Isolation: every row these tests write carries the reserved user id
`RESERVED_TEST_UID`, and setUp/tearDown delete exactly that user's rows and
nothing else. They therefore run against whichever backend `db_manager` has
already connected to — which matters because it is a module-level singleton, so
whichever test module imports first decides, and an env-var nudge here would
arrive too late to be reliable.
"""

import os
import sys
import unittest
from datetime import datetime, timedelta, timezone

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Prefer the SQLite fallback when this module is imported first. db_manager is a
# singleton that connects on import, so this only takes effect if nothing has
# imported it yet; the tests are safe either way (see the isolation note above).
os.environ.setdefault('MYSQLHOST', '127.0.0.1')
os.environ.setdefault('MYSQLPORT', '1')
for _k in ('ADMIN_USERNAME', 'ADMIN_PASSWORD', 'ADMIN_USERNAMES', 'ADMIN_MOBILES'):
    os.environ[_k] = ''

from shared.database.db_manager import db_manager                      # noqa: E402
from shared.logic.trade_actions import CLOSING_ACTIONS, OPENING_ACTIONS  # noqa: E402
from shared.logic.strategies.public_catalog import interval_for         # noqa: E402
from v2.engine.evolution.evolution_engine import evolution_engine       # noqa: E402
from v2.engine.execution.paper_trader_v2 import PaperTraderV2           # noqa: E402

# A user id no real account will ever hold. Every write here is tagged with it
# and removed again, so these tests cannot disturb real data on any backend.
RESERVED_TEST_UID = 990000


class V2AccountingBase(unittest.TestCase):
    """Seeds the reserved test user, and removes every trace of it afterwards."""

    uid = RESERVED_TEST_UID

    def setUp(self):
        self.assertEqual(self.uid, RESERVED_TEST_UID,
                         'tests must only ever write as the reserved user')
        if not db_manager.use_sqlite:                       # pragma: no cover
            self.skipTest(
                'refusing to write to a live MySQL. db_manager is a singleton '
                'that connects on import, so whichever test module imports '
                'first picks the backend; run this file on its own '
                '(python -m unittest tests.test_v2_accounting) and it uses the '
                'SQLite fallback.')
        self._n = 0
        self._wipe()

    def tearDown(self):
        self._wipe()

    def _exec(self, sql, params=()):
        conn = db_manager._get_connection()
        cur = conn.cursor()
        db_manager._execute(cur, sql, params)
        conn.commit()
        conn.close()

    def _wipe(self):
        for table in ('v2_trade_ledger', 'v2_positions',
                      'v2_evolution_state', 'v2_evolution_history'):
            self._exec("DELETE FROM %s WHERE user_id = %%s" % table, (self.uid,))

    def row(self, action, pnl=0.0, commission=0.0, bot_id='bot',
            strategy='ichimoku', symbol='BTCUSDT'):
        """Append one ledger row, in chronological order."""
        db_manager.v2_save_trade({
            'trade_id': 'test_%s_%s' % (self.uid, self._n),
            'session_id': 'test', 'user_id': self.uid, 'bot_id': bot_id,
            'symbol': symbol, 'side': 'BUY', 'action': action,
            'quantity': 1.0, 'price': 100.0, 'pnl': pnl, 'commission': commission,
            'strategy': strategy,
            'timestamp': datetime(2026, 1, 1, tzinfo=timezone.utc)
                         + timedelta(minutes=self._n),
        })
        self._n += 1

    def flip(self, pnl, commission=0.0, **kw):
        """A reversal: the CLOSE carries the result, the REVERSAL is the entry."""
        self.row('CLOSE', pnl, commission, **kw)
        self.row('REVERSAL', 0.0, commission, **kw)


class TestReversalIsNotAClosedTrade(V2AccountingBase):
    """A flip is ONE trade. Counting its entry leg halves every win rate."""

    def test_taxonomy_places_reversal_on_the_opening_side(self):
        self.assertIn('REVERSAL', OPENING_ACTIONS)
        self.assertNotIn('REVERSAL', CLOSING_ACTIONS)

    def test_metrics_count_each_flip_once(self):
        for pnl in (10.0, 20.0, -5.0, 30.0):
            self.flip(pnl)
        self.row('STOP_LOSS', -8.0)

        m = evolution_engine.compute_metrics(
            db_manager.v2_get_closed_trades_for_eval(self.uid, 'ichimoku', 'BTCUSDT'))

        # 4 flips + 1 stop-out = 5 outcomes, 3 of them winners.
        self.assertEqual(m['closed_trades'], 5)
        self.assertAlmostEqual(m['win_rate'], 60.0)
        self.assertAlmostEqual(m['expectancy'], (10 + 20 - 5 + 30 - 8) / 5)

    def test_phantom_rows_do_not_become_break_evens(self):
        for pnl in (10.0, -4.0):
            self.flip(pnl)
        stats = db_manager.v2_get_bot_ledger_stats(self.uid)['bot']
        self.assertEqual(stats['closed_trades'], 2)
        self.assertEqual(stats['breakeven'], 0)

    def test_close_filter_excludes_the_entry_leg(self):
        self.flip(10.0)
        self.row('INCREASE')
        rows = db_manager.v2_get_user_trades(user_id=self.uid, trade_type='CLOSE')
        self.assertEqual([r['action'] for r in rows], ['CLOSE'])


class TestProfitFactorSemantics(V2AccountingBase):
    """Undefined and zero are opposite facts and must not share a value."""

    def test_no_losses_is_undefined_not_zero(self):
        self.row('TAKE_PROFIT', 7.0)
        self.assertIsNone(db_manager.v2_get_bot_ledger_stats(self.uid)['bot']['profit_factor'])

    def test_no_wins_is_a_real_zero(self):
        self.row('STOP_LOSS', -7.0)
        self.assertEqual(db_manager.v2_get_bot_ledger_stats(self.uid)['bot']['profit_factor'], 0.0)

    def test_mixed_is_the_ratio(self):
        self.row('TAKE_PROFIT', 20.0)
        self.row('STOP_LOSS', -10.0)
        self.assertAlmostEqual(
            db_manager.v2_get_bot_ledger_stats(self.uid)['bot']['profit_factor'], 2.0)

    def test_break_evens_are_counted_separately(self):
        self.row('TAKE_PROFIT', 5.0)
        self.row('CLOSE', 0.0)
        stats = db_manager.v2_get_bot_ledger_stats(self.uid)['bot']
        self.assertEqual((stats['wins'], stats['losses'], stats['breakeven']), (1, 0, 1))
        self.assertEqual(stats['wins'] + stats['losses'] + stats['breakeven'],
                         stats['closed_trades'])


class TestRealisedPnlSurvivesRestart(V2AccountingBase):
    """In-memory cash resets on restart; the ledger is the durable record."""

    def test_entry_commission_is_subtracted_exactly_once(self):
        # A CLOSE's pnl is already net of ITS commission; the entry's is not.
        self.row('OPEN', 0.0, 0.40)
        self.row('TAKE_PROFIT', 25.0, 0.40)
        summary = db_manager.v2_get_realized_summary(self.uid)
        self.assertAlmostEqual(summary['pnl_sum'], 25.0)
        self.assertAlmostEqual(summary['opening_commission'], 0.40)
        self.assertAlmostEqual(summary['realized'], 24.60)

    def test_balance_is_rebuilt_from_the_ledger(self):
        self.row('OPEN', 0.0, 0.40)
        self.row('STOP_LOSS', -30.0, 0.40)

        trader = PaperTraderV2(initial_capital=100000)
        trader.load_positions(self.uid, db_manager)
        info = trader.get_account_info(self.uid)

        # The bug this replaces reported a pristine 100000 here.
        self.assertAlmostEqual(info['equity'], 100000 - 30.40, places=2)

    def test_a_never_traded_account_still_reads_as_pristine(self):
        trader = PaperTraderV2(initial_capital=100000)
        trader.load_positions(self.uid, db_manager)
        self.assertAlmostEqual(trader.get_account_info(self.uid)['equity'], 100000.0)


class TestSurfacesReconcile(V2AccountingBase):
    """Balance, bot cards and the admin console must tell the same story."""

    def test_bot_stats_survive_bot_id_churn(self):
        # The same strategy+symbol written under three bot_ids, as a restart or
        # an evolution tweak produces.
        for bot, pnls in (('hash_a', [40.0, -10.0]), ('hash_b', [25.0]),
                          ('hash_c', [-12.0])):
            for pnl in pnls:
                self.row('TAKE_PROFIT' if pnl > 0 else 'STOP_LOSS', pnl, bot_id=bot)

        by_pair = db_manager.v2_get_pair_ledger_stats(self.uid)[('ichimoku', 'BTCUSDT')]
        by_bot = db_manager.v2_get_bot_ledger_stats(self.uid)

        self.assertEqual(by_pair['closed_trades'], 4)
        self.assertLess(max(v['closed_trades'] for v in by_bot.values()), 4,
                        'no single bot_id should hold the whole history')

    def test_admin_net_pnl_equals_the_account_balance(self):
        self.row('OPEN', 0.0, 0.50)
        self.row('TAKE_PROFIT', 60.0, 0.50)
        self.row('OPEN', 0.0, 0.50)
        self.row('STOP_LOSS', -20.0, 0.50)

        trader = PaperTraderV2(initial_capital=100000)
        trader.load_positions(self.uid, db_manager)
        equity = trader.get_account_info(self.uid)['equity']

        admin = db_manager.admin_user_trade_stats()[self.uid]
        self.assertAlmostEqual(admin['net_pnl'], equity - 100000, places=2)
        # total_pnl is the raw column sum and is NOT the net figure.
        self.assertAlmostEqual(admin['total_pnl'], 40.0, places=2)
        self.assertAlmostEqual(admin['net_pnl'], 39.0, places=2)

    def test_account_total_equals_the_sum_of_its_pairs(self):
        self.row('TAKE_PROFIT', 10.0, symbol='BTCUSDT')
        self.row('STOP_LOSS', -4.0, symbol='ETHUSDT')
        self.row('TAKE_PROFIT', 6.0, symbol='ETHUSDT', strategy='bollinger')

        admin = db_manager.admin_user_trade_stats()[self.uid]
        pairs = db_manager.v2_get_pair_ledger_stats(self.uid)
        self.assertEqual(admin['closed_trades'],
                         sum(v['closed_trades'] for v in pairs.values()))


class TestEverySurfaceReconciles(V2AccountingBase):
    """The invariant the whole dashboard rests on.

    Built from a ledger with the same SHAPE as real data — opens, position
    adds, flips written as CLOSE+REVERSAL pairs, stop-outs and take-profits,
    spread over several strategy/symbol pairs. Verified once against a real
    976-row production ledger; this keeps that result from regressing.
    """

    def _seed_realistic_ledger(self):
        """Returns (closed, wins, pnl_sum, entry_commission)."""
        plan = [
            ('ichimoku', 'BTCUSDT', [12.0, -5.0, 30.0, -8.0]),
            ('ichimoku', 'ETHUSDT', [-3.0, -9.0, 4.0]),
            ('bollinger', 'SOLUSDT', [7.0, 7.0, -2.0, 0.0]),   # includes a break-even
            ('ml_forecast', 'ETHUSDT', [-15.0]),
        ]
        comm = 0.25
        closed = wins = 0
        pnl_sum = entry_comm = 0.0

        for strategy, symbol, pnls in plan:
            kw = dict(strategy=strategy, symbol=symbol,
                      bot_id='bot_%s_%s' % (strategy, symbol))
            self.row('OPEN', 0.0, comm, **kw)
            entry_comm += comm
            self.row('INCREASE', 0.0, comm, **kw)      # adds carry no outcome
            entry_comm += comm

            for i, pnl in enumerate(pnls):
                if i % 2 == 0:                          # a flip: two rows, one trade
                    self.row('CLOSE', pnl, comm, **kw)
                    self.row('REVERSAL', 0.0, comm, **kw)
                    entry_comm += comm
                else:
                    self.row('STOP_LOSS' if pnl < 0 else 'TAKE_PROFIT', pnl, comm, **kw)
                pnl_sum += pnl
                closed += 1
                wins += 1 if pnl > 0 else 0

        return closed, wins, pnl_sum, entry_comm

    def test_all_surfaces_agree(self):
        closed, wins, pnl_sum, entry_comm = self._seed_realistic_ledger()
        realized = pnl_sum - entry_comm

        # Headline counter counts round-trips, not rows.
        self.assertEqual(db_manager.v2_get_closed_trade_count(self.uid), closed)
        self.assertGreater(db_manager.v2_get_total_trade_count(user_id=self.uid), closed)

        # Balance.
        trader = PaperTraderV2(initial_capital=100000)
        trader.load_positions(self.uid, db_manager)
        equity = trader.get_account_info(self.uid)['equity']
        self.assertAlmostEqual(equity, 100000 + realized, places=2)

        # Admin, and its agreement with the balance.
        admin = db_manager.admin_user_trade_stats()[self.uid]
        self.assertEqual(admin['closed_trades'], closed)
        self.assertEqual(admin['wins'], wins)
        self.assertAlmostEqual(admin['net_pnl'], realized, places=2)
        self.assertAlmostEqual(admin['net_pnl'], equity - 100000, places=2)

        # Bot cards sum to the account.
        pairs = db_manager.v2_get_pair_ledger_stats(self.uid)
        self.assertEqual(sum(v['closed_trades'] for v in pairs.values()), closed)
        self.assertEqual(sum(v['wins'] for v in pairs.values()), wins)
        self.assertAlmostEqual(sum(v['realized_pnl'] for v in pairs.values()),
                               pnl_sum, places=2)

        # Admin strategy usage matches the same pairs.
        usage = [r for r in db_manager.admin_strategy_usage()
                 if r['user_id'] == self.uid]
        self.assertEqual(len(usage), len(pairs))
        self.assertEqual(sum(r['closed_trades'] for r in usage), closed)

        # Evolution reads the same trades the cards do — per pair, not just in
        # total, so an offsetting pair of errors cannot hide.
        for (strategy, symbol), card in pairs.items():
            row = next(r for r in usage
                       if r['strategy'] == strategy and r['symbol'] == symbol)
            metrics = evolution_engine.compute_metrics(
                db_manager.v2_get_closed_trades_for_eval(
                    self.uid, strategy, symbol, limit=100000))
            self.assertEqual(card['closed_trades'], row['closed_trades'],
                             '%s/%s: card vs admin' % (strategy, symbol))
            self.assertEqual(card['closed_trades'], metrics['closed_trades'],
                             '%s/%s: card vs evolution' % (strategy, symbol))
            self.assertAlmostEqual(card['win_rate'], row['win_rate'], places=6)
            self.assertAlmostEqual(card['win_rate'], metrics['win_rate'], places=6)


class TestStrategyOwnsItsTimeframe(unittest.TestCase):
    """The interval is a property of the strategy, not a user preference."""

    def test_every_catalog_strategy_has_a_timeframe(self):
        for internal in ('combined', 'ichimoku', 'bollinger', 'macd_rsi',
                         'ml_forecast', 'quant_alpha', 'quant_meanrev',
                         'quant_momentum'):
            self.assertIn(interval_for(internal),
                          ('1m', '5m', '15m', '30m', '1h', '4h', '1d'))

    def test_unknown_strategy_falls_back_rather_than_failing(self):
        self.assertEqual(interval_for('no_such_strategy'), '15m')


if __name__ == '__main__':
    unittest.main(verbosity=2)

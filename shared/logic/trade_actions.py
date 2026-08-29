"""
Ledger action taxonomy — the single definition of what counts as a trade
=======================================================================

Every row in `v2_trade_ledger` carries an `action`. Getting the classification
wrong silently corrupts every performance figure in the product, so it is
defined once, here, and imported everywhere rather than re-listed at each call
site.

Why REVERSAL is an OPENING action
---------------------------------
When a strategy flips direction, `PaperTraderV2.execute_trade` writes **two**
rows:

    _close_position(...)                      -> action='CLOSE',    pnl = the real result
    _open_position(..., action='REVERSAL')    -> action='REVERSAL', pnl = 0

The REVERSAL row is the *entry* leg of the new position. Its P&L is always
exactly zero because the outcome of the trade that just ended is recorded on the
paired CLOSE.

REVERSAL was previously listed as a closing action. Every flip therefore counted
as two closed trades, one of which could never be a win — so win rate was
divided by an inflated denominator and expectancy was diluted toward zero. On
real account data that reported a 22.0% win rate where the truth was 39.6%, and
showed one strategy at 38.9% when it was actually winning 63.8% of its trades.
Losses looked roughly half as severe as they were, and the evolution engine was
tuning itself against those diluted numbers.

INCREASE is likewise an opening action: it adds to a position that is already
open, and carries no P&L of its own.
"""

# Rows that ESTABLISH or add to exposure. No realised P&L; never a "trade
# outcome".
OPENING_ACTIONS = ('OPEN', 'INCREASE', 'REVERSAL')

# Rows that REALISE a result. These, and only these, are the closed trades that
# win rate, expectancy and profit factor are computed over.
CLOSING_ACTIONS = ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT')


def sql_in_list(actions=CLOSING_ACTIONS) -> str:
    """Render an action tuple as a SQL IN list.

    Values are module constants, never user input, but building the fragment
    here keeps every query in step with the taxonomy above instead of carrying
    its own hand-typed copy that can drift.
    """
    return ','.join("'%s'" % a for a in actions)


def is_closing(action) -> bool:
    return action in CLOSING_ACTIONS


def is_opening(action) -> bool:
    return action in OPENING_ACTIONS

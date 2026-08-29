"""
V2 Trading Pipeline — Institutional Grade
=========================================

The central coordinator that drives a single trading "tick" 
through the modular decision flow.
"""

from typing import Dict, Any, List
from loguru import logger
from shared.logic.trade_actions import CLOSING_ACTIONS
import datetime

class TradingPipelineV2:
    def __init__(
        self, 
        risk_engine, 
        portfolio_engine, 
        paper_trader, 
        db_manager,
        socketio=None,
        bot_manager=None
    ):
        self.risk = risk_engine
        self.portfolio = portfolio_engine
        self.trader = paper_trader
        self.db = db_manager
        self.socketio = socketio
        self.bot_manager = bot_manager

    def run_tick(self, bot_id: str, bot: Any, signal: Any, current_price: float, atr_value: float):
        """
        Execute a full pipeline pass for a single bot signal.
        """
        print(f"[PIPELINE] Tick received: {bot.config.symbol} @ {current_price}")
        config = bot.config
        user_id = config.user_id
        symbol = config.symbol
        strategy = config.strategy
        leverage = config.leverage

        # ── 0. TP / SL enforcement (runs every tick, independent of the signal) ──
        # Auto-close an open position the moment it reaches the bot's configured
        # take-profit or stop-loss, so Command Deck settings are actually honored.
        if self.check_exit_conditions(bot_id, bot, current_price, atr_value):
            return  # position was closed this tick — no new entry

        # 1. State Context
        pos = self.portfolio.get_position_state(user_id, symbol)
        has_pos = pos is not None
        pos_side = pos.get('side', '') if has_pos else ''

        # 2. Risk Gating
        # Determine expected move (default 0.5% if not provided)
        expected_move = getattr(signal, 'expected_move_pct', 0.5) / 100
        signal_score = getattr(signal, 'score', 0)
        
        print(f"[PIPELINE] Signal: {signal.signal} (Score: {signal_score:.2f})")
        
        if signal.signal == 'HOLD':
             print("[PIPELINE] ❌ No signal generated (HOLD)")
        
        # Volatility filter check (placeholder for now, will be passed from loop)
        vol_filter = {'allowed': True}

        # Per-bot Signal Sensitivity → risk-gate thresholds (score floor + cost bar).
        # Keeps the portfolio-level gate in step with the strategy's own gates.
        from shared.logic.strategies.v3_quant_strategies import (
            resolve_sensitivity, ROUND_TRIP_COST_PCT)
        sens = resolve_sensitivity(getattr(config, 'sensitivity', 'conservative'))

        allowed, reason = self.risk.pre_trade_gate(
            user_id, symbol, signal.signal, signal_score, expected_move, vol_filter,
            score_floor=sens['risk_floor'],
            estimated_cost_pct=ROUND_TRIP_COST_PCT * sens['cost_mult'],
        )
        
        print(f"[PIPELINE] Risk check: {allowed} ({reason})")
        
        if not allowed:
            if signal.signal in ('BUY', 'SELL'):
                logger.info(f"🛡️ [V2-PIPELINE-{bot_id}] Risk rejected {signal.signal}: {reason}")
                print(f"[PIPELINE] ❌ Rejected by risk engine: {reason}")
            return

        # 3. Decision Logic (Signal + Position state)
        if signal.signal in ('BUY', 'SELL'):
            print(f"[PIPELINE] ✅ Execution triggered: {signal.signal} {config.max_quantity}")
            # 4. Execution & Persistence
            results = self.trader.execute_trade(
                user_id=user_id,
                symbol=symbol,
                side=signal.signal,
                quantity=config.max_quantity, # Default to max_quantity for auto-trading
                leverage=leverage,
                strategy=strategy,
                volatility=atr_value/current_price if current_price > 0 else 0.02
            )
            
            # Sync to DB and update stats
            self.portfolio.sync_position(user_id, symbol, results, strategy, leverage)
            self._handle_trade_results(bot, results, signal.signal, config.max_quantity, current_price)

    def check_exit_conditions(self, bot_id, bot, current_price, atr_value) -> bool:
        """Close an open position if it hit the bot's take-profit or stop-loss.

        Safe to call every loop iteration (not just on a new candle) so exits are
        monitored continuously. Returns True if a position was closed (so the
        caller skips new entries this tick). P&L% is read from the live position,
        matching the leveraged percentage shown in the UI.
        """
        config = bot.config
        user_id = config.user_id
        symbol = config.symbol

        pos = self.trader.get_position(user_id, symbol)
        if not pos or pos.quantity <= 0:
            return False

        pnl_pct = pos.unrealized_pnl_pct(current_price)
        tp = float(getattr(config, 'take_profit', 0) or 0)
        sl = float(getattr(config, 'stop_loss', 0) or 0)

        exit_action = None
        if tp > 0 and pnl_pct >= tp:
            exit_action = 'TAKE_PROFIT'
        elif sl > 0 and pnl_pct <= -sl:
            exit_action = 'STOP_LOSS'

        if not exit_action:
            return False

        emoji = '🎯' if exit_action == 'TAKE_PROFIT' else '🛑'
        logger.info(
            f"{emoji} [V2-PIPELINE-{bot_id}] {exit_action} on {symbol}: "
            f"pnl={pnl_pct:.2f}% (tp={tp}%, sl={sl}%) — closing {pos.quantity} @ {current_price}"
        )
        print(f"[PIPELINE] {emoji} {exit_action}: closing {symbol} at pnl {pnl_pct:.2f}%")

        vol = atr_value / current_price if current_price > 0 else 0.02
        results = self.trader.close_position(user_id, symbol, action=exit_action, volatility=vol)

        if not results or not results[0].get('success'):
            err = results[0].get('error') if results else 'unknown'
            logger.warning(f"⚠️ [V2-PIPELINE-{bot_id}] {exit_action} close failed: {err}")
            return False

        close_side = results[0].get('side', '')
        closed_qty = results[0].get('quantity', pos.quantity)
        self.portfolio.sync_position(user_id, symbol, results, config.strategy, config.leverage)
        self._handle_trade_results(bot, results, close_side, closed_qty, current_price)
        return True

    def _apply_evolved_params(self, bot, result):
        """Push evolved parameters onto the running bot so they take effect now.

        Only touches knobs the pipeline actually reads (take_profit / stop_loss /
        sensitivity). Evolution never stops a bot: a strategy that keeps losing
        is de-risked to the tightest settings its bounds allow and left running,
        because a bot that stops trading stops producing the evidence it needs
        to recover.
        """
        from v2.engine.evolution.evolution_engine import SENSITIVITY_BY_RANK

        params = result.get('params') or {}
        cfg = bot.config
        before = (cfg.take_profit, cfg.stop_loss, getattr(cfg, 'sensitivity', None))

        if 'take_profit' in params:
            cfg.take_profit = float(params['take_profit'])
        if 'stop_loss' in params:
            cfg.stop_loss = float(params['stop_loss'])
        if 'sensitivity_rank' in params:
            rank = int(max(0, min(2, params['sensitivity_rank'])))
            cfg.sensitivity = SENSITIVITY_BY_RANK[rank]

        if before != (cfg.take_profit, cfg.stop_loss, getattr(cfg, 'sensitivity', None)):
            logger.info(f"🧬 [V2-EVO] Applied gen {result.get('generation')} to {bot.bot_id}: "
                        f"TP={cfg.take_profit}% SL={cfg.stop_loss}% sens={cfg.sensitivity}")

    def _handle_trade_results(self, bot, results, side, quantity, price):
        """Standardized processing for trades (stats, logs, sockets)."""
        current_session = self.db.v2_get_active_session_id()
        if not current_session:
            logger.warning(f"⚠️ [V2-PIPELINE-{bot.bot_id}] No active session found for trade logging.")
        
        for res in results:
            if not res.get('success'):
                continue
                
            # Update Bot Stats
            pnl = res.get('pnl', 0) or 0
            if self.bot_manager:
                self.bot_manager.increment_trades(bot.bot_id, res.get('side', side), pnl)
            
            # Save to Ledger (Institutional Single Source of Truth)
            record = {
                'trade_id': res.get('trade_id'),
                'session_id': current_session,
                'user_id': bot.config.user_id,
                'bot_id': bot.bot_id,
                'symbol': bot.config.symbol,
                'side': res.get('side', side),
                'action': res.get('action', 'TRADE'),
                'quantity': res.get('quantity', quantity),
                'price': res.get('price', price),
                'pnl': pnl,
                'commission': res.get('commission', 0),
                'strategy': res.get('strategy', bot.config.strategy),
                'timestamp': datetime.datetime.utcnow()
            }
            self.db.v2_save_trade(record)
            
            # Atomically update session counters if it was a closing or reversal action
            if res.get('action') in CLOSING_ACTIONS:
                self.db.v2_update_session_counters(current_session, pnl)

                # ── Evolution: learn from this closed trade ──
                # Runs on every close (that's where outcome is known). The engine
                # gates itself on sample size, so this is cheap on most calls.
                # With autopilot on it applies the lesson itself and we push the
                # new parameters onto this running bot immediately; with autopilot
                # off it stores the lesson for the user to review and Proceed.
                try:
                    from v2.engine.evolution.evolution_engine import evolution_engine
                    result = evolution_engine.evolve(bot.config.user_id, bot.config.strategy,
                                                     bot.config.symbol)

                    if result and result.get('auto_applied'):
                        self._apply_evolved_params(bot, result)

                    if result and self.socketio and (result.get('awaiting_approval')
                                                     or result.get('auto_applied')):
                        # The browser only ever sees the public catalog code —
                        # the internal strategy id is the firm's IP and must not
                        # ride out on a socket event.
                        from shared.logic.strategies.public_catalog import to_public
                        self.socketio.emit('v2_evolution_ready', {
                            'strategy': to_public(bot.config.strategy),
                            'symbol': bot.config.symbol,
                            'regime': result.get('regime'),
                            'verdict': result.get('verdict'),
                            'changes': result.get('changes'),
                            'meter': result.get('meter'),
                            'journey': result.get('journey'),
                            'auto_applied': bool(result.get('auto_applied')),
                            'generation': result.get('generation'),
                        }, room=f"user_{bot.config.user_id}")
                except Exception as e:
                    logger.warning(f"⚠️ [V2-EVO] evolution step failed: {e}")

            # Sockets
            if self.socketio:
                self.socketio.emit('v2_trade_executed', {
                    'side': res.get('side', side),
                    'symbol': bot.config.symbol,
                    'quantity': res.get('quantity', quantity),
                    'price': res.get('price', price),
                    'pnl': pnl,
                    'action': res.get('action'),
                    'bot_id': bot.bot_id,
                    'engine': 'v2'
                }, room=f"user_{bot.config.user_id}")

            logger.info(f"✅ [V2-PIPELINE-{bot.bot_id}] Ledger entry: {res.get('action')} {res.get('side')} | PnL: {pnl:.2f}")

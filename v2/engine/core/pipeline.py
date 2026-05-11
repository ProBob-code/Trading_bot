"""
V2 Trading Pipeline — Institutional Grade
=========================================

The central coordinator that drives a single trading "tick" 
through the modular decision flow.
"""

from typing import Dict, Any, List
from loguru import logger
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
        
        allowed, reason = self.risk.pre_trade_gate(
            user_id, symbol, signal.signal, signal_score, expected_move, vol_filter
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
            if res.get('action') in ('CLOSE', 'STOP_LOSS', 'TAKE_PROFIT', 'REVERSAL'):
                self.db.v2_update_session_counters(current_session, pnl)

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

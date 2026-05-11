"""
Multi-Bot Orchestrator
======================

Drives the paper trading simulation loop:
1. Fetch OHLCV data
2. Detect market regime
3. Run each bot's strategy → signals
4. Simulate execution (slippage/fees/order type/delay)
5. Validate and clean trade records
6. Insert to DB, update wallet
7. Recalculate metrics
8. Enforce risk guardrails (daily loss, correlation, circuit breaker)
"""

import uuid
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd

from .bot_config import BotConfig
from .simulation import TradeSimulator
from .data_cleaner import DataCleaner
from .db import PaperDB
from .metrics import MetricsCalculator
from .risk import RiskManager
from .comparator import BotComparator
from .validation import MonteCarloSimulator
from shared.config.settings import is_v1_enabled

from .strategies.base import StrategyBot, TradeSignal
from .strategies.breakout import BreakoutBot
from .strategies.mean_reversion import MeanReversionBot
from .strategies.ichimoku import IchimokuBot
from .strategies.ml_forecast import MLForecastBot

logger = logging.getLogger(__name__)


# Strategy registry (Gated for V1 Shutdown)
STRATEGY_MAP = {}

if is_v1_enabled():
    STRATEGY_MAP = {
        'breakout': BreakoutBot,
        'mean_reversion': MeanReversionBot,
        'ichimoku': IchimokuBot,
        'ml_forecast': MLForecastBot,
    }
else:
    logger.info("🛡️ V1 Strategy Registration Skipped (V1 Trading Disabled)")


class Orchestrator:
    """
    Multi-bot paper trading orchestrator.
    
    Manages the lifecycle of multiple strategy bots,
    each with isolated capital, running on shared or different instruments.
    """
    
    def __init__(
        self,
        configs: List[BotConfig],
        db: PaperDB = None,
    ):
        self.db = db or PaperDB()
        self.configs: Dict[str, BotConfig] = {c.bot_id: c for c in configs}
        self.bots: Dict[str, StrategyBot] = {}
        self.simulators: Dict[str, TradeSimulator] = {}
        self.cleaner = DataCleaner()
        self.risk_manager = RiskManager()
        self.mc = MonteCarloSimulator(n_simulations=500)
        
        # Open trade tracking per bot
        self._open_trades: Dict[str, Dict[str, Any]] = {}  # bot_id → trade_info
        
        # Initialize bots and wallets
        for bot_id, config in self.configs.items():
            self._init_bot(config)
    
    def _init_bot(self, config: BotConfig):
        """Initialize a single bot: strategy, simulator, wallet."""
        strategy_cls = STRATEGY_MAP.get(config.strategy)
        if not strategy_cls:
            logger.error(f"Unknown strategy: {config.strategy}")
            return
        
        self.bots[config.bot_id] = strategy_cls(
            params=config.params,
            min_rr=config.min_rr,
        )
        
        self.simulators[config.bot_id] = TradeSimulator(
            base_slippage_pct=config.slippage_base_pct,
            spread_pct=config.spread_pct,
            maker_fee_pct=config.maker_fee_pct,
            taker_fee_pct=config.taker_fee_pct,
            execution_mode=config.execution_mode,
            max_position_volume_pct=config.max_position_volume_pct,
        )
        
        self.db.init_wallet(config.bot_id, config.virtual_capital)
        logger.info(f"Initialized bot: {config.bot_id} ({config.strategy})")
    
    def detect_regime(self, df: pd.DataFrame) -> Dict[str, str]:
        """
        Simple regime detection.
        Uses ADX for trend and ATR percentile for volatility.
        """
        if len(df) < 50:
            return {'trend': 'unknown', 'volatility': 'unknown'}
        
        # ADX approximation using directional movement
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        
        tr = np.maximum(
            high[-14:] - low[-14:],
            np.maximum(
                np.abs(high[-14:] - close[-15:-1]),
                np.abs(low[-14:] - close[-15:-1])
            )
        )
        atr_14 = np.mean(tr)
        
        # Trend: use slope of 20-bar SMA
        sma_20 = np.mean(close[-20:])
        sma_20_prev = np.mean(close[-40:-20]) if len(close) >= 40 else sma_20
        trend_slope = (sma_20 - sma_20_prev) / max(sma_20_prev, 1) * 100
        
        if abs(trend_slope) > 2:
            trend = 'trending'
        else:
            trend = 'ranging'
        
        # Volatility: ATR percentile vs rolling
        atr_pct = atr_14 / close[-1] * 100 if close[-1] > 0 else 0
        if atr_pct > 3:
            volatility = 'high'
        elif atr_pct < 1:
            volatility = 'low'
        else:
            volatility = 'normal'
        
        return {'trend': trend, 'volatility': volatility}
    
    def process_bar(
        self,
        df: pd.DataFrame,
        bar_idx: int,
    ) -> List[Dict[str, Any]]:
        """
        Process a single bar across all bots.
        
        Returns list of executed trade records.
        """
        executed_trades = []
        regime = self.detect_regime(df)
        
        for bot_id, config in self.configs.items():
            if not config.enabled:
                continue
            
            bot = self.bots.get(bot_id)
            sim = self.simulators.get(bot_id)
            if not bot or not sim:
                continue
            
            wallet = self.db.get_wallet(bot_id)
            if not wallet:
                continue
            
            current_equity = wallet.get('current_equity', config.virtual_capital)
            peak_equity = wallet.get('peak_equity', config.virtual_capital)
            
            # ── Check for open trade exit ──
            if bot_id in self._open_trades:
                trade_result = self._check_exit(
                    bot_id, df, bar_idx, config, sim, wallet
                )
                if trade_result:
                    executed_trades.append(trade_result)
                continue  # Skip new entry while in trade
            
            # ── Risk checks ──
            # Drawdown ladder
            adjusted_risk, circuit_breaker = self.risk_manager.apply_drawdown_ladder(
                current_equity, peak_equity, config.risk_per_trade_pct
            )
            
            # Daily loss check
            ok, reason = self.risk_manager.check_daily_loss(
                sum(w.get('current_equity', 0) for w in self.db.get_all_wallets())
            )
            if not ok:
                continue
            
            # Concurrent trades check
            ok, reason = self.risk_manager.check_concurrent_trades(
                bot_id, config.max_concurrent_trades
            )
            if not ok:
                continue
            
            # ── Generate signal ──
            try:
                df_with_indicators = bot.calculate_indicators(df)
                signal = bot.generate_signal(df_with_indicators, bar_idx, regime)
            except Exception as e:
                logger.error(f"Bot {bot_id} signal error: {e}")
                continue
            
            if signal is None:
                continue
            
            # ── Validate signal ──
            is_valid, validation_reason = bot.validate_signal(signal)
            if not is_valid:
                logger.debug(f"Bot {bot_id} signal rejected: {validation_reason}")
                continue
            
            # ── Position sizing ──
            sizing = self.risk_manager.calculate_position_size(
                capital=current_equity,
                risk_pct=adjusted_risk,
                entry_price=signal.entry_price,
                sl_price=signal.sl_price,
            )
            
            if sizing['rejected']:
                continue
            
            position_size = sizing['units']
            position_value = sizing['value']
            
            # ── Liquidity check ──
            bar = df.iloc[bar_idx]
            avg_vol_20 = df['volume'].iloc[max(0, bar_idx-20):bar_idx].mean() if bar_idx >= 20 else bar.get('volume', 0)
            
            ok, reason = self.risk_manager.check_liquidity(
                position_size, avg_vol_20, config.max_position_volume_pct
            )
            if not ok:
                logger.debug(f"Bot {bot_id} liquidity reject: {reason}")
                continue
            
            # ── Correlation check ──
            total_capital = sum(
                w.get('current_equity', 0) for w in self.db.get_all_wallets()
            )
            ok, reason = self.risk_manager.check_correlation(
                config.instrument, signal.side, position_value, total_capital
            )
            if not ok:
                logger.debug(f"Bot {bot_id} correlation reject: {reason}")
                continue
            
            # ── Simulate entry execution ──
            atr = df_with_indicators.iloc[bar_idx].get('atr', 0) if 'atr' in df_with_indicators.columns else 0
            
            # For next_bar_open mode, we need next bar data
            if config.execution_mode == 'next_bar_open' and bar_idx + 1 < len(df):
                exec_bar = df.iloc[bar_idx + 1]
            else:
                exec_bar = bar
            
            fill = sim.simulate_market_order(
                side=signal.side,
                intended_price=signal.entry_price,
                position_size_units=position_size,
                bar_open=exec_bar.get('open', signal.entry_price),
                bar_high=exec_bar.get('high', signal.entry_price * 1.01),
                bar_low=exec_bar.get('low', signal.entry_price * 0.99),
                bar_close=exec_bar.get('close', signal.entry_price),
                bar_volume=exec_bar.get('volume', 0),
                atr=atr,
                avg_volume_20=avg_vol_20,
            )
            
            # ── Store open trade ──
            self._open_trades[bot_id] = {
                'trade_id': f"PT-{uuid.uuid4().hex[:12]}",
                'bot_id': bot_id,
                'instrument': config.instrument,
                'timeframe': config.timeframe,
                'side': signal.side,
                'order_type': signal.order_type,
                'entry_price': fill.fill_price,
                'sl_price': signal.sl_price,
                'tp_price': signal.tp_price,
                'position_size': position_size,
                'risk_percent': adjusted_risk,
                'slippage_applied': fill.slippage_applied,
                'fees_paid': fill.fees_paid,
                'trade_reason': signal.reason,
                'regime_at_entry': regime.get('trend', ''),
                'indicator_snapshot': signal.indicator_snapshot,
                'timestamp_open': datetime.now(timezone.utc),
                'entry_bar_idx': bar_idx,
            }
            
            # Register position for correlation tracking
            self.risk_manager.register_position(
                config.instrument, signal.side, position_value, bot_id
            )
            
            # Update wallet open positions count
            self.db.update_wallet(bot_id, {
                'open_positions': (wallet.get('open_positions', 0) or 0) + 1,
                'circuit_breaker_active': circuit_breaker,
                'risk_multiplier': adjusted_risk / config.risk_per_trade_pct,
            })
            
            logger.info(
                f"Bot {bot_id}: ENTRY {signal.side.upper()} {config.instrument} "
                f"@ {fill.fill_price:.4f} (intended={signal.entry_price:.4f}, "
                f"slip={fill.slippage_applied:.4f})"
            )
        
        return executed_trades
    
    def _check_exit(
        self,
        bot_id: str,
        df: pd.DataFrame,
        bar_idx: int,
        config: BotConfig,
        sim: TradeSimulator,
        wallet: Dict,
    ) -> Optional[Dict[str, Any]]:
        """Check if an open trade should exit (SL/TP/time)."""
        trade = self._open_trades[bot_id]
        bar = df.iloc[bar_idx]
        
        entry = trade['entry_price']
        sl = trade['sl_price']
        tp = trade['tp_price']
        side = trade['side']
        size = trade['position_size']
        
        bar_high = bar.get('high', 0)
        bar_low = bar.get('low', 0)
        bar_close = bar.get('close', 0)
        
        exit_price = None
        exit_reason = None
        
        # Time-based exit
        bars_held = bar_idx - trade.get('entry_bar_idx', bar_idx)
        if config.max_bars_in_trade and bars_held >= config.max_bars_in_trade:
            exit_price = bar_close
            exit_reason = f"TIME EXIT: {bars_held} bars exceeded max {config.max_bars_in_trade}"
        
        # SL/TP check
        if side == 'buy':
            if bar_low <= sl:
                # SL hit — simulate stop order (with gap risk)
                atr = 0
                if 'atr' in df.columns:
                    atr = df.iloc[bar_idx].get('atr', 0)
                avg_vol = df['volume'].iloc[max(0, bar_idx-20):bar_idx].mean() if bar_idx >= 20 else bar.get('volume', 0)
                
                stop_fill = sim.simulate_stop_order(
                    'sell', sl, size,
                    bar.get('open', sl), bar_high, bar_low, bar_close,
                    bar.get('volume', 0), atr, avg_vol,
                )
                exit_price = stop_fill.fill_price
                trade['slippage_applied'] += stop_fill.slippage_applied
                trade['fees_paid'] += stop_fill.fees_paid
                exit_reason = f"STOP LOSS HIT at {exit_price:.4f} ({stop_fill.result.value})"
            elif bar_high >= tp:
                exit_price = tp
                exit_reason = f"TAKE PROFIT HIT at {tp:.4f}"
        else:  # sell
            if bar_high >= sl:
                atr = 0
                if 'atr' in df.columns:
                    atr = df.iloc[bar_idx].get('atr', 0)
                avg_vol = df['volume'].iloc[max(0, bar_idx-20):bar_idx].mean() if bar_idx >= 20 else bar.get('volume', 0)
                
                stop_fill = sim.simulate_stop_order(
                    'buy', sl, size,
                    bar.get('open', sl), bar_high, bar_low, bar_close,
                    bar.get('volume', 0), atr, avg_vol,
                )
                exit_price = stop_fill.fill_price
                trade['slippage_applied'] += stop_fill.slippage_applied
                trade['fees_paid'] += stop_fill.fees_paid
                exit_reason = f"STOP LOSS HIT at {exit_price:.4f} ({stop_fill.result.value})"
            elif bar_low <= tp:
                exit_price = tp
                exit_reason = f"TAKE PROFIT HIT at {tp:.4f}"
        
        if exit_price is None:
            return None
        
        # ── Calculate PnL ──
        if side == 'buy':
            raw_pnl = (exit_price - entry) * size
        else:
            raw_pnl = (entry - exit_price) * size
        
        net_pnl = raw_pnl - trade['fees_paid'] - trade['slippage_applied']
        
        # R-multiple
        risk_distance = abs(entry - sl)
        r_multiple = net_pnl / (risk_distance * size) if risk_distance > 0 and size > 0 else 0
        
        # Trade result
        if net_pnl > 0:
            result = 'WIN'
        elif net_pnl < 0:
            result = 'LOSS'
        else:
            result = 'BREAKEVEN'
        
        # Build trade record
        trade_record = {
            'trade_id': trade['trade_id'],
            'bot_id': bot_id,
            'instrument': trade['instrument'],
            'timeframe': trade['timeframe'],
            'side': side,
            'order_type': trade['order_type'],
            'entry_price': entry,
            'exit_price': exit_price,
            'sl_price': sl,
            'tp_price': tp,
            'position_size': size,
            'risk_percent': trade['risk_percent'],
            'r_multiple': round(r_multiple, 4),
            'slippage_applied': trade['slippage_applied'],
            'fees_paid': trade['fees_paid'],
            'net_pnl': round(net_pnl, 4),
            'trade_result': result,
            'trade_reason': trade['trade_reason'] + f" | {exit_reason}",
            'regime_at_entry': trade['regime_at_entry'],
            'indicator_snapshot': trade['indicator_snapshot'],
            'timestamp_open': trade['timestamp_open'],
            'timestamp_close': datetime.now(timezone.utc),
            'bars_held': bars_held,
        }
        
        # Validate and clean
        is_valid, reason, cleaned = self.cleaner.validate_and_clean(trade_record)
        if not is_valid:
            logger.warning(f"Trade validation failed for {bot_id}: {reason}")
            del self._open_trades[bot_id]
            return None
        
        # Insert to DB
        self.db.insert_trade(cleaned)
        
        # Update wallet
        current_equity = wallet.get('current_equity', config.virtual_capital) + net_pnl
        peak = max(wallet.get('peak_equity', config.virtual_capital), current_equity)
        dd_pct = ((peak - current_equity) / peak * 100) if peak > 0 else 0
        
        self.db.update_wallet(bot_id, {
            'current_equity': round(current_equity, 2),
            'peak_equity': round(peak, 2),
            'total_pnl': round(wallet.get('total_pnl', 0) + net_pnl, 2),
            'realized_pnl': round(wallet.get('realized_pnl', 0) + net_pnl, 2),
            'current_drawdown_pct': round(dd_pct, 2),
            'open_positions': max(0, (wallet.get('open_positions', 0) or 0) - 1),
        })
        
        # Record for risk tracking
        self.risk_manager.record_daily_pnl(net_pnl)
        self.risk_manager.remove_position(bot_id, config.instrument)
        
        # Record for expectancy gate
        bot = self.bots.get(bot_id)
        if bot:
            bot.record_result(r_multiple)
        
        # Clean up
        del self._open_trades[bot_id]
        
        logger.info(
            f"Bot {bot_id}: EXIT {side.upper()} {config.instrument} "
            f"@ {exit_price:.4f} | PnL={net_pnl:+.2f} | R={r_multiple:+.2f} | {result}"
        )
        
        return trade_record
    
    def run_backtest(
        self,
        df: pd.DataFrame,
        start_idx: int = 0,
    ):
        """
        Run a full backtest across all bars in the DataFrame.
        Call process_bar for each bar sequentially.
        """
        all_trades = []
        
        for idx in range(start_idx, len(df)):
            trades = self.process_bar(df, idx)
            all_trades.extend(trades)
        
        # Force close any remaining open trades at last bar
        for bot_id in list(self._open_trades.keys()):
            config = self.configs.get(bot_id)
            sim = self.simulators.get(bot_id)
            wallet = self.db.get_wallet(bot_id)
            if config and sim and wallet:
                result = self._check_exit(bot_id, df, len(df) - 1, config, sim, wallet)
                if result:
                    all_trades.append(result)
        
        logger.info(f"Backtest complete: {len(all_trades)} trades across {len(self.configs)} bots")
        return all_trades
    
    def update_all_metrics(self):
        """Recalculate and store metrics for all bots."""
        all_metrics = {}
        
        for bot_id, config in self.configs.items():
            trades = self.db.get_all_trades_for_bot(bot_id)
            wallet = self.db.get_wallet(bot_id)
            capital = wallet.get('initial_capital', config.virtual_capital) if wallet else config.virtual_capital
            
            metrics = MetricsCalculator.calculate(trades, capital)
            
            # Monte Carlo
            pnls = [t.get('net_pnl', 0) or 0 for t in trades]
            if len(pnls) >= 5:
                mc_result = self.mc.simulate(pnls, capital)
                metrics.update({
                    'mc_worst_dd': mc_result.get('mc_worst_dd', 0),
                    'mc_95pct_dd': mc_result.get('mc_95pct_dd', 0),
                    'mc_risk_of_ruin': mc_result.get('mc_risk_of_ruin', 0),
                })
            
            # Store
            metrics['bot_name'] = config.name
            metrics['strategy'] = config.strategy
            metrics['instrument'] = config.instrument
            
            # Comparator will add safety label
            all_metrics[bot_id] = metrics
        
        # Comparison & safety classification
        comparison = BotComparator.compare(all_metrics)
        for comp in comparison:
            bot_id = comp['bot_id']
            all_metrics[bot_id]['composite_score'] = comp['composite_score']
            all_metrics[bot_id]['safety_label'] = comp['safety_label']
            all_metrics[bot_id]['is_overfit_flagged'] = comp['is_overfit_flagged']
        
        # Save to DB
        for bot_id, metrics in all_metrics.items():
            self.db.upsert_performance(bot_id, metrics)
        
        logger.info(f"Updated metrics for {len(all_metrics)} bots")
        return all_metrics

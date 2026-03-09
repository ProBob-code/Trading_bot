"""
V2 Paper Trader
================

Institutional-grade paper trading engine for V2.
Fully isolated from V1 — does NOT inherit from V1's PaperTrader.

Features:
- Uses ExecutionEngine for all fill price computation
- Stores leverage, margin, liquidation data per position
- Full execution audit trail in trade history
- Supports cross/isolated margin modes
- Liquidation checks on price updates
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional
import threading
import uuid
from loguru import logger

from .execution_engine import ExecutionEngine, ExecutionResult


class V2Position:
    """A V2 position with leverage and margin tracking."""
    
    __slots__ = [
        'symbol', 'side', 'quantity', 'avg_price', 'leverage',
        'margin_used', 'liquidation_price', 'margin_mode', 'opened_at'
    ]
    
    def __init__(
        self,
        symbol: str,
        side: str,
        quantity: float,
        avg_price: float,
        leverage: float = 1.0,
        margin_mode: str = 'isolated'
    ):
        self.symbol = symbol
        self.side = side.upper()  # 'LONG' or 'SHORT'
        self.quantity = quantity
        self.avg_price = avg_price
        self.leverage = leverage
        self.margin_mode = margin_mode
        self.opened_at = datetime.now(timezone.utc)
        
        # Computed fields
        notional = avg_price * quantity
        self.margin_used = notional / leverage
        self.liquidation_price = self._compute_liquidation()
    
    def _compute_liquidation(self) -> float:
        """Compute liquidation price (futures-style)."""
        if self.leverage <= 1.0:
            return 0.0  # No liquidation at 1× leverage
        if self.side == 'LONG':
            return self.avg_price * (1 - 1 / self.leverage)
        else:
            return self.avg_price * (1 + 1 / self.leverage)
    
    def unrealized_pnl(self, current_price: float) -> float:
        """Compute leveraged unrealized P&L."""
        if self.side == 'LONG':
            base_pnl = (current_price - self.avg_price) * self.quantity
        else:
            base_pnl = (self.avg_price - current_price) * self.quantity
        return base_pnl * self.leverage
    
    def unrealized_pnl_pct(self, current_price: float) -> float:
        """Compute unrealized P&L as percentage of margin."""
        if self.margin_used <= 0:
            return 0.0
        return (self.unrealized_pnl(current_price) / self.margin_used) * 100
    
    def is_liquidated(self, current_price: float) -> bool:
        """Check if position should be liquidated."""
        if self.leverage <= 1.0:
            return False
        if self.side == 'LONG':
            return current_price <= self.liquidation_price
        else:
            return current_price >= self.liquidation_price
    
    def to_dict(self, current_price: float = None) -> dict:
        price = current_price or self.avg_price
        return {
            'symbol': self.symbol,
            'side': self.side,
            'quantity': self.quantity,
            'avg_price': self.avg_price,
            'current_price': price,
            'leverage': self.leverage,
            'margin_used': round(self.margin_used, 2),
            'liquidation_price': round(self.liquidation_price, 2),
            'margin_mode': self.margin_mode,
            'market_value': round(price * self.quantity, 2),
            'unrealized_pnl': round(self.unrealized_pnl(price), 2),
            'unrealized_pnl_pct': round(self.unrealized_pnl_pct(price), 2),
            'opened_at': self.opened_at.isoformat(),
        }


class V2Account:
    """Isolated account state for V2 paper trading."""
    
    def __init__(self, initial_capital: float = 100000):
        self.initial_capital = initial_capital
        self.cash = initial_capital
        self.positions: Dict[str, V2Position] = {}  # symbol → V2Position
        self.closed_trades: List[Dict] = []
        self.trade_history: List[Dict] = []  # Full execution audit trail


class PaperTraderV2:
    """
    V2 institutional paper trading engine.
    
    - Backend authoritative fill prices via ExecutionEngine
    - Leveraged positions with margin and liquidation
    - Full audit trail per trade
    - Thread-safe multi-account
    """
    
    def __init__(
        self,
        initial_capital: float = 100000,
        execution_engine: Optional[ExecutionEngine] = None,
        deterministic: bool = False
    ):
        self.initial_capital = initial_capital
        self.execution_engine = execution_engine or ExecutionEngine(deterministic=deterministic)
        self.lock = threading.Lock()
        
        # Multi-account state
        self.accounts: Dict[int, V2Account] = {}
        
        # Live prices
        self.current_prices: Dict[str, float] = {}
        
        logger.info(f"[V2-TRADER] PaperTraderV2 initialized (capital=${initial_capital:,.0f})")
    
    def _get_account(self, user_id: Optional[int]) -> V2Account:
        """Get or create user account."""
        uid = user_id if user_id is not None else 0
        if uid not in self.accounts:
            self.accounts[uid] = V2Account(self.initial_capital)
        return self.accounts[uid]
    
    def set_prices(self, prices: Dict[str, float]):
        """Update live prices and check liquidations."""
        with self.lock:
            self.current_prices.update(prices)
            self._check_liquidations()
    
    def _check_liquidations(self):
        """Auto-close positions that breach liquidation price."""
        for uid, acc in self.accounts.items():
            liquidated = []
            for symbol, pos in acc.positions.items():
                price = self.current_prices.get(symbol)
                if price and pos.is_liquidated(price):
                    liquidated.append((symbol, pos, price))
            
            for symbol, pos, price in liquidated:
                self._force_close(acc, symbol, pos, price, reason='LIQUIDATION')
                logger.warning(
                    f"[V2-TRADER] ⚠️ LIQUIDATED user {uid}: {pos.side} {pos.quantity} "
                    f"{symbol} @ {price:.4f} (liq={pos.liquidation_price:.4f})"
                )
    
    def _force_close(self, acc: V2Account, symbol: str, pos: V2Position, price: float, reason: str = 'FORCED'):
        """Force-close a position (liquidation or panic)."""
        realized_pnl = pos.unrealized_pnl(price)
        
        # Return margin + P&L to cash
        acc.cash += pos.margin_used + realized_pnl
        
        closed_record = {
            'symbol': symbol,
            'side': pos.side,
            'quantity': pos.quantity,
            'entry_price': pos.avg_price,
            'exit_price': price,
            'leverage': pos.leverage,
            'realized_pnl': round(realized_pnl, 2),
            'reason': reason,
            'closed_at': datetime.now().isoformat(),
        }
        acc.closed_trades.append(closed_record)
        acc.trade_history.append({
            **closed_record,
            'type': 'CLOSE',
            'commission': 0,  # Liquidation = no commission
        })
        
        del acc.positions[symbol]
    
    def execute_trade(
        self,
        user_id: int,
        symbol: str,
        side: str,
        quantity: float,
        leverage: float = 1.0,
        volatility: float = 0.02,
        volume: float = 100e6,
        strategy: str = 'manual',
        margin_mode: str = 'isolated',
    ) -> Dict:
        """
        Execute a V2 trade.
        
        Returns full execution result with fill_price, commission, spread, slippage.
        """
        with self.lock:
            acc = self._get_account(user_id)
            market_price = self.current_prices.get(symbol)
            
            if market_price is None or market_price <= 0:
                return {'success': False, 'error': f'No price available for {symbol}'}
            
            side_upper = side.upper()
            
            # Determine if this is an open or close
            existing_pos = acc.positions.get(symbol)
            
            if existing_pos and self._is_closing(existing_pos, side_upper):
                return self._close_position(acc, user_id, symbol, quantity, existing_pos,
                                            market_price, volatility, volume, strategy)
            else:
                return self._open_position(acc, user_id, symbol, side_upper, quantity,
                                           leverage, market_price, volatility, volume,
                                           strategy, margin_mode)
    
    def _is_closing(self, pos: V2Position, side: str) -> bool:
        """Check if this order closes the existing position."""
        if pos.side == 'LONG' and side == 'SELL':
            return True
        if pos.side == 'SHORT' and side == 'BUY':
            return True
        return False
    
    def _generate_trade_id(self, symbol: str, trade_type: str) -> str:
        """Generate unique trade ID: {SYMBOL}_{YYYYMMDD_HHMMSS_ffffff}_{TYPE}_{uuid8}"""
        ts = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')
        uid = uuid.uuid4().hex[:8]
        return f"{symbol}_{ts}_{trade_type}_{uid}".upper()

    def _open_position(
        self, acc, user_id, symbol, side, quantity, leverage,
        market_price, volatility, volume, strategy, margin_mode
    ) -> Dict:
        """Open a new position or add to existing."""
        # Execute through engine
        exec_side = 'BUY' if side in ('BUY', 'LONG') else 'SELL'
        result: ExecutionResult = self.execution_engine.execute(
            exec_side, market_price, quantity, volatility, volume
        )
        
        # Check margin requirements
        notional = result.fill_price * quantity
        required_margin = notional / leverage
        total_cost = required_margin + result.commission
        
        if total_cost > acc.cash:
            return {
                'success': False,
                'error': f'Insufficient margin: need ${total_cost:.2f}, have ${acc.cash:.2f}'
            }
        
        # Deduct margin + commission from cash
        acc.cash -= total_cost
        
        # Map side to position side
        pos_side = 'LONG' if exec_side == 'BUY' else 'SHORT'
        
        if symbol in acc.positions:
            # Add to existing position (average price)
            pos = acc.positions[symbol]
            old_notional = pos.avg_price * pos.quantity
            new_notional = result.fill_price * quantity
            total_qty = pos.quantity + quantity
            pos.avg_price = (old_notional + new_notional) / total_qty
            pos.quantity = total_qty
            pos.margin_used += required_margin
            pos.liquidation_price = pos._compute_liquidation()
        else:
            # Create new position
            acc.positions[symbol] = V2Position(
                symbol=symbol,
                side=pos_side,
                quantity=quantity,
                avg_price=result.fill_price,
                leverage=leverage,
                margin_mode=margin_mode,
            )
        
        # Generate trade ID
        trade_id = self._generate_trade_id(symbol, 'OPEN')

        # Audit trail
        trade_record = {
            'type': 'OPEN',
            'trade_id': trade_id,
            'symbol': symbol,
            'side': exec_side,
            'position_side': pos_side,
            'quantity': quantity,
            'strategy': strategy,
            'leverage': leverage,
            'margin_mode': margin_mode,
            'user_id': user_id,
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **result.to_dict(),
        }
        acc.trade_history.append(trade_record)
        
        logger.info(
            f"[V2-TRADER] {'🟢' if exec_side == 'BUY' else '🔴'} OPEN {pos_side}: "
            f"user {user_id} | {quantity} {symbol} @ {result.fill_price:.4f} "
            f"(lev={leverage}×, margin=${required_margin:.2f}) [{trade_id}]"
        )
        
        return {
            'success': True,
            'type': 'OPEN',
            'trade_id': trade_id,
            'symbol': symbol,
            'side': exec_side,
            'position_side': pos_side,
            'quantity': quantity,
            'strategy': strategy,
            'leverage': leverage,
            **result.to_dict(),
        }
    
    def _close_position(
        self, acc, user_id, symbol, quantity, pos,
        market_price, volatility, volume, strategy
    ) -> Dict:
        """Close an existing position (full or partial)."""
        close_qty = min(quantity, pos.quantity)
        
        # Execute through engine
        exec_side = 'SELL' if pos.side == 'LONG' else 'BUY'
        result: ExecutionResult = self.execution_engine.execute(
            exec_side, market_price, close_qty, volatility, volume
        )
        
        # Calculate leveraged P&L
        if pos.side == 'LONG':
            base_pnl = (result.fill_price - pos.avg_price) * close_qty
        else:
            base_pnl = (pos.avg_price - result.fill_price) * close_qty
        
        realized_pnl = base_pnl * pos.leverage
        net_pnl = realized_pnl - result.commission
        
        # Return margin proportionally + P&L
        margin_returned = pos.margin_used * (close_qty / pos.quantity)
        acc.cash += margin_returned + realized_pnl - result.commission
        
        # Update or remove position
        if close_qty >= pos.quantity:
            del acc.positions[symbol]
        else:
            pos.quantity -= close_qty
            pos.margin_used -= margin_returned
        
        # Generate trade ID and compute duration
        trade_id = self._generate_trade_id(symbol, 'CLOSE')
        exit_time = datetime.now(timezone.utc)
        duration_seconds = (exit_time - pos.opened_at).total_seconds()

        # Record closed trade
        closed_record = {
            'type': 'CLOSE',
            'trade_id': trade_id,
            'symbol': symbol,
            'side': exec_side,
            'position_side': pos.side,
            'direction': pos.side,
            'quantity': close_qty,
            'entry_price': pos.avg_price,
            'exit_price': result.fill_price,
            'leverage': pos.leverage,
            'realized_pnl': round(realized_pnl, 2),
            'net_pnl': round(net_pnl, 2),
            'strategy': strategy,
            'user_id': user_id,
            'duration_seconds': round(duration_seconds, 2),
            'timestamp': exit_time.isoformat(),
            **result.to_dict(),
        }
        acc.trade_history.append(closed_record)
        acc.closed_trades.append(closed_record)
        
        pnl_emoji = '💰' if realized_pnl >= 0 else '💸'
        logger.info(
            f"[V2-TRADER] {pnl_emoji} CLOSE {pos.side}: user {user_id} | "
            f"{close_qty} {symbol} @ {result.fill_price:.4f} | "
            f"P&L=${realized_pnl:.2f} (lev={pos.leverage}×) [{trade_id}]"
        )
        
        return {
            'success': True,
            'type': 'CLOSE',
            'trade_id': trade_id,
            'symbol': symbol,
            'side': exec_side,
            'position_side': pos.side,
            'direction': pos.side,
            'quantity': close_qty,
            'entry_price': pos.avg_price,
            'exit_price': result.fill_price,
            'leverage': pos.leverage,
            'realized_pnl': round(realized_pnl, 2),
            'net_pnl': round(net_pnl, 2),
            'duration_seconds': round(duration_seconds, 2),
            **result.to_dict(),
        }
    
    def get_positions(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all V2 positions with leverage/margin data."""
        with self.lock:
            acc = self._get_account(user_id)
            return [
                pos.to_dict(self.current_prices.get(pos.symbol))
                for pos in acc.positions.values()
            ]
    
    def get_account_info(self, user_id: Optional[int] = None) -> Dict:
        """Get V2 account info with margin tracking."""
        with self.lock:
            acc = self._get_account(user_id)
            
            # Total margin used across all positions
            total_margin_used = sum(p.margin_used for p in acc.positions.values())
            
            # Total unrealized P&L
            total_unrealized = sum(
                p.unrealized_pnl(self.current_prices.get(p.symbol, p.avg_price))
                for p in acc.positions.values()
            )
            
            # Equity = cash + margin_in_positions + unrealized
            equity = acc.cash + total_margin_used + total_unrealized
            
            return {
                'account_id': f'V2_PAPER_{user_id or 0}',
                'initial_capital': acc.initial_capital,
                'cash': round(acc.cash, 2),
                'equity': round(equity, 2),
                'total_value': round(equity, 2),  # Alias for frontend
                'margin_used': round(total_margin_used, 2),
                'available_margin': round(acc.cash, 2),
                'buying_power': round(acc.cash, 2),  # Alias for frontend
                'unrealized_pnl': round(total_unrealized, 2),
                'realized_pnl': round(equity - acc.initial_capital - total_unrealized, 2),
                'total_pnl': round(equity - acc.initial_capital, 2),
                'pnl': round(equity - acc.initial_capital, 2),  # Alias for frontend
                'pnl_pct': round(((equity - acc.initial_capital) / acc.initial_capital) * 100, 4)
                    if acc.initial_capital > 0 else 0,
                'pnl_percentage': round(((equity - acc.initial_capital) / acc.initial_capital) * 100, 4)
                    if acc.initial_capital > 0 else 0,
                'open_positions': len(acc.positions),
                'total_trades': len(acc.trade_history),
                'mode': 'FUTURES_SIMULATION',
            }
    
    def get_trade_history(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get full V2 trade audit trail."""
        return self._get_account(user_id).trade_history
    
    def get_closed_trades(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get closed trade records."""
        return self._get_account(user_id).closed_trades
    
    def panic_close_all(self, user_id: int) -> Dict:
        """Force-close all positions for a user."""
        with self.lock:
            acc = self._get_account(user_id)
            closed_count = 0
            
            for symbol in list(acc.positions.keys()):
                pos = acc.positions[symbol]
                price = self.current_prices.get(symbol, pos.avg_price)
                self._force_close(acc, symbol, pos, price, reason='PANIC_CLOSE')
                closed_count += 1
            
            return {
                'success': True,
                'closed_count': closed_count,
                'cash': round(acc.cash, 2),
            }
    
    def reset(self, user_id: Optional[int] = None):
        """Reset V2 account state."""
        with self.lock:
            if user_id is not None:
                if user_id in self.accounts:
                    self.accounts[user_id] = V2Account(self.initial_capital)
                    logger.info(f"[V2-TRADER] Reset account for user {user_id}")
            else:
                self.accounts.clear()
                logger.info("[V2-TRADER] Reset all accounts")

    def save_positions(self, user_id: int, db_manager):
        """Persist all open positions to v2_positions table."""
        with self.lock:
            acc = self._get_account(user_id)
            for symbol, pos in acc.positions.items():
                db_manager.v2_save_position(user_id, {
                    'symbol': symbol,
                    'side': pos.side,
                    'quantity': pos.quantity,
                    'avg_price': pos.avg_price,
                    'leverage': pos.leverage,
                    'margin_mode': pos.margin_mode,
                    'margin_used': pos.margin_used,
                    'strategy': getattr(pos, 'strategy', None),
                    'opened_at': pos.opened_at.isoformat() if pos.opened_at else None,
                })
            logger.info(f"[V2-TRADER] Saved {len(acc.positions)} positions for user {user_id}")

    def load_positions(self, user_id: int, db_manager):
        """Load open positions from v2_positions table (for restart recovery)."""
        with self.lock:
            rows = db_manager.v2_get_positions(user_id)
            if not rows:
                return
            acc = self._get_account(user_id)
            for row in rows:
                symbol = row['symbol']
                pos = V2Position(
                    symbol=symbol,
                    side=row['side'],
                    quantity=row['quantity'],
                    avg_price=row['avg_price'],
                    leverage=row.get('leverage', 1.0),
                    margin_mode=row.get('margin_mode', 'isolated'),
                )
                pos.margin_used = row.get('margin_used', 0)
                if row.get('opened_at'):
                    try:
                        pos.opened_at = datetime.fromisoformat(str(row['opened_at']))
                    except (ValueError, TypeError):
                        pos.opened_at = datetime.now(timezone.utc)
                acc.positions[symbol] = pos
            logger.info(f"[V2-TRADER] Loaded {len(rows)} positions for user {user_id} from DB")

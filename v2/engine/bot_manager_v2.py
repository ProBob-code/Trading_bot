"""
V2 Bot Manager
================

Multi-strategy portfolio bot management for V2.

Key differences from V1:
- Bot ID includes strategy + config hash
- Multiple bots per symbol allowed (different strategies/configs)
- Duplicate config hash rejected
- Leverage stored per bot
"""

import hashlib
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Optional, List
from loguru import logger
from shared.services.system_state import get_system_state
from shared.database.db_manager import db_manager


class V2TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class V2BotStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    DISABLED = "disabled"  # Auto-disabled by drawdown limit
    ERROR = "error"


@dataclass
class V2BotConfig:
    """V2 bot configuration with leverage and risk parameters."""
    user_id: int
    symbol: str
    market: str
    strategy: str
    mode: V2TradingMode = V2TradingMode.PAPER
    interval: str = "1m"
    position_size: float = 10.0
    stop_loss: float = 5.0
    take_profit: float = 10.0
    max_quantity: float = 1.0
    leverage: float = 1.0
    risk_pct: float = 2.0
    
    def config_hash(self) -> str:
        """
        Compute deterministic hash of configuration.
        
        Includes: strategy, symbol, interval, stop_loss, take_profit, leverage, risk_pct
        """
        hash_input = (
            f"{self.strategy}:{self.symbol}:{self.interval}:"
            f"{self.stop_loss}:{self.take_profit}:{self.leverage}:{self.risk_pct}"
        )
        return hashlib.sha256(hash_input.encode()).hexdigest()[:8]


@dataclass
class V2BotStats:
    """V2 bot statistics."""
    total_trades: int = 0
    buy_trades: int = 0
    sell_trades: int = 0
    realized_pnl: float = 0.0
    unrealized_pnl: float = 0.0
    total_pnl: float = 0.0
    signals_generated: int = 0
    last_signal: str = "HOLD"
    last_price: float = 0.0


@dataclass
class V2TradingBot:
    """V2 trading bot instance."""
    user_id: int
    bot_id: str
    config: V2BotConfig
    config_hash: str
    status: V2BotStatus = V2BotStatus.STOPPED
    stats: V2BotStats = field(default_factory=V2BotStats)
    start_time: Optional[datetime] = None
    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    disable_reason: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {
            'user_id': self.user_id,
            'bot_id': self.bot_id,
            'config_hash': self.config_hash,
            'symbol': self.config.symbol,
            'market': self.config.market,
            'strategy': self.config.strategy,
            'mode': self.config.mode.value,
            'interval': self.config.interval,
            'leverage': self.config.leverage,
            'risk_pct': self.config.risk_pct,
            'status': self.status.value,
            'disable_reason': self.disable_reason,
            'stats': {
                'total_trades': self.stats.total_trades,
                'buy_trades': self.stats.buy_trades,
                'sell_trades': self.stats.sell_trades,
                'total_pnl': self.stats.total_pnl,
                'realized_pnl': self.stats.realized_pnl,
                'unrealized_pnl': self.stats.unrealized_pnl,
                'signals_generated': self.stats.signals_generated,
                'last_signal': self.stats.last_signal,
                'last_price': self.stats.last_price,
            },
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
        }


class BotManagerV2:
    """
    V2 Bot Manager — multi-strategy portfolio management.
    
    Features:
    - Bot IDs include strategy + config hash
    - Multiple strategies per symbol
    - Duplicate config hash rejection
    - Auto-disable on drawdown limit breach
    """
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.bots: Dict[str, V2TradingBot] = {}
        self.lock = threading.Lock()
        self._initialized = True
        logger.info("[V2-BOTMGR] BotManagerV2 initialized")
    
    def generate_bot_id(self, user_id: int, market: str, symbol: str, strategy: str, config_hash: str) -> str:
        """
        Generate V2 bot ID.
        
        Format: u{user_id}_{market}_{symbol}_{strategy}_{config_hash}
        """
        return f"u{user_id}_{market}_{symbol}_{strategy}_{config_hash}".lower()
    
    def start_bot(
        self,
        user_id: int,
        symbol: str,
        market: str,
        strategy: str,
        mode: str = "paper",
        interval: str = "1m",
        trade_function=None,
        **kwargs
    ) -> Dict:
        """Start a V2 bot with config hash validation."""
        
        config = V2BotConfig(
            user_id=user_id,
            symbol=symbol,
            market=market,
            strategy=strategy,
            mode=V2TradingMode(mode),
            interval=interval,
            position_size=kwargs.get('position_size', 10.0),
            stop_loss=kwargs.get('stop_loss', 5.0),
            take_profit=kwargs.get('take_profit', 10.0),
            max_quantity=kwargs.get('max_quantity', 1.0),
            leverage=kwargs.get('leverage', 1.0),
            risk_pct=kwargs.get('risk_pct', 2.0),
        )
        
        c_hash = config.config_hash()
        bot_id = self.generate_bot_id(user_id, market, symbol, strategy, c_hash)
        
        # ── Lazy Session Initialization ──
        sys_state = get_system_state()
        if not sys_state.get_session_id():
            session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            sys_state.set_session_id(session_id)
            db_manager.v2_create_session(session_id, sys_state.get_engine_version())
            db_manager.v2_update_session_status(session_id, "ACTIVE")
            logger.info(f"[V2-BOTMGR] 🆕 Authorized NEW institutional session: {session_id}")
        
        with self.lock:
            # Check for duplicate config hash
            if bot_id in self.bots and self.bots[bot_id].status == V2BotStatus.RUNNING:
                return {
                    'success': False,
                    'error': f'Bot with identical config already running (hash={c_hash})',
                    'bot_id': bot_id,
                }
            
            bot = V2TradingBot(
                user_id=user_id,
                bot_id=bot_id,
                config=config,
                config_hash=c_hash,
                status=V2BotStatus.RUNNING,
                start_time=datetime.now(),
                stop_flag=threading.Event(),
            )
            
            bot._trade_function = trade_function
            self.bots[bot_id] = bot
            
            logger.info(f"[V2-BOTMGR] 🚀 Bot started: {bot_id} (strategy={strategy}, hash={c_hash})")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'config_hash': c_hash,
                'message': f'V2 Bot {bot_id} started with {strategy}',
            }
    
    def stop_bot(self, bot_id: str) -> Dict:
        """Stop a V2 bot."""
        with self.lock:
            if bot_id not in self.bots:
                return {'success': False, 'error': 'Bot not found'}
            
            bot = self.bots[bot_id]
            bot.stop_flag.set()
            bot.status = V2BotStatus.STOPPED
            
            if bot.thread and bot.thread.is_alive():
                bot.thread.join(timeout=5)
            
            del self.bots[bot_id]
            logger.info(f"[V2-BOTMGR] 🛑 Bot stopped: {bot_id}")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'final_stats': bot.stats.__dict__,
            }
    
    def disable_bot(self, bot_id: str, reason: str = "Max drawdown breached") -> Dict:
        """Auto-disable a bot (drawdown limit breach)."""
        with self.lock:
            if bot_id not in self.bots:
                return {'success': False, 'error': 'Bot not found'}
            
            bot = self.bots[bot_id]
            bot.stop_flag.set()
            bot.status = V2BotStatus.DISABLED
            bot.disable_reason = reason
            
            logger.warning(f"[V2-BOTMGR] ⚠️ Bot DISABLED: {bot_id} — {reason}")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'status': 'disabled',
                'reason': reason,
            }
    
    def get_all_bots(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get all V2 bots, optionally filtered by user."""
        if user_id is not None:
            return [b.to_dict() for b in self.bots.values() if b.user_id == user_id]
        return [b.to_dict() for b in self.bots.values()]
    
    def get_bot(self, bot_id: str) -> Optional[Dict]:
        """Get single bot status."""
        if bot_id not in self.bots:
            return None
        return self.bots[bot_id].to_dict()
    
    def get_running_bots(self, user_id: Optional[int] = None) -> List[Dict]:
        """Get only running V2 bots."""
        return [
            b.to_dict() for b in self.bots.values()
            if b.status == V2BotStatus.RUNNING and (user_id is None or b.user_id == user_id)
        ]
    
    def increment_trades(self, bot_id: str, side: str, pnl: float = 0):
        """Update bot trade statistics."""
        if bot_id in self.bots:
            stats = self.bots[bot_id].stats
            stats.total_trades += 1
            if side.lower() == 'buy':
                stats.buy_trades += 1
            else:
                stats.sell_trades += 1
            stats.realized_pnl += pnl
            stats.total_pnl = stats.realized_pnl + stats.unrealized_pnl
    
    def update_bot_stats(self, bot_id: str, **kwargs):
        """Update bot statistics."""
        if bot_id in self.bots:
            stats = self.bots[bot_id].stats
            for key, value in kwargs.items():
                if hasattr(stats, key):
                    setattr(stats, key, value)
    
    def stop_all(self):
        """Stop all V2 bots."""
        with self.lock:
            for bot_id in list(self.bots.keys()):
                bot = self.bots[bot_id]
                bot.stop_flag.set()
                bot.status = V2BotStatus.STOPPED
                del self.bots[bot_id]
            logger.info("[V2-BOTMGR] 🛑 All V2 bots stopped")


# Singleton
bot_manager_v2 = BotManagerV2()

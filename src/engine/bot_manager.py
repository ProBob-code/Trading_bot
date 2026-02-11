"""
Bot Manager - Multi-Bot Trading Architecture
============================================

Manages multiple parallel trading bots that:
- Run in separate threads
- Persist across page refreshes
- Support strategy hot-swapping
- Handle both Paper and Live trading modes
"""

import threading
import time
import json
from pathlib import Path
from typing import Dict, Optional, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from loguru import logger

# Path to persist bot configurations
BOT_CONFIGS_FILE = Path(__file__).parent.parent.parent / "bot_configs.json"


class TradingMode(Enum):
    PAPER = "paper"
    LIVE = "live"


class BotStatus(Enum):
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"


@dataclass
class BotConfig:
    """Configuration for a trading bot."""
    symbol: str
    market: str  # 'crypto' or 'stocks'
    strategy: str
    mode: TradingMode = TradingMode.PAPER
    interval: str = "1m"
    position_size: float = 10.0  # % of balance
    stop_loss: float = 5.0
    take_profit: float = 10.0
    max_quantity: float = 1.0  # Maximum quantity per trade
    auto_restart_enabled: bool = True  # Auto-restart on failure or server restart


@dataclass
class BotStats:
    """Statistics for a trading bot."""
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
class TradingBot:
    """Individual trading bot instance."""
    bot_id: str
    config: BotConfig
    status: BotStatus = BotStatus.STOPPED
    stats: BotStats = field(default_factory=BotStats)
    start_time: Optional[datetime] = None
    thread: Optional[threading.Thread] = None
    stop_flag: threading.Event = field(default_factory=threading.Event)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for API response."""
        return {
            'bot_id': self.bot_id,
            'symbol': self.config.symbol,
            'market': self.config.market,
            'strategy': self.config.strategy,
            'mode': self.config.mode.value,
            'interval': self.config.interval,
            'status': self.status.value,
            'stats': {
                'total_trades': self.stats.total_trades,
                'buy_trades': self.stats.buy_trades,
                'sell_trades': self.stats.sell_trades,
                'total_pnl': self.stats.total_pnl,
                'signals_generated': self.stats.signals_generated,
                'last_signal': self.stats.last_signal,
                'last_price': self.stats.last_price
            },
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'uptime_seconds': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }


class BotManager:
    """
    Central manager for all trading bots.
    
    Features:
    - Start/stop multiple bots
    - Strategy hot-swapping
    - Server-side persistence (survives page refresh)
    - Mode switching (paper/live)
    """
    
    _instance = None
    
    def __new__(cls):
        """Singleton pattern - only one BotManager instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self.bots: Dict[str, TradingBot] = {}
        self.lock = threading.Lock()
        self._initialized = True
        logger.info("🤖 BotManager initialized")
    
    def generate_bot_id(self, market: str, symbol: str) -> str:
        """Generate unique bot ID."""
        return f"{market}_{symbol}".lower()
    
    def start_bot(
        self,
        symbol: str,
        market: str,
        strategy: str,
        mode: str = "paper",
        interval: str = "1m",
        trade_function = None,
        **kwargs
    ) -> Dict:
        """
        Start a new trading bot.
        
        Returns:
            Dict with bot_id and status
        """
        bot_id = self.generate_bot_id(market, symbol)
        
        with self.lock:
            # Check if bot already exists
            if bot_id in self.bots and self.bots[bot_id].status == BotStatus.RUNNING:
                return {'success': False, 'error': 'Bot already running', 'bot_id': bot_id}
            
            # Create bot config
            config = BotConfig(
                symbol=symbol,
                market=market,
                strategy=strategy,
                mode=TradingMode(mode),
                interval=interval,
                position_size=kwargs.get('position_size', 10.0),
                stop_loss=kwargs.get('stop_loss', 5.0),
                take_profit=kwargs.get('take_profit', 10.0),
                max_quantity=kwargs.get('max_quantity', 1.0)
            )
            
            # Create or update bot
            bot = TradingBot(
                bot_id=bot_id,
                config=config,
                status=BotStatus.RUNNING,
                start_time=datetime.now(),
                stop_flag=threading.Event()
            )
            
            # Store trade function reference
            bot._trade_function = trade_function
            
            self.bots[bot_id] = bot
            
            logger.info(f"🚀 Bot started: {bot_id} ({strategy})")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'message': f'Bot {bot_id} started with {strategy}'
            }
    
    def stop_bot(self, bot_id: str) -> Dict:
        """Stop a running bot."""
        with self.lock:
            if bot_id not in self.bots:
                return {'success': False, 'error': 'Bot not found'}
            
            bot = self.bots[bot_id]
            bot.stop_flag.set()
            bot.status = BotStatus.STOPPED
            
            # Remove from bots dictionary so it disappears from active lists
            del self.bots[bot_id]
            
            logger.info(f"🛑 Bot stopped and removed: {bot_id}")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'message': f'Bot {bot_id} stopped and removed',
                'final_stats': bot.stats.__dict__
            }
    
    def update_strategy(self, bot_id: str, new_strategy: str) -> Dict:
        """Hot-swap strategy for a running bot."""
        with self.lock:
            if bot_id not in self.bots:
                return {'success': False, 'error': 'Bot not found'}
            
            bot = self.bots[bot_id]
            old_strategy = bot.config.strategy
            bot.config.strategy = new_strategy
            
            logger.info(f"🔄 Strategy changed: {bot_id} ({old_strategy} → {new_strategy})")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'old_strategy': old_strategy,
                'new_strategy': new_strategy,
                'message': f'Strategy changed to {new_strategy}'
            }
    
    def update_mode(self, bot_id: str, new_mode: str) -> Dict:
        """Switch trading mode (paper/live)."""
        with self.lock:
            if bot_id not in self.bots:
                return {'success': False, 'error': 'Bot not found'}
            
            bot = self.bots[bot_id]
            old_mode = bot.config.mode.value
            bot.config.mode = TradingMode(new_mode)
            
            logger.info(f"💱 Mode changed: {bot_id} ({old_mode} → {new_mode})")
            
            return {
                'success': True,
                'bot_id': bot_id,
                'old_mode': old_mode,
                'new_mode': new_mode
            }
    
    def get_bot(self, bot_id: str) -> Optional[Dict]:
        """Get single bot status."""
        if bot_id not in self.bots:
            return None
        return self.bots[bot_id].to_dict()
    
    def get_all_bots(self) -> List[Dict]:
        """Get all bots status."""
        return [bot.to_dict() for bot in self.bots.values()]
    
    def get_running_bots(self) -> List[Dict]:
        """Get only running bots."""
        return [
            bot.to_dict() 
            for bot in self.bots.values() 
            if bot.status == BotStatus.RUNNING
        ]
    
    def update_bot_stats(self, bot_id: str, **kwargs):
        """Update bot statistics."""
        if bot_id in self.bots:
            stats = self.bots[bot_id].stats
            for key, value in kwargs.items():
                if hasattr(stats, key):
                    setattr(stats, key, value)
    
    def increment_trades(self, bot_id: str, side: str, pnl: float = 0):
        """Increment trade counters."""
        if bot_id in self.bots:
            stats = self.bots[bot_id].stats
            stats.total_trades += 1
            if side.lower() == 'buy':
                stats.buy_trades += 1
            else:
                stats.sell_trades += 1
            stats.total_pnl += pnl
    
    def is_running(self, bot_id: str) -> bool:
        """Check if bot is running."""
        if bot_id not in self.bots:
            return False
        return self.bots[bot_id].status == BotStatus.RUNNING
    
    def stop_all(self):
        """Stop all running bots."""
        for bot_id in list(self.bots.keys()):
            self.stop_bot(bot_id)
        logger.info("🛑 All bots stopped")
    
    def save_configs(self):
        """Save all bot configurations to disk for persistence."""
        configs = []
        for bot_id, bot in self.bots.items():
            if bot.config.auto_restart_enabled:
                configs.append({
                    'bot_id': bot_id,
                    'symbol': bot.config.symbol,
                    'market': bot.config.market,
                    'strategy': bot.config.strategy,
                    'mode': bot.config.mode.value,
                    'interval': bot.config.interval,
                    'position_size': bot.config.position_size,
                    'stop_loss': bot.config.stop_loss,
                    'take_profit': bot.config.take_profit,
                    'max_quantity': bot.config.max_quantity,
                    'auto_restart_enabled': bot.config.auto_restart_enabled
                })
        
        try:
            with open(BOT_CONFIGS_FILE, 'w') as f:
                json.dump(configs, f, indent=4)
            logger.info(f"💾 Saved {len(configs)} bot configs to disk")
        except Exception as e:
            logger.error(f"Error saving bot configs: {e}")
    
    def load_configs(self) -> List[Dict]:
        """Load saved bot configurations from disk."""
        if not BOT_CONFIGS_FILE.exists():
            logger.info("No saved bot configs found")
            return []
        
        try:
            with open(BOT_CONFIGS_FILE, 'r') as f:
                configs = json.load(f)
            logger.info(f"📂 Loaded {len(configs)} bot configs from disk")
            return configs
        except Exception as e:
            logger.error(f"Error loading bot configs: {e}")
            return []
    
    def clear_saved_configs(self):
        """Clear saved bot configurations."""
        if BOT_CONFIGS_FILE.exists():
            BOT_CONFIGS_FILE.unlink()
            logger.info("🗑️ Cleared saved bot configs")


# Singleton instance
bot_manager = BotManager()


def get_bot_manager() -> BotManager:
    """Get the singleton BotManager instance."""
    return bot_manager

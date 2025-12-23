"""
Configuration Settings Manager
==============================

Handles loading and accessing configuration from YAML files and environment variables.
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional
from dataclasses import dataclass, field
import yaml
from dotenv import load_dotenv
from loguru import logger


# Load environment variables
load_dotenv()


@dataclass
class TechnicalConfig:
    """Technical analysis parameters."""
    ma_fast: int = 9
    ma_slow: int = 21
    ma_signal: int = 44
    rsi_period: int = 14
    rsi_overbought: int = 70
    rsi_oversold: int = 30
    bb_period: int = 20
    bb_std: float = 2.0
    atr_period: int = 14


@dataclass
class IchimokuConfig:
    """Ichimoku Cloud parameters."""
    conversion_period: int = 9
    base_period: int = 26
    span_b_period: int = 52


@dataclass
class MLConfig:
    """Machine Learning parameters."""
    model_type: str = "xgboost"
    lookback_period: int = 60
    prediction_horizon: int = 5
    retrain_interval: str = "weekly"
    confidence_threshold: float = 0.6


@dataclass
class StopLossConfig:
    """Stop loss configuration."""
    type: str = "atr"
    atr_multiplier: float = 2.0
    fixed_pct: float = 2.0
    trailing_pct: float = 1.0


@dataclass
class TakeProfitConfig:
    """Take profit configuration."""
    type: str = "atr"
    atr_multiplier: float = 3.0
    fixed_pct: float = 4.0
    risk_reward_ratio: float = 2.0


@dataclass
class RiskConfig:
    """Risk management configuration."""
    position_sizing: str = "fixed_fractional"
    max_position_pct: float = 10.0
    max_daily_loss_pct: float = 2.0
    max_drawdown_pct: float = 10.0
    max_positions: int = 5
    stop_loss: StopLossConfig = field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = field(default_factory=TakeProfitConfig)


@dataclass
class BacktestConfig:
    """Backtesting configuration."""
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 100000
    commission: float = 0.001
    slippage: float = 0.0005


@dataclass 
class BrokerConfig:
    """Broker configuration."""
    name: str = "paper"
    api_key: str = ""
    secret_key: str = ""
    base_url: str = ""


class Settings:
    """
    Main settings class that loads and provides access to all configuration.
    
    Usage:
        settings = Settings()
        print(settings.mode)
        print(settings.risk.max_drawdown_pct)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize settings from configuration file.
        
        Args:
            config_path: Path to config.yaml file. If None, uses default location.
        """
        self._config: Dict[str, Any] = {}
        self._load_config(config_path)
        self._parse_config()
        
    def _load_config(self, config_path: Optional[str] = None):
        """Load configuration from YAML file."""
        if config_path is None:
            # Look for config in project root
            project_root = Path(__file__).parent.parent.parent
            config_path = project_root / "config.yaml"
            
            # Check for local config override
            local_config = project_root / "config.local.yaml"
            if local_config.exists():
                config_path = local_config
                
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found at {config_path}, using defaults")
            self._config = {}
            return
            
        with open(config_path, 'r') as f:
            self._config = yaml.safe_load(f) or {}
            
        # Replace environment variable placeholders
        self._substitute_env_vars(self._config)
        
        logger.info(f"Loaded configuration from {config_path}")
        
    def _substitute_env_vars(self, config: Dict):
        """Recursively substitute ${VAR} patterns with environment variables."""
        for key, value in config.items():
            if isinstance(value, dict):
                self._substitute_env_vars(value)
            elif isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                env_var = value[2:-1]
                config[key] = os.getenv(env_var, "")
                
    def _parse_config(self):
        """Parse configuration into typed objects."""
        # Mode
        self.mode: str = self._config.get("mode", "backtest")
        
        # Market settings
        market = self._config.get("market", {})
        self.market_type: str = market.get("type", "stocks")
        self.exchange: str = market.get("exchange", "US")
        self.symbols: list = market.get("symbols", ["AAPL", "MSFT"])
        
        # Timeframe
        timeframe = self._config.get("timeframe", {})
        self.interval: str = timeframe.get("interval", "15m")
        self.timezone: str = timeframe.get("timezone", "America/New_York")
        
        # Data settings
        data = self._config.get("data", {})
        self.data_provider: str = data.get("provider", "yfinance")
        self.history_days: int = data.get("history_days", 365)
        self.cache_enabled: bool = data.get("cache_enabled", True)
        self.cache_dir: str = data.get("cache_dir", "./data/cache")
        
        # Strategy settings
        strategy = self._config.get("strategy", {})
        self.strategy_name: str = strategy.get("name", "hybrid")
        
        # Technical config
        tech = strategy.get("technical", {})
        self.technical = TechnicalConfig(
            ma_fast=tech.get("ma_fast", 9),
            ma_slow=tech.get("ma_slow", 21),
            ma_signal=tech.get("ma_signal", 44),
            rsi_period=tech.get("rsi_period", 14),
            rsi_overbought=tech.get("rsi_overbought", 70),
            rsi_oversold=tech.get("rsi_oversold", 30),
            bb_period=tech.get("bb_period", 20),
            bb_std=tech.get("bb_std", 2.0),
            atr_period=tech.get("atr_period", 14),
        )
        
        # Ichimoku config
        ich = strategy.get("ichimoku", {})
        self.ichimoku = IchimokuConfig(
            conversion_period=ich.get("conversion_period", 9),
            base_period=ich.get("base_period", 26),
            span_b_period=ich.get("span_b_period", 52),
        )
        
        # ML config
        ml = strategy.get("ml", {})
        self.ml = MLConfig(
            model_type=ml.get("model_type", "xgboost"),
            lookback_period=ml.get("lookback_period", 60),
            prediction_horizon=ml.get("prediction_horizon", 5),
            retrain_interval=ml.get("retrain_interval", "weekly"),
            confidence_threshold=ml.get("confidence_threshold", 0.6),
        )
        
        # Risk config
        risk = self._config.get("risk", {})
        sl = risk.get("stop_loss", {})
        tp = risk.get("take_profit", {})
        
        self.risk = RiskConfig(
            position_sizing=risk.get("position_sizing", "fixed_fractional"),
            max_position_pct=risk.get("max_position_pct", 10.0),
            max_daily_loss_pct=risk.get("max_daily_loss_pct", 2.0),
            max_drawdown_pct=risk.get("max_drawdown_pct", 10.0),
            max_positions=risk.get("max_positions", 5),
            stop_loss=StopLossConfig(
                type=sl.get("type", "atr"),
                atr_multiplier=sl.get("atr_multiplier", 2.0),
                fixed_pct=sl.get("fixed_pct", 2.0),
                trailing_pct=sl.get("trailing_pct", 1.0),
            ),
            take_profit=TakeProfitConfig(
                type=tp.get("type", "atr"),
                atr_multiplier=tp.get("atr_multiplier", 3.0),
                fixed_pct=tp.get("fixed_pct", 4.0),
                risk_reward_ratio=tp.get("risk_reward_ratio", 2.0),
            ),
        )
        
        # Backtest config
        bt = self._config.get("backtest", {})
        self.backtest = BacktestConfig(
            start_date=bt.get("start_date", "2024-01-01"),
            end_date=bt.get("end_date", "2024-12-31"),
            initial_capital=bt.get("initial_capital", 100000),
            commission=bt.get("commission", 0.001),
            slippage=bt.get("slippage", 0.0005),
        )
        
        # Broker config
        broker = self._config.get("broker", {})
        broker_name = broker.get("name", "paper")
        broker_settings = broker.get(broker_name, {})
        
        self.broker = BrokerConfig(
            name=broker_name,
            api_key=broker_settings.get("api_key", ""),
            secret_key=broker_settings.get("secret_key", ""),
            base_url=broker_settings.get("base_url", ""),
        )
        
        # Alerts
        alerts = self._config.get("alerts", {})
        self.alerts_enabled: bool = alerts.get("enabled", True)
        self.discord_webhook: str = alerts.get("discord", {}).get("webhook_url", "")
        
        # Logging
        logging = self._config.get("logging", {})
        self.log_level: str = logging.get("level", "INFO")
        self.log_file: str = logging.get("file", "./logs/trading_bot.log")
        
    def get_raw(self, key: str, default: Any = None) -> Any:
        """Get raw config value by key path (e.g., 'strategy.technical.ma_fast')."""
        keys = key.split(".")
        value = self._config
        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
            else:
                return default
        return value if value is not None else default


# Singleton instance
_settings: Optional[Settings] = None


def get_settings(config_path: Optional[str] = None) -> Settings:
    """
    Get the global settings instance.
    
    Args:
        config_path: Optional path to configuration file.
        
    Returns:
        Settings instance.
    """
    global _settings
    if _settings is None:
        _settings = Settings(config_path)
    return _settings


def reload_settings(config_path: Optional[str] = None) -> Settings:
    """Force reload of settings."""
    global _settings
    _settings = Settings(config_path)
    return _settings

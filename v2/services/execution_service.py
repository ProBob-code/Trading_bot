"""
V2 Service Layer — Execution Service
======================================
Orchestrates trade lifecycle: signal → risk check → execution → logging.
"""
from loguru import logger


class ExecutionService:
    """Centralized trade execution orchestrator for V2."""
    
    def __init__(self, pipeline=None, paper_trader=None, db_manager=None):
        self.pipeline = pipeline
        self.paper_trader = paper_trader
        self.db = db_manager
        logger.info("[V2-SVC] ExecutionService initialized")
    
    def execute_trade(self, bot, signal, current_price, atr_value=0.0):
        """Run a trade through the full pipeline."""
        if self.pipeline:
            return self.pipeline.run_tick(
                bot_id=bot.bot_id, bot=bot,
                signal=signal, current_price=current_price,
                atr_value=atr_value
            )
        return None

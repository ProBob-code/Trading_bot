"""
V2 Service Layer — Risk Service
=================================
Centralized risk gating and compliance checks for V2.
"""
from loguru import logger


class RiskService:
    """Risk management service for the V2 institutional engine."""
    
    def __init__(self, risk_engine=None, db_manager=None):
        self.risk_engine = risk_engine
        self.db = db_manager
        logger.info("[V2-SVC] RiskService initialized")
    
    def check_risk(self, bot, signal, current_price):
        """Run pre-trade risk checks."""
        if self.risk_engine:
            return self.risk_engine.pre_trade_check(
                bot_id=bot.bot_id,
                signal=signal,
                price=current_price
            )
        return {'allowed': True, 'reason': 'No risk engine configured'}

"""
Alerts Module
=============

Telegram and Email notification service for trading alerts.
"""

from .alert_service import AlertService, get_alert_service

__all__ = ['AlertService', 'get_alert_service']

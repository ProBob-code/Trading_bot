"""
Alert Service
=============

Sends trading alerts via Telegram and Email.
Notifies on: signals, trades executed, SL/TP hits, strategy changes.
"""

import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from typing import Dict, Optional
from datetime import datetime
from loguru import logger


class AlertService:
    """
    Multi-channel alert service for trading notifications.
    
    Supports:
    - Telegram bot messages
    - Email notifications
    """
    
    def __init__(self):
        self.telegram_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.telegram_chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        
        self.email_smtp = os.getenv('SMTP_SERVER', 'smtp.gmail.com')
        self.email_port = int(os.getenv('SMTP_PORT', 587))
        self.email_user = os.getenv('SMTP_USER', '')
        self.email_pass = os.getenv('SMTP_PASS', '')
        self.email_to = os.getenv('ALERT_EMAIL', '')
        
        self._enabled = True
        logger.info("🔔 AlertService initialized")
    
    def send_telegram(self, message: str) -> bool:
        """
        Send alert via Telegram bot.
        
        Requires TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID env vars.
        """
        if not self.telegram_token or not self.telegram_chat_id:
            logger.debug("Telegram not configured, skipping alert")
            return False
        
        try:
            import requests
            url = f"https://api.telegram.org/bot{self.telegram_token}/sendMessage"
            payload = {
                'chat_id': self.telegram_chat_id,
                'text': message,
                'parse_mode': 'HTML'
            }
            response = requests.post(url, json=payload, timeout=10)
            
            if response.status_code == 200:
                logger.info(f"📱 Telegram alert sent")
                return True
            else:
                logger.error(f"Telegram failed: {response.text}")
                return False
                
        except Exception as e:
            logger.error(f"Telegram error: {e}")
            return False
    
    def send_email(self, subject: str, body: str) -> bool:
        """
        Send alert via Email.
        
        Requires SMTP credentials in env vars.
        """
        if not self.email_user or not self.email_to:
            logger.debug("Email not configured, skipping alert")
            return False
        
        try:
            msg = MIMEMultipart()
            msg['From'] = self.email_user
            msg['To'] = self.email_to
            msg['Subject'] = f"🤖 GodBotTrade: {subject}"
            
            msg.attach(MIMEText(body, 'html'))
            
            with smtplib.SMTP(self.email_smtp, self.email_port) as server:
                server.starttls()
                server.login(self.email_user, self.email_pass)
                server.send_message(msg)
            
            logger.info(f"📧 Email alert sent: {subject}")
            return True
            
        except Exception as e:
            logger.error(f"Email error: {e}")
            return False
    
    def alert_trade(self, trade: Dict):
        """Send trade execution alert."""
        side = trade.get('side', 'UNKNOWN')
        symbol = trade.get('symbol', 'UNKNOWN')
        price = trade.get('price', 0)
        quantity = trade.get('quantity', 0)
        
        emoji = '🟢' if side == 'BUY' else '🔴'
        
        message = f"""
{emoji} <b>Trade Executed</b>

Symbol: {symbol}
Side: {side}
Quantity: {quantity}
Price: ${price:,.2f}
Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        self.send_telegram(message)
        self.send_email(f"{side} {symbol}", message.replace('<b>', '').replace('</b>', ''))
    
    def alert_signal(self, symbol: str, signal: str, strategy: str, confidence: float = 0):
        """Send trading signal alert."""
        emoji = '📈' if signal == 'BUY' else '📉' if signal == 'SELL' else '⏸️'
        
        message = f"""
{emoji} <b>Signal: {signal}</b>

Symbol: {symbol}
Strategy: {strategy}
Confidence: {confidence:.1f}%
Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        self.send_telegram(message)
    
    def alert_stop_loss(self, symbol: str, entry_price: float, exit_price: float, loss_pct: float):
        """Send stop-loss hit alert."""
        message = f"""
🛑 <b>Stop Loss Triggered</b>

Symbol: {symbol}
Entry: ${entry_price:,.2f}
Exit: ${exit_price:,.2f}
Loss: -{loss_pct:.2f}%
        """.strip()
        
        self.send_telegram(message)
        self.send_email(f"STOP LOSS {symbol}", message.replace('<b>', '').replace('</b>', ''))
    
    def alert_take_profit(self, symbol: str, entry_price: float, exit_price: float, profit_pct: float):
        """Send take-profit hit alert."""
        message = f"""
🎯 <b>Take Profit Reached!</b>

Symbol: {symbol}
Entry: ${entry_price:,.2f}
Exit: ${exit_price:,.2f}
Profit: +{profit_pct:.2f}%
        """.strip()
        
        self.send_telegram(message)
        self.send_email(f"PROFIT {symbol}", message.replace('<b>', '').replace('</b>', ''))
    
    def alert_error(self, error: str):
        """Send error alert."""
        message = f"""
⚠️ <b>Trading Error</b>

{error}
Time: {datetime.now().strftime('%H:%M:%S')}
        """.strip()
        
        self.send_telegram(message)
        self.send_email("ERROR", error)


# Singleton instance
_alert_service = None

def get_alert_service() -> AlertService:
    """Get or create alert service instance."""
    global _alert_service
    if _alert_service is None:
        _alert_service = AlertService()
    return _alert_service

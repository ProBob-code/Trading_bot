import uuid
import threading
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque
from src.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Unified trade logging service with round-trip P&L tracking, backed by MySQL.
    """
    
    def __init__(self):
        # In-memory state for active session
        self.all_trades: List[Dict] = []
        
        # Position tracking for round-trip matching
        # key: f"{user_id}_{mode}_{bot_id}_{symbol}" -> deque of trade dicts (ENTRIES)
        self.open_positions: Dict[str, deque] = {}
        
        self.lock = threading.Lock()
        
        # Load history from MySQL immediately
        self._load_and_replay_history()

    def log_trade(self, 
                 symbol: str, 
                 side: str, 
                 quantity: float, 
                 price: float, 
                 user_id: int,
                 pnl: float = 0.0, 
                 strategy: str = "manual",
                 bot_id: str = "manual",
                 mode: str = "paper",
                 account_value: float = 0.0,
                 notes: str = "") -> Dict:
        """
        Log a trade event, calculate round-trip P&L, and save to MySQL.
        """
        with self.lock:
            timestamp = datetime.now()
            today_str = timestamp.strftime("%Y-%m-%d")
            
            # Normalize inputs
            symbol = symbol.upper()
            side = side.upper()  # BUY or SELL
            quantity = float(quantity)
            price = float(price)
            pnl = float(pnl)
            
            # Generate ID
            trade_id = str(uuid.uuid4())
            
            # Determine Trade Type and Round Trip ID
            tracker_key = f"{user_id}_{mode}_{bot_id}_{symbol}"
            if tracker_key not in self.open_positions:
                self.open_positions[tracker_key] = deque()
            
            position_queue = self.open_positions[tracker_key]
            
            trade_type = "ENTRY"
            round_trip_id = None
            
            # FIFO Round-trip matching logic
            is_closing = False
            if position_queue:
                first_pos = position_queue[0]
                if (first_pos['side'] == 'BUY' and side == 'SELL') or \
                   (first_pos['side'] == 'SELL' and side == 'BUY'):
                    is_closing = True
            
            if is_closing:
                trade_type = "EXIT"
                remaining_qty = quantity
                matched_entries = []
                
                while remaining_qty > 0 and position_queue:
                    entry = position_queue[0]
                    match_qty = min(remaining_qty, entry['remaining_qty'])
                    
                    matched_entries.append({
                        "entry_id": entry['trade_id'],
                        "quantity": match_qty
                    })
                    
                    entry['remaining_qty'] -= match_qty
                    remaining_qty -= match_qty
                    
                    if entry['remaining_qty'] <= 1e-9:
                        position_queue.popleft()
                
                if matched_entries:
                    round_trip_id = matched_entries[0]['entry_id']
            else:
                trade_type = "ENTRY"
                entry_record = {
                    "trade_id": trade_id,
                    "side": side,
                    "original_qty": quantity,
                    "remaining_qty": quantity,
                    "price": price
                }
                position_queue.append(entry_record)
                round_trip_id = trade_id
            
            # Construct Record
            record = {
                "user_id": user_id,
                "trade_id": trade_id,
                "round_trip_id": round_trip_id,
                "timestamp": timestamp,
                "date": timestamp.date(),
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "pnl": pnl,
                "strategy": strategy,
                "bot_id": bot_id,
                "mode": mode,
                "account_value": account_value,
                "notes": notes,
                "trade_type": trade_type
            }
            
            # Save to MySQL
            db_manager.add_trade(record)
            
            # Store in memory for immediate UI response (optional but keeps existing behavior)
            # Convert timestamp/date to str for memory compatibility if needed, 
            # or just store as is since get_history currently expects dicts from memory.
            mem_record = record.copy()
            mem_record['timestamp'] = mem_record['timestamp'].isoformat()
            mem_record['date'] = str(mem_record['date'])
            self.all_trades.append(mem_record)
            
            return mem_record

    def _load_and_replay_history(self):
        """Load all historical trades from MySQL and rebuild open positions."""
        logger.info("Loading and replaying trade history from MySQL...")
        
        # Reset state
        self.all_trades = []
        self.open_positions = {}
        
        # Get all trades for all users (or filter by active users if needed)
        # For simplicity, we get all trades from the trades table
        conn = db_manager._get_connection()
        try:
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM trades ORDER BY timestamp ASC")
            rows = cursor.fetchall()
            
            for record in rows:
                # Convert datetime/date objects to strings for existing UI/Logic compatibility
                if isinstance(record['timestamp'], datetime):
                    record['timestamp'] = record['timestamp'].isoformat()
                if hasattr(record['date'], 'isoformat'):
                    record['date'] = record['date'].isoformat()
                
                self.all_trades.append(record)
                self._replay_trade(record)
                
            logger.info(f"Replayed {len(self.all_trades)} trades from MySQL. System ready.")
        except Exception as e:
            logger.error(f"Error loading history from MySQL: {e}")
        finally:
            if conn.is_connected():
                cursor.close()
                conn.close()

    def _replay_trade(self, record: Dict):
        """Replay a single trade to rebuild memory state."""
        user_id = record.get('user_id', 0)
        mode = record.get('mode', 'paper')
        bot_id = record.get('bot_id', 'manual')
        symbol = record.get('symbol', '')
        tracker_key = f"{user_id}_{mode}_{bot_id}_{symbol}"
        
        if tracker_key not in self.open_positions:
            self.open_positions[tracker_key] = deque()
            
        queue = self.open_positions[tracker_key]
        
        qty = float(record['quantity'])
        side = record['side']
        
        # Check if closing
        is_closing = False
        if queue:
            first = queue[0]
            if (first['side'] == 'BUY' and side == 'SELL') or \
               (first['side'] == 'SELL' and side == 'BUY'):
                is_closing = True
        
        if is_closing:
            remaining = qty
            while remaining > 0 and queue:
                entry = queue[0]
                match = min(remaining, entry['remaining_qty'])
                entry['remaining_qty'] -= match
                remaining -= match
                
                if entry['remaining_qty'] <= 1e-9:
                    queue.popleft()
        else:
            queue.append({
                "trade_id": record.get('trade_id'),
                "side": side,
                "original_qty": qty,
                "remaining_qty": qty,
                "price": record['price']
            })

    def get_history(self, 
                   user_id: int,
                   start_date: str = None, 
                   end_date: str = None, 
                   symbol: str = None,
                   limit: int = 100) -> List[Dict]:
        """Get filtered trade history from MySQL via db_manager."""
        return db_manager.get_user_trades(user_id, limit=limit)

    def get_daily_summary(self, user_id: int, date_str: str) -> Dict:
        """Get summary stats for a specific day and user from memory or MySQL."""
        # For efficiency, use memory since it's already replayed
        trades = [t for t in self.all_trades if t['date'] == date_str and t.get('user_id') == user_id]
        
        stats = {
            "date": date_str,
            "total_trades": len(trades),
            "total_volume": sum(float(t['quantity']) * float(t['price']) for t in trades),
            "total_pnl": sum(float(t['pnl']) for t in trades),
            "wins": len([t for t in trades if float(t['pnl']) > 0]),
            "losses": len([t for t in trades if float(t['pnl']) < 0]),
            "symbols": list(set(t['symbol'] for t in trades))
        }
        return stats

# Global Singleton
_instance = None

def get_trade_logger(base_path: str = None):
    global _instance
    if _instance is None:
        _instance = TradeLogger()
    return _instance

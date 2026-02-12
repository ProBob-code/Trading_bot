import json
import uuid
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Any
from collections import deque

logger = logging.getLogger(__name__)

class TradeLogger:
    """
    Unified trade logging service with round-trip P&L tracking.
    
    Features:
    - UUID tracking for every trade
    - FIFO Round-trip pairing (entry <-> exit) for P&L calculation
    - Daily JSONL file rotation
    - Auto-replay history on startup to rebuild open position state
    """
    
    def __init__(self, data_dir: str = "data"):
        # We'll store reports in the project root's reports directory by default
        # But allow overriding. If relative, it's relative to CWD.
        self.reports_dir = Path(data_dir) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory state
        self.all_trades: List[Dict] = []
        
        # Position tracking for round-trip matching
        # key: f"{mode}_{bot_id}_{symbol}" -> deque of trade dicts (ENTRIES)
        self.open_positions: Dict[str, deque] = {}
        
        self.lock = threading.Lock()
        
        # Load history immediately
        self._load_and_replay_history()

    def log_trade(self, 
                 symbol: str, 
                 side: str, 
                 quantity: float, 
                 price: float, 
                 pnl: float = 0.0, 
                 strategy: str = "manual",
                 bot_id: str = "manual",
                 mode: str = "paper",
                 account_value: float = 0.0,
                 notes: str = "") -> Dict:
        """
        Log a trade event, calculate round-trip P&L, and save to disk.
        """
        with self.lock:
            timestamp = datetime.now()
            today_str = timestamp.strftime("%Y-%m-%d")
            
            # Normalize inputs
            symbol = symbol.upper()
            side = side.upper()  # BUY or SELL
            quantity = float(quantity)
            price = float(price)
            pnl = float(pnl) # This is the realized P&L passed from the caller (if any)
            
            # Generate ID
            trade_id = str(uuid.uuid4())
            
            # Determine Trade Type and Round Trip ID
            # We need to know if this is opening or closing a position
            # This logic depends on the bot's intent, but we can infer it or have it passed
            # For now, we infer based on side and existing positions in our tracker
            
            tracker_key = f"{mode}_{bot_id}_{symbol}"
            if tracker_key not in self.open_positions:
                self.open_positions[tracker_key] = deque()
            
            position_queue = self.open_positions[tracker_key]
            
            # Determine if ENTRY or EXIT
            # Simple heuristic: 
            # If we have open positions of opposite side, this is an EXIT (or partial exit)
            # If queue is empty, it's an ENTRY
            # If queue has same side, it's an ADD (ENTRY)
            
            trade_type = "ENTRY"
            round_trip_id = None
            
            # Check the side of the first position in queue (if any)
            is_closing = False
            if position_queue:
                first_pos = position_queue[0]
                # If opposite side, we are closing
                if (first_pos['side'] == 'BUY' and side == 'SELL') or \
                   (first_pos['side'] == 'SELL' and side == 'BUY'):
                    is_closing = True
            
            if is_closing:
                trade_type = "EXIT"
                # Match with open positions (FIFO)
                # We might close multiple entries if quantity is large
                remaining_qty = quantity
                matched_entries = []
                
                while remaining_qty > 0 and position_queue:
                    entry = position_queue[0]
                    match_qty = min(remaining_qty, entry['remaining_qty'])
                    
                    matched_entries.append({
                        "entry_id": entry['trade_id'],
                        "quantity": match_qty
                    })
                    
                    # Update entry
                    entry['remaining_qty'] -= match_qty
                    remaining_qty -= match_qty
                    
                    # If entry fully closed, remove from queue
                    if entry['remaining_qty'] <= 1e-9:
                        position_queue.popleft()
                
                # If we closed something, we can link IDs
                # For round_trip_id, we can use the trade_id of the *first* entry we matched
                if matched_entries:
                    round_trip_id = matched_entries[0]['entry_id']
                    
            else:
                # It's an ENTRY (Open New or Add)
                trade_type = "ENTRY"
                entry_record = {
                    "trade_id": trade_id,
                    "side": side,
                    "original_qty": quantity,
                    "remaining_qty": quantity,
                    "price": price
                }
                position_queue.append(entry_record)
                round_trip_id = trade_id # Entry is its own start of round trip
            
            # Construct Record
            record = {
                "trade_id": trade_id,
                "round_trip_id": round_trip_id,
                "timestamp": timestamp.isoformat(),
                "date": today_str,
                "symbol": symbol,
                "side": side,
                "quantity": quantity,
                "price": price,
                "value": quantity * price,
                "pnl": pnl, # Realized P&L from broker
                "strategy": strategy,
                "bot_id": bot_id,
                "mode": mode,
                "account_value": account_value,
                "notes": notes,
                "trade_type": trade_type
            }
            
            # Save
            self._write_trade(record, today_str)
            self.all_trades.append(record)
            
            return record

    def _write_trade(self, record: Dict, date_str: str):
        """Append trade to daily JSONL file."""
        daily_dir = self.reports_dir / date_str
        daily_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = daily_dir / "trades.jsonl"
        try:
            with open(file_path, "a") as f:
                f.write(json.dumps(record) + "\n")
        except Exception as e:
            logger.error(f"Failed to write trade log: {e}")

    def _load_and_replay_history(self):
        """Load all historical trades and replay them to rebuild open positions."""
        logger.info("Loading and replaying trade history...")
        
        # Reset state
        self.all_trades = []
        self.open_positions = {}
        
        # 1. Find all trade files
        # We expect structure: reports/YYYY-MM-DD/trades.jsonl
        trade_files = sorted(self.reports_dir.glob("*/trades.jsonl"))
        
        # 2. Load all trades chronologically
        for file_path in trade_files:
            try:
                with open(file_path, "r") as f:
                    for line in f:
                        if not line.strip(): continue
                        try:
                            record = json.loads(line)
                            self.all_trades.append(record)
                            self._replay_trade(record)
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Error reading {file_path}: {e}")
                
        logger.info(f"Replayed {len(self.all_trades)} trades. System ready.")

    def _replay_trade(self, record: Dict):
        """Replay a single trade to rebuild memory state."""
        tracker_key = f"{record['mode']}_{record['bot_id']}_{record['symbol']}"
        if tracker_key not in self.open_positions:
            self.open_positions[tracker_key] = deque()
            
        queue = self.open_positions[tracker_key]
        
        # Determine if Entry or Exit based on what was logged
        # But for replaying, we need to apply the logic precisely
        
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
            # Remove from queue
            remaining = qty
            while remaining > 0 and queue:
                entry = queue[0]
                match = min(remaining, entry['remaining_qty'])
                entry['remaining_qty'] -= match
                remaining -= match
                
                if entry['remaining_qty'] <= 1e-9:
                    queue.popleft()
        else:
            # Add to queue
            queue.append({
                "trade_id": record.get('trade_id'),
                "side": side,
                "original_qty": qty,
                "remaining_qty": qty,
                "price": record['price']
            })

    def get_history(self, 
                   start_date: str = None, 
                   end_date: str = None, 
                   symbol: str = None,
                   limit: int = 100) -> List[Dict]:
        """Get filtered trade history."""
        # Simple In-Memory Filtering
        # For production with millions of trades, use SQLite/DB
        
        matches = self.all_trades
        
        if symbol:
            matches = [t for t in matches if t['symbol'] == symbol]
            
        if start_date:
            matches = [t for t in matches if t['date'] >= start_date]
            
        if end_date:
            matches = [t for t in matches if t['date'] <= end_date]
            
        # Return newest first
        return sorted(matches, key=lambda x: x['timestamp'], reverse=True)[:limit]

    def get_daily_summary(self, date_str: str) -> Dict:
        """Get summary stats for a specific day."""
        trades = [t for t in self.all_trades if t['date'] == date_str]
        
        stats = {
            "date": date_str,
            "total_trades": len(trades),
            "total_volume": sum(t['value'] for t in trades),
            "total_pnl": sum(t['pnl'] for t in trades),
            "wins": len([t for t in trades if t['pnl'] > 0]),
            "losses": len([t for t in trades if t['pnl'] < 0]),
            "symbols": list(set(t['symbol'] for t in trades))
        }
        return stats

# Global Singleton
_instance = None

def get_trade_logger(base_path: str = None):
    global _instance
    if _instance is None:
        if base_path:
             _instance = TradeLogger(base_path)
        else:
             # Default to project root
             root = Path(__file__).parent.parent.parent
             _instance = TradeLogger(str(root))
    return _instance

if __name__ == "__main__":
    # Simple Test
    logger = get_trade_logger()
    logger.log_trade("BTCUSDT", "BUY", 1.0, 50000, bot_id="test_bot")
    logger.log_trade("BTCUSDT", "SELL", 0.5, 55000, pnl=2500, bot_id="test_bot") # Partial exit
    print("Logged 2 trades.")

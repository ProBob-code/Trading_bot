import json
import threading
import logging
from datetime import datetime
from typing import Dict, Optional
from shared.database.db_manager import db_manager

logger = logging.getLogger(__name__)

class SystemStateService:
    """
    Manages global system state (Paused/Running) with MySQL persistence.
    """
    
    def __init__(self):
        self.lock = threading.Lock()
        self.local_cache = {
            "is_paused": False,
            "last_updated": datetime.now().isoformat(),
            "session_id": None,
            "engine_version": "v2.0-institutional"
        }
        self._load_state()

    def get_session_id(self) -> Optional[str]:
        """Get current global session ID."""
        with self.lock:
            return self.local_cache.get("session_id")

    def set_session_id(self, sid: Optional[str]):
        """Set global session ID."""
        with self.lock:
            old_sid = self.local_cache.get("session_id")
            self.local_cache["session_id"] = sid
            if sid:
                logger.info(f"🆔 SESSION ID UPDATED: {old_sid} -> {sid}")
            else:
                logger.info(f"🆔 SESSION ID CLEARED (Previous: {old_sid})")

    def get_engine_version(self) -> str:
        """Get current engine version."""
        return self.local_cache.get("engine_version", "v2.0-institutional")

    def is_paused(self) -> bool:
        """Check if system is globally paused."""
        # We could refresh from DB here, but for performance we use local cache
        # and rely on set_paused updating it.
        with self.lock:
            return self.local_cache.get("is_paused", False)

    def set_paused(self, paused: bool):
        """Update pause state and persist to MySQL."""
        with self.lock:
            self.local_cache["is_paused"] = paused
            self.local_cache["last_updated"] = datetime.now().isoformat()
            
            # Persist to MySQL
            db_manager.set_system_state_val("is_paused", "1" if paused else "0")
            db_manager.set_system_state_val("last_updated", self.local_cache["last_updated"])
            
            state_str = "PAUSED" if paused else "RESUMED"
            logger.info(f"🛑 SYSTEM GLOBAL STATE: {state_str}")

    def _load_state(self):
        """Load state from MySQL."""
        try:
            paused_val = db_manager.get_system_state_val("is_paused")
            if paused_val is not None:
                self.local_cache["is_paused"] = paused_val == "1"
            
            updated_val = db_manager.get_system_state_val("last_updated")
            if updated_val:
                self.local_cache["last_updated"] = updated_val
                
            logger.info(f"Loaded system state from MySQL: {'PAUSED' if self.local_cache['is_paused'] else 'RUNNING'}")
        except Exception as e:
            logger.error(f"Failed to load system state from MySQL: {e}")

# Singleton Instance
_instance = None

def get_system_state(base_path: str = None):
    global _instance
    if _instance is None:
        _instance = SystemStateService()
    return _instance

if __name__ == "__main__":
    # Test
    svc = get_system_state()
    print(f"Initial State: {svc.is_paused()}")
    svc.set_paused(True)
    print(f"After Pause: {svc.is_paused()}")
    svc.set_paused(False)
    print(f"After Resume: {svc.is_paused()}")

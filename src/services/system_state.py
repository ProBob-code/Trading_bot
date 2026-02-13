import json
import threading
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

class SystemStateService:
    """
    Manages global system state (Paused/Running) with persistence.
    Singelton pattern to ensure unified state across threads.
    """
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.state_file = self.data_dir / "system_state.json"
        
        self.lock = threading.Lock()
        self.state = {
            "is_paused": False,
            "last_updated": datetime.now().isoformat()
        }
        
        self._load_state()

    def is_paused(self) -> bool:
        """Check if system is globally paused."""
        with self.lock:
            return self.state.get("is_paused", False)

    def set_paused(self, paused: bool):
        """Update pause state and persist to disk."""
        with self.lock:
            self.state["is_paused"] = paused
            self.state["last_updated"] = datetime.now().isoformat()
            self._save_state()
            state_str = "PAUSED" if paused else "RESUMED"
            logger.info(f"🛑 SYSTEM GLOBAL STATE: {state_str}")

    def _load_state(self):
        """Load state from JSON file."""
        if self.state_file.exists():
            try:
                with open(self.state_file, "r") as f:
                    data = json.load(f)
                    self.state.update(data)
                # Ensure boolean type
                self.state["is_paused"] = bool(self.state.get("is_paused", False))
                logger.info(f"Loaded system state: {'PAUSED' if self.state['is_paused'] else 'RUNNING'}")
            except Exception as e:
                logger.error(f"Failed to load system state: {e}")

    def _save_state(self):
        """Save state to JSON file."""
        try:
            with open(self.state_file, "w") as f:
                json.dump(self.state, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to save system state: {e}")

# Singleton Instance
_instance = None

def get_system_state(base_path: str = None):
    global _instance
    if _instance is None:
        if base_path:
             _instance = SystemStateService(base_path)
        else:
             # Default to project root/data
             root = Path(__file__).parent.parent.parent / "data"
             _instance = SystemStateService(str(root))
    return _instance

if __name__ == "__main__":
    # Test
    svc = get_system_state()
    print(f"Initial State: {svc.is_paused()}")
    svc.set_paused(True)
    print(f"After Pause: {svc.is_paused()}")
    svc.set_paused(False)
    print(f"After Resume: {svc.is_paused()}")

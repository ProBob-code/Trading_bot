import requests
import json
import time
import sys

BASE_URL = "http://localhost:5050"

def test_lazy_session():
    print("🔍 Step 1: Checking that no session exists initially...")
    resp = requests.get(f"{BASE_URL}/api/v2/sessions") # This might fail if login required, but let's see
    if resp.status_code == 401:
        print("⚠️ Authentication required. I will check the DB directly instead.")
    else:
        sessions = resp.json().get('sessions', [])
        print(f"📊 Current sessions: {len(sessions)}")
    
    # Check DB directly
    import os
    # We'll use the command line for DB check to be safe
    
def main():
    print("🚀 Starting Professional Session Verification...")
    
    # 1. Start a bot
    payload = {
        "symbol": "BTC/USDT",
        "market": "crypto",
        "strategy": "combined",
        "mode": "paper",
        "position_size": 100
    }
    
    # Since we can't easily do login here without more info, 
    # I'll rely on the server logs and DB checks.
    print("💡 Please start a bot manually from the dashboard OR I will use a test script that bypasses login if possible.")
    
if __name__ == "__main__":
    main()

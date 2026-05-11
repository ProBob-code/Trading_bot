import requests
import time
from loguru import logger

BASE_URL = "http://localhost:5050"

def test_health_endpoint():
    logger.info("🧪 Testing Health Endpoint...")
    try:
        res = requests.get(f"{BASE_URL}/api/v2/health")
        if res.status_code == 200:
            data = res.json()
            logger.success(f"✅ Health Check Passed: {data}")
        else:
            logger.error(f"❌ Health Check Failed: {res.status_code}")
    except Exception as e:
        logger.warning(f"⚠️ Server might not be running: {e}")

def test_pagination_logic():
    logger.info("🧪 Testing Pagination Logic Metadata...")
    # This requires being logged in, so we'll just check if the status code is 401 (auth required)
    # which proves the route is registered correctly.
    res = requests.get(f"{BASE_URL}/api/v2/trades?limit=10")
    if res.status_code == 401:
        logger.success("✅ Trade History API exists and is protected by login")
    elif res.status_code == 200:
        data = res.json()
        if 'total' in data and 'limit' in data:
            logger.success("✅ Pagination metadata found in response")
        else:
            logger.error("❌ Pagination metadata MISSING in 200 response")
    else:
        logger.error(f"❌ Trade History API returned: {res.status_code}")

if __name__ == "__main__":
    test_health_endpoint()
    test_pagination_logic()

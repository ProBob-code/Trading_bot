#!/bin/bash

# GodBotTrade Startup Script
# =========================

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
cd "$SCRIPT_DIR"

# Check if venv exists
if [ ! -d "venv" ]; then
    echo "❌ Virtual environment 'venv' not found!"
    echo "💡 Please create it first (e.g., python3 -m venv venv && venv/bin/pip install -r requirements.txt)"
    exit 1
fi

echo "🚀 Starting GodBotTrade Server..."
echo "📊 Dashboard: http://localhost:5050"

# Run the server using the venv python
./venv/bin/python3 api_server.py

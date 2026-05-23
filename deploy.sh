#!/bin/bash
# GoatBot Trade V2 - Easy Deploy Script

echo "🚀 Staging updated assets and codebase..."
git add api_server.py v2/web/godbot_home.html v2/web/godbot_login.html v2/web/index.html v2/web/logo.svg

echo "📝 Committing changes..."
git commit -m "feat: premium landing website, revamped glassmorphic auth page and interactive timeframe fixes"

echo "⬆️ Pushing changes to Railway..."
git push origin main

echo "✅ Done! Monitor the build on your Railway Dashboard: https://railway.app/"

#!/usr/bin/env bash
# One-command local deployment (replaces Railway for testing).
# NEW FILE — deploy.sh (Railway) is untouched.
set -euo pipefail
cd "$(dirname "$0")"

COMPOSE="docker compose"
if ! docker compose version >/dev/null 2>&1; then
    COMPOSE="docker-compose"
fi

echo "🔨 Building and starting GodBot (v3 quant engine) + MySQL..."
# docker-compose 1.29 can't recreate containers on Docker Engine 29
# (KeyError: 'ContainerConfig'), so take them down first. Volumes persist.
$COMPOSE -f docker-compose.local.yml down --remove-orphans 2>/dev/null || true
$COMPOSE -f docker-compose.local.yml up -d --build

echo ""
echo "⏳ Waiting for API to come up..."
for i in $(seq 1 60); do
    if curl -sf http://localhost:5050/api/stats >/dev/null 2>&1; then
        echo "✅ API is live."
        break
    fi
    sleep 2
done

IP=$(hostname -I | awk '{print $1}')
echo ""
echo "=============================================="
echo "  GodBot backend is running"
echo "  Local:      http://localhost:5050"
echo "  Team (LAN): http://${IP}:5050"
echo "  MySQL:      ${IP}:3307 (probobindb/probobindb)"
echo "=============================================="
echo ""
echo "Logs:   $COMPOSE -f docker-compose.local.yml logs -f godbot"
echo "Stop:   $COMPOSE -f docker-compose.local.yml down"

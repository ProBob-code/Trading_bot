#!/usr/bin/env bash
# Expose the local GodBot backend to overseas teammates via Cloudflare Tunnel.
#
# Usage:  ./setup_tunnel.sh godbot.yourdomain.com
#
# Requires a domain already added to your Cloudflare account (free plan is fine).
# Run this ONCE; afterwards the tunnel runs as a systemd service and survives reboot.
#
# IMPORTANT: this only creates the tunnel. You must ALSO add a Cloudflare Access
# policy (step 6 below) or the dashboard is open to the whole internet.
set -euo pipefail
cd "$(dirname "$0")"

HOSTNAME_ARG="${1:-}"
if [[ -z "$HOSTNAME_ARG" ]]; then
    echo "Usage: $0 <hostname>   e.g. $0 godbot.yourdomain.com" >&2
    exit 1
fi

TUNNEL_NAME="godbot"
CF_DIR="$HOME/.cloudflared"

# 1. Authenticate — opens a browser to pick which domain the tunnel belongs to.
if [[ ! -f "$CF_DIR/cert.pem" ]]; then
    echo "🔐 Authorising cloudflared with Cloudflare (a browser window will open)..."
    cloudflared tunnel login
else
    echo "🔐 Already authorised ($CF_DIR/cert.pem exists)."
fi

# 2. Create the named tunnel (idempotent).
if cloudflared tunnel list 2>/dev/null | awk '{print $2}' | grep -qx "$TUNNEL_NAME"; then
    echo "🚇 Tunnel '$TUNNEL_NAME' already exists."
else
    echo "🚇 Creating tunnel '$TUNNEL_NAME'..."
    cloudflared tunnel create "$TUNNEL_NAME"
fi

TUNNEL_ID=$(cloudflared tunnel list | awk -v n="$TUNNEL_NAME" '$2==n {print $1}')
if [[ -z "$TUNNEL_ID" ]]; then
    echo "❌ Could not determine tunnel ID for '$TUNNEL_NAME'." >&2
    exit 1
fi
echo "   Tunnel ID: $TUNNEL_ID"

# 3. Point the hostname at the tunnel (creates a proxied CNAME in Cloudflare DNS).
echo "🌐 Routing $HOSTNAME_ARG -> $TUNNEL_NAME"
cloudflared tunnel route dns --overwrite-dns "$TUNNEL_NAME" "$HOSTNAME_ARG"

# 4. Write the ingress config.
#    Traffic hits Cloudflare's edge, then rides the tunnel to the local port.
cat > "$CF_DIR/config.yml" <<EOF
tunnel: $TUNNEL_ID
credentials-file: $CF_DIR/$TUNNEL_ID.json

ingress:
  - hostname: $HOSTNAME_ARG
    service: http://localhost:5050
    originRequest:
      # The dashboard uses SocketIO; keep idle streams alive.
      noTLSVerify: true
      connectTimeout: 30s
  - service: http_status:404
EOF
echo "📝 Wrote $CF_DIR/config.yml"

# 5. Install as a boot-persistent service so the team keeps access after a reboot.
echo "⚙️  Installing systemd service (needs sudo)..."
sudo cloudflared --config "$CF_DIR/config.yml" service install || true
sudo systemctl enable --now cloudflared

echo ""
echo "=============================================="
echo "  Tunnel live:  https://$HOSTNAME_ARG"
echo "=============================================="
echo ""
echo "⚠️  STEP 6 — DO THIS NOW, the URL is public until you do:"
echo ""
echo "  1. Go to https://one.dash.cloudflare.com  ->  Access  ->  Applications"
echo "  2. Add an application  ->  Self-hosted"
echo "  3. Domain: $HOSTNAME_ARG"
echo "  4. Add policy:  Action=Allow, Include -> Emails  ->  list your teammates"
echo "  5. Save. Teammates now get an email one-time-code before reaching the app."
echo ""
echo "Status: sudo systemctl status cloudflared"
echo "Logs:   sudo journalctl -u cloudflared -f"

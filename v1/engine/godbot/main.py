"""
GodBot Trade – Paper Trading Runner
====================================

CLI entry point. Starts the Flask API server and orchestrator.
Asserts PAPER_MODE is active before any execution.

Usage:
    python -m godbot.main [--port 5050] [--instrument BTCUSDT] [--capital 100000]
"""

import argparse
import logging
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%H:%M:%S',
)
logger = logging.getLogger(__name__)


def main():
    """Main entry point for the paper trading system."""
    
    # ══════════════════════════════════════
    # SAFETY CHECK — NON-NEGOTIABLE
    # ══════════════════════════════════════
    from v1.engine.godbot import PAPER_MODE
    assert PAPER_MODE, (
        "❌ FATAL: PAPER_MODE is not active! "
        "Cannot start paper trading system without PAPER_MODE=True. "
        "This is a safety check to prevent accidental live trading."
    )
    logger.info("✅ PAPER_MODE confirmed active — no live trading possible")
    
    # Parse arguments
    parser = argparse.ArgumentParser(description='GodBot Paper Trading System')
    parser.add_argument('--port', type=int, default=5050, help='API port (default: 5050)')
    parser.add_argument('--instrument', type=str, default='BTCUSDT', help='Trading instrument')
    parser.add_argument('--capital', type=float, default=100000.0, help='Capital per bot')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Initialize components
    from v1.engine.godbot.db import PaperDB
    from v1.engine.godbot.bot_config import create_default_bots
    from v1.engine.godbot.api import paper_bp, _bots, _get_db
    
    logger.info(f"🚀 Starting GodBot Paper Trading System")
    logger.info(f"   Instrument: {args.instrument}")
    logger.info(f"   Capital per bot: ${args.capital:,.2f}")
    logger.info(f"   Port: {args.port}")
    
    # Create default bots
    db = PaperDB()
    default_bots = create_default_bots(args.instrument, args.capital)
    for bot in default_bots:
        _bots[bot.bot_id] = bot
        db.init_wallet(bot.bot_id, bot.virtual_capital)
        logger.info(f"   📋 Registered: {bot.name}")
    
    # Setup Flask app
    try:
        from flask import Flask
    except ImportError:
        logger.error("Flask is required. Install with: pip install flask")
        sys.exit(1)
    
    app = Flask(__name__)
    app.register_blueprint(paper_bp)
    
    # Serve static files
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
    
    @app.route('/web/<path:filename>')
    def serve_static(filename):
        from flask import send_from_directory
        return send_from_directory(web_dir, filename)
    
    logger.info(f"")
    logger.info(f"═══════════════════════════════════════════")
    logger.info(f"  GodBot Paper Trading Dashboard")
    logger.info(f"  http://localhost:{args.port}/paper")
    logger.info(f"  API: http://localhost:{args.port}/paper/status")
    logger.info(f"═══════════════════════════════════════════")
    logger.info(f"")
    
    app.run(host='0.0.0.0', port=args.port, debug=args.debug)


if __name__ == '__main__':
    main()

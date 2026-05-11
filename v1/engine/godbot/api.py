"""
GodBot Paper Trading API
=========================

Flask Blueprint with 18 endpoints for managing bots, trades,
comparison, debugging, simulation, validation, stress testing,
portfolio allocation, and data integrity.
"""

from flask import Blueprint, request, jsonify
import logging
import uuid
from datetime import datetime, timezone
from typing import Dict, Any

from . import PAPER_MODE
from .db import PaperDB
from .bot_config import BotConfig, create_default_bots
from .metrics import MetricsCalculator
from .comparator import BotComparator
from .validation import MonteCarloSimulator, WalkForwardValidator, RandomizationTest
from .stress_testing import run_full_stress_test
from .risk import RegimeDriftDetector
from .portfolio_allocator import PortfolioAllocator

logger = logging.getLogger(__name__)

paper_bp = Blueprint('paper', __name__, url_prefix='/paper')

# Shared state (initialized on first request)
_db: PaperDB = None
_bots: Dict[str, BotConfig] = {}


def _get_db() -> PaperDB:
    global _db
    if _db is None:
        _db = PaperDB()
    return _db


def _get_bots() -> Dict[str, BotConfig]:
    return _bots


# ═══════════════════════════════════════
# 1. System Status
# ═══════════════════════════════════════

@paper_bp.route('/status', methods=['GET'])
def get_status():
    """System status and PAPER_MODE confirmation."""
    db = _get_db()
    return jsonify({
        'paper_mode': PAPER_MODE,
        'total_bots': len(_bots),
        'total_trades': db.count_trades(),
        'active_bots': [b.to_dict() for b in _bots.values() if b.enabled],
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })


# ═══════════════════════════════════════
# 2-3. Bot CRUD
# ═══════════════════════════════════════

@paper_bp.route('/bots', methods=['GET'])
def list_bots():
    """List all configured bots."""
    db = _get_db()
    result = []
    for bot_id, config in _bots.items():
        wallet = db.get_wallet(bot_id)
        perf = db.get_performance(bot_id)
        result.append({
            'config': config.to_dict(),
            'wallet': wallet,
            'performance': perf[0] if perf else None,
        })
    return jsonify({'bots': result})


@paper_bp.route('/bots', methods=['POST'])
def add_bot():
    """Add a new bot configuration."""
    data = request.get_json() or {}
    config = BotConfig.from_dict(data)
    db = _get_db()
    _bots[config.bot_id] = config
    db.init_wallet(config.bot_id, config.virtual_capital)
    return jsonify({'status': 'created', 'bot': config.to_dict()}), 201


@paper_bp.route('/bots/<bot_id>', methods=['DELETE'])
def remove_bot(bot_id: str):
    """Remove a bot (keeps trade history)."""
    if bot_id in _bots:
        _bots[bot_id].enabled = False
        return jsonify({'status': 'disabled', 'bot_id': bot_id})
    return jsonify({'error': 'Bot not found'}), 404


@paper_bp.route('/bots/defaults', methods=['POST'])
def create_defaults():
    """Create default set of 4 strategy bots."""
    data = request.get_json() or {}
    instrument = data.get('instrument', 'BTCUSDT')
    capital = data.get('capital', 100000.0)
    
    db = _get_db()
    bots = create_default_bots(instrument, capital)
    for bot in bots:
        _bots[bot.bot_id] = bot
        db.init_wallet(bot.bot_id, bot.virtual_capital)
    
    return jsonify({
        'status': 'created',
        'bots': [b.to_dict() for b in bots]
    }), 201


# ═══════════════════════════════════════
# 4-5. Trades
# ═══════════════════════════════════════

@paper_bp.route('/trades', methods=['GET'])
def get_trades():
    """Get trades with optional filters and pagination."""
    db = _get_db()
    bot_id = request.args.get('bot_id')
    instrument = request.args.get('instrument')
    limit = int(request.args.get('limit', 50))
    offset = int(request.args.get('offset', 0))
    
    trades = db.get_trades(bot_id=bot_id, instrument=instrument,
                           limit=limit, offset=offset)
    total = db.count_trades(bot_id=bot_id)
    
    return jsonify({
        'trades': trades,
        'total': total,
        'limit': limit,
        'offset': offset,
    })


@paper_bp.route('/trades/<trade_id>/debug', methods=['GET'])
def debug_trade(trade_id: str):
    """
    Debug a specific trade.
    Returns: signal explanation, indicator snapshot,
    slippage impact, hypothetical no-slippage PnL.
    """
    db = _get_db()
    trades = db.get_trades(limit=100000)
    trade = next((t for t in trades if t.get('trade_id') == trade_id), None)
    
    if not trade:
        return jsonify({'error': 'Trade not found'}), 404
    
    # Calculate hypothetical PnL without slippage/fees
    entry = trade.get('entry_price', 0)
    exit_p = trade.get('exit_price', 0)
    size = trade.get('position_size', 0)
    side = trade.get('side', '')
    slippage = trade.get('slippage_applied', 0)
    fees = trade.get('fees_paid', 0)
    
    if exit_p and entry:
        if side == 'buy':
            gross_pnl = (exit_p - entry) * size
        else:
            gross_pnl = (entry - exit_p) * size
        hypothetical_pnl = gross_pnl + slippage + fees  # Without costs
    else:
        gross_pnl = 0
        hypothetical_pnl = 0
    
    return jsonify({
        'trade': trade,
        'debug': {
            'gross_pnl': round(gross_pnl, 4),
            'hypothetical_no_cost_pnl': round(hypothetical_pnl, 4),
            'slippage_impact': round(slippage, 4),
            'fees_impact': round(fees, 4),
            'cost_drag_pct': round((slippage + fees) / max(abs(gross_pnl), 1) * 100, 2) if gross_pnl else 0,
            'indicator_snapshot': trade.get('indicator_snapshot', {}),
            'trade_reason': trade.get('trade_reason', ''),
        }
    })


# ═══════════════════════════════════════
# 6-7. Metrics & Equity
# ═══════════════════════════════════════

@paper_bp.route('/metrics/<bot_id>', methods=['GET'])
def get_metrics(bot_id: str):
    """Get full metrics for a bot."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    wallet = db.get_wallet(bot_id)
    
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    
    metrics = MetricsCalculator.calculate(trades, initial_capital)
    return jsonify({'bot_id': bot_id, 'metrics': metrics})


@paper_bp.route('/equity/<bot_id>', methods=['GET'])
def get_equity(bot_id: str):
    """Get equity curve data for charting."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    wallet = db.get_wallet(bot_id)
    
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    metrics = MetricsCalculator.calculate(trades, initial_capital)
    
    return jsonify({
        'bot_id': bot_id,
        'equity_curve': metrics.get('equity_curve', []),
        'drawdown_curve': metrics.get('drawdown_curve', []),
        'total_pnl': metrics.get('total_pnl', 0),
        'max_drawdown_pct': metrics.get('max_drawdown_pct', 0),
    })


# ═══════════════════════════════════════
# 8. Regime Performance
# ═══════════════════════════════════════

@paper_bp.route('/regime/<bot_id>', methods=['GET'])
def get_regime_performance(bot_id: str):
    """Get regime-specific performance breakdown."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    wallet = db.get_wallet(bot_id)
    
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    metrics = MetricsCalculator.calculate(trades, initial_capital)
    
    return jsonify({
        'bot_id': bot_id,
        'regime_performance': {
            'trending': metrics.get('trend_expectancy', 0),
            'ranging': metrics.get('range_expectancy', 0),
            'high_volatility': metrics.get('high_vol_expectancy', 0),
            'low_volatility': metrics.get('low_vol_expectancy', 0),
        }
    })


# ═══════════════════════════════════════
# 9. Comparison
# ═══════════════════════════════════════

@paper_bp.route('/compare', methods=['GET'])
def compare_bots():
    """Compare all bots with composite scoring."""
    db = _get_db()
    all_metrics = {}
    
    for bot_id in _bots:
        trades = db.get_all_trades_for_bot(bot_id)
        wallet = db.get_wallet(bot_id)
        initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
        all_metrics[bot_id] = MetricsCalculator.calculate(trades, initial_capital)
    
    comparison = BotComparator.compare(all_metrics)
    return jsonify({'comparison': comparison})


# ═══════════════════════════════════════
# 10. Validation (Walk-Forward + MC)
# ═══════════════════════════════════════

@paper_bp.route('/validation/<bot_id>', methods=['GET'])
def get_validation(bot_id: str):
    """Run walk-forward and Monte Carlo validation for a bot."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    wallet = db.get_wallet(bot_id)
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    
    if len(trades) < 10:
        return jsonify({
            'bot_id': bot_id,
            'error': 'Need at least 10 trades for validation',
            'walk_forward': None,
            'monte_carlo': None,
        })
    
    # Walk-forward
    split_point = len(trades) // 2
    wf = WalkForwardValidator()
    wf_result = wf.validate(trades[:split_point], trades[split_point:])
    
    # Monte Carlo
    pnls = [t.get('net_pnl', 0) or 0 for t in trades]
    mc = MonteCarloSimulator(n_simulations=1000)
    mc_result = mc.simulate(pnls, initial_capital)
    
    # Randomization
    metrics = MetricsCalculator.calculate(trades, initial_capital)
    rt = RandomizationTest(n_shuffles=100)
    rand_result = rt.test(pnls, metrics.get('expectancy', 0))
    
    return jsonify({
        'bot_id': bot_id,
        'walk_forward': wf_result,
        'monte_carlo': mc_result,
        'randomization': rand_result,
    })


# ═══════════════════════════════════════
# 11-12. Run & Reset
# ═══════════════════════════════════════

@paper_bp.route('/run', methods=['POST'])
def run_simulation():
    """Trigger a simulation run (placeholder — orchestrator drives this)."""
    return jsonify({
        'status': 'Simulation runs are driven by the orchestrator. '
                  'Use godbot.main to start.',
        'bots': len(_bots),
    })


@paper_bp.route('/reset', methods=['POST'])
def reset_data():
    """Reset all paper trading data."""
    data = request.get_json() or {}
    bot_id = data.get('bot_id')
    
    db = _get_db()
    if bot_id:
        db.reset_bot(bot_id)
        return jsonify({'status': f'Reset bot {bot_id}'})
    else:
        db.reset_all()
        return jsonify({'status': 'All data reset'})


# ═══════════════════════════════════════
# 14. Stress Testing
# ═══════════════════════════════════════

@paper_bp.route('/stress/<bot_id>', methods=['GET'])
def get_stress_test(bot_id: str):
    """
    Run parameter stability + slippage stress + tail risk for a bot.
    Returns combined stress test results with overall grade.
    """
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    
    if len(trades) < 5:
        return jsonify({
            'bot_id': bot_id,
            'error': 'Need at least 5 trades for stress testing',
        })
    
    config = _bots.get(bot_id)
    base_params = config.to_dict() if config else {}
    
    wallet = db.get_wallet(bot_id)
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    
    result = run_full_stress_test(trades, base_params, initial_capital)
    return jsonify({'bot_id': bot_id, 'stress_test': result})


# ═══════════════════════════════════════
# 15. Regime Drift Detection
# ═══════════════════════════════════════

@paper_bp.route('/regime-drift/<bot_id>', methods=['GET'])
def get_regime_drift(bot_id: str):
    """Rolling regime z-scores with early warnings."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    
    detector = RegimeDriftDetector()
    result = detector.detect(trades)
    return jsonify({'bot_id': bot_id, 'regime_drift': result})


# ═══════════════════════════════════════
# 16. Portfolio Allocation
# ═══════════════════════════════════════

@paper_bp.route('/allocation', methods=['GET'])
def get_allocation():
    """Portfolio allocation recommendations across all bots."""
    db = _get_db()
    
    all_metrics = {}
    total_capital = 0
    
    for bot_id in _bots:
        trades = db.get_all_trades_for_bot(bot_id)
        wallet = db.get_wallet(bot_id)
        initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
        total_capital += initial_capital
        
        metrics = MetricsCalculator.calculate(trades, initial_capital)
        
        # Add safety label from comparator
        safety = BotComparator.classify_safety(metrics)
        metrics['safety_label'] = safety.get('label', 'CAUTION')
        
        # Add composite score
        perf = db.get_performance(bot_id)
        if perf:
            metrics['composite_score'] = perf[0].get('composite_score', 0)
        
        all_metrics[bot_id] = metrics
    
    allocator = PortfolioAllocator()
    result = allocator.allocate(all_metrics, total_capital)
    return jsonify(result)


# ═══════════════════════════════════════
# 17. Capital Efficiency
# ═══════════════════════════════════════

@paper_bp.route('/efficiency/<bot_id>', methods=['GET'])
def get_efficiency(bot_id: str):
    """Capital efficiency metrics for a bot."""
    db = _get_db()
    trades = db.get_all_trades_for_bot(bot_id)
    wallet = db.get_wallet(bot_id)
    initial_capital = wallet.get('initial_capital', 100000) if wallet else 100000
    
    metrics = MetricsCalculator.calculate(trades, initial_capital)
    
    return jsonify({
        'bot_id': bot_id,
        'efficiency': {
            'max_margin_used': metrics.get('max_margin_used', 0),
            'avg_capital_deployed': metrics.get('avg_capital_deployed', 0),
            'return_per_max_margin': metrics.get('return_per_max_margin', 0),
            'return_per_avg_capital': metrics.get('return_per_avg_capital', 0),
            'capital_efficiency_ratio': metrics.get('capital_efficiency_ratio', 0),
            'total_pnl': metrics.get('total_pnl', 0),
            'avg_holding_seconds': metrics.get('avg_holding_seconds', 0),
        }
    })


# ═══════════════════════════════════════
# 18. Data Integrity
# ═══════════════════════════════════════

@paper_bp.route('/integrity', methods=['GET'])
def get_integrity():
    """Data integrity log — reproducibility audit trail."""
    db = _get_db()
    limit = int(request.args.get('limit', 50))
    log = db.get_integrity_log(limit)
    return jsonify({'integrity_log': log, 'total': len(log)})


# ═══════════════════════════════════════
# Dashboard Page
# ═══════════════════════════════════════

@paper_bp.route('/', methods=['GET'])
def dashboard():
    """Serve the paper trading dashboard."""
    import os
    web_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'web')
    dashboard_path = os.path.join(web_dir, 'paper_dashboard.html')
    
    if os.path.exists(dashboard_path):
        with open(dashboard_path, 'r') as f:
            return f.read(), 200, {'Content-Type': 'text/html'}
    
    return jsonify({'error': 'Dashboard not found', 'path': dashboard_path}), 404

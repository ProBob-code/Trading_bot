"""
Bot Comparator
==============

Multi-bot comparison engine with weighted composite scoring,
SAFE/DANGEROUS classification, and overfitting detection.

Composite Score =
  25% Expectancy + 15% Sharpe + 10% Sortino + 15% ProfitFactor
  + 10% Calmar + 10% RegimeAdaptability + 10% Stability + 5% TradeCount
"""

import numpy as np
from typing import Dict, Any, List


# Minimum trades for full scoring
MIN_TRADE_THRESHOLD = 30


class BotComparator:
    """
    Compares multiple bots using a weighted composite score.
    Classifies bots as SAFE or DANGEROUS with quantitative thresholds.
    """
    
    # Composite weights
    WEIGHTS = {
        'expectancy': 0.25,
        'sharpe': 0.15,
        'sortino': 0.10,
        'profit_factor': 0.15,
        'calmar': 0.10,
        'regime_adaptability': 0.10,
        'stability': 0.10,
        'trade_count': 0.05,
    }
    
    @staticmethod
    def compare(bot_metrics: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Compare bots and return ranked list with composite scores.
        
        Args:
            bot_metrics: {bot_id: metrics_dict} from MetricsCalculator
            
        Returns:
            List of dicts sorted by composite score (best first)
        """
        if not bot_metrics:
            return []
        
        results = []
        
        # Normalize each factor across all bots
        all_values = {
            'expectancy': [],
            'sharpe': [],
            'sortino': [],
            'profit_factor': [],
            'calmar': [],
            'stability': [],
        }
        
        for bot_id, m in bot_metrics.items():
            all_values['expectancy'].append(m.get('expectancy', 0))
            all_values['sharpe'].append(m.get('sharpe_ratio', 0))
            all_values['sortino'].append(m.get('sortino_ratio', 0))
            all_values['profit_factor'].append(min(m.get('profit_factor', 0), 10))  # Cap PF
            all_values['calmar'].append(min(m.get('calmar_ratio', 0), 10))
            all_values['stability'].append(min(m.get('stability_score', 0), 100))
        
        # Min-max normalization
        normalized = {}
        for key, vals in all_values.items():
            arr = np.array(vals, dtype=float)
            mn, mx = arr.min(), arr.max()
            if mx - mn > 0:
                normalized[key] = (arr - mn) / (mx - mn)
            else:
                normalized[key] = np.ones_like(arr) * 0.5
        
        bot_ids = list(bot_metrics.keys())
        
        for i, bot_id in enumerate(bot_ids):
            m = bot_metrics[bot_id]
            total_trades = m.get('total_trades', 0)
            
            # Regime adaptability: how many regimes have positive expectancy
            regime_scores = [
                1 if m.get('trend_expectancy', 0) > 0 else 0,
                1 if m.get('range_expectancy', 0) > 0 else 0,
                1 if m.get('high_vol_expectancy', 0) > 0 else 0,
                1 if m.get('low_vol_expectancy', 0) > 0 else 0,
            ]
            regime_adaptability = sum(regime_scores) / 4.0
            
            # Trade count score (penalize low count)
            if total_trades >= MIN_TRADE_THRESHOLD:
                trade_count_score = 1.0
            elif total_trades > 0:
                trade_count_score = total_trades / MIN_TRADE_THRESHOLD
            else:
                trade_count_score = 0.0
            
            # Composite score
            composite = (
                BotComparator.WEIGHTS['expectancy'] * normalized['expectancy'][i]
                + BotComparator.WEIGHTS['sharpe'] * normalized['sharpe'][i]
                + BotComparator.WEIGHTS['sortino'] * normalized['sortino'][i]
                + BotComparator.WEIGHTS['profit_factor'] * normalized['profit_factor'][i]
                + BotComparator.WEIGHTS['calmar'] * normalized['calmar'][i]
                + BotComparator.WEIGHTS['regime_adaptability'] * regime_adaptability
                + BotComparator.WEIGHTS['stability'] * normalized['stability'][i]
                + BotComparator.WEIGHTS['trade_count'] * trade_count_score
            )
            
            # Apply penalty for low trade count
            if total_trades < MIN_TRADE_THRESHOLD:
                composite *= (total_trades / MIN_TRADE_THRESHOLD)
            
            # Safety classification
            safety = BotComparator.classify_safety(m)
            
            # Overfitting risk assessment
            overfit_flags = BotComparator.assess_overfitting(m)
            
            results.append({
                'bot_id': bot_id,
                'composite_score': round(composite * 100, 2),
                'safety_label': safety['label'],
                'safety_reasons': safety['reasons'],
                'overfit_flags': overfit_flags,
                'is_overfit_flagged': len(overfit_flags) > 0,
                'regime_adaptability': round(regime_adaptability * 100, 2),
                'trade_count_score': round(trade_count_score * 100, 2),
                # Key metrics for comparison table
                'total_trades': total_trades,
                'win_rate': m.get('win_rate', 0),
                'expectancy': m.get('expectancy', 0),
                'net_expectancy': m.get('net_expectancy', 0),
                'sharpe_ratio': m.get('sharpe_ratio', 0),
                'sortino_ratio': m.get('sortino_ratio', 0),
                'profit_factor': m.get('profit_factor', 0),
                'calmar_ratio': m.get('calmar_ratio', 0),
                'max_drawdown_pct': m.get('max_drawdown_pct', 0),
                'total_pnl': m.get('total_pnl', 0),
                'total_return_pct': m.get('total_return_pct', 0),
                'risk_of_ruin': m.get('risk_of_ruin', 0),
                'avg_r': m.get('avg_r', 0),
                'stability_score': m.get('stability_score', 0),
                'big_winner_dependent': m.get('big_winner_dependent', False),
            })
        
        # Sort by composite score descending
        results.sort(key=lambda x: x['composite_score'], reverse=True)
        
        # Add rank
        for i, r in enumerate(results):
            r['rank'] = i + 1
        
        return results
    
    @staticmethod
    def classify_safety(m: Dict[str, Any]) -> Dict[str, Any]:
        """
        Classify bot as SAFE or DANGEROUS.
        
        SAFE if:
          - Expectancy > 0
          - Sharpe > 1
          - Max DD < 20%
          - Not overfit flagged
          - Risk of ruin < 5%
        
        DANGEROUS if:
          - Negative expectancy
          - Overfit flagged
          - MC worst DD > 40%
          - Risk of ruin > 20%
        """
        reasons = []
        
        expectancy = m.get('expectancy', 0)
        sharpe = m.get('sharpe_ratio', 0)
        max_dd = m.get('max_drawdown_pct', 0)
        ror = m.get('risk_of_ruin', 0)
        mc_worst = m.get('mc_worst_dd', 0)
        is_overfit = m.get('is_overfit_flagged', False)
        
        # Dangerous checks
        if expectancy < 0:
            reasons.append("Negative expectancy")
        if is_overfit:
            reasons.append("Overfit flagged")
        if mc_worst > 40:
            reasons.append(f"MC worst DD {mc_worst:.1f}% > 40%")
        if ror > 20:
            reasons.append(f"Risk of ruin {ror:.1f}% > 20%")
        
        if reasons:
            return {'label': 'DANGEROUS', 'reasons': reasons}
        
        # Safe checks
        safe_reasons = []
        if expectancy > 0:
            safe_reasons.append(f"Positive expectancy ({expectancy:.2f})")
        if sharpe > 1:
            safe_reasons.append(f"Good Sharpe ({sharpe:.2f})")
        if max_dd < 20:
            safe_reasons.append(f"Controlled DD ({max_dd:.1f}%)")
        if ror < 5:
            safe_reasons.append(f"Low risk of ruin ({ror:.1f}%)")
        
        if len(safe_reasons) >= 3:
            return {'label': 'SAFE', 'reasons': safe_reasons}
        
        return {'label': 'CAUTION', 'reasons': safe_reasons or ['Insufficient data']}
    
    @staticmethod
    def assess_overfitting(m: Dict[str, Any]) -> List[str]:
        """Assess overfitting risk indicators."""
        flags = []
        
        total_trades = m.get('total_trades', 0)
        if total_trades < MIN_TRADE_THRESHOLD:
            flags.append(f"Too few trades ({total_trades} < {MIN_TRADE_THRESHOLD})")
        
        if m.get('big_winner_dependent', False):
            flags.append("Relies on rare big winners (>30% from >3R trades)")
        
        # High in-sample vs out-of-sample divergence (set by validation module)
        oos_ratio = m.get('oos_expectancy_ratio', None)
        if oos_ratio is not None and oos_ratio > 2.0:
            flags.append(f"In-sample/OOS expectancy ratio {oos_ratio:.1f}x (>2x = overfit)")
        
        return flags

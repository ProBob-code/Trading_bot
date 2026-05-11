"""
Trading Report Generator
========================

Generates comprehensive trading reports with visualization.
"""

from datetime import datetime
from typing import Dict, List, Optional
import pandas as pd


class TradingReportGenerator:
    """
    Generates trading performance reports.
    
    Creates detailed reports including:
    - Summary statistics
    - P&L breakdown
    - Trade history
    - Ichimoku signal analysis
    """
    
    @staticmethod
    def format_report_for_display(report: Dict) -> str:
        """
        Format a trading report for display in Streamlit.
        
        Args:
            report: Report dictionary from IchimokuAutoTrader.generate_report()
            
        Returns:
            Formatted markdown string
        """
        if not report or not report.get('start_time'):
            return "No trading data available."
        
        # Header
        lines = [
            "# 📊 Automated Trading Report",
            "",
            f"**Symbol:** {report['symbol']}",
            f"**Period:** {report['start_time'][:19]} to {report['stop_time'][:19]}",
            f"**Duration:** {report['duration_minutes']:.1f} minutes",
            "",
            "---",
            "",
        ]
        
        # Capital Summary
        lines.extend([
            "## 💰 Capital Summary",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Initial Capital | ₹{report['initial_capital']:,.2f} |",
            f"| Final Capital | ₹{report['final_capital']:,.2f} |",
            f"| **Total P&L** | ₹{report['total_pnl']:+,.2f} |",
            f"| **ROI** | {report['roi_percent']:+.2f}% |",
            "",
        ])
        
        # Trade Statistics
        lines.extend([
            "## 📈 Trade Statistics",
            "",
            f"| Metric | Value |",
            f"|--------|-------|",
            f"| Total Trades | {report['total_trades']} |",
            f"| Winning Trades | {report['winning_trades']} |",
            f"| Losing Trades | {report['losing_trades']} |",
            f"| **Win Rate** | {report['win_rate']:.1f}% |",
            f"| Profit Factor | {report['profit_factor']} |",
            f"| Avg Trade P&L | ₹{report['avg_trade_pnl']:,.2f} |",
            f"| Max Drawdown | {report['max_drawdown_pct']:.1f}% |",
            "",
        ])
        
        # P&L Breakdown
        lines.extend([
            "## 💹 P&L Breakdown",
            "",
            f"| Category | Amount |",
            f"|----------|--------|",
            f"| Gross Profit | ₹{report['gross_profit']:,.2f} |",
            f"| Gross Loss | ₹{report['gross_loss']:,.2f} |",
            f"| **Net P&L** | ₹{report['total_pnl']:+,.2f} |",
            "",
        ])
        
        # Signal Analysis
        lines.extend([
            "## 🌩️ Ichimoku Signal Analysis",
            "",
            f"| Signal Type | Count |",
            f"|-------------|-------|",
            f"| Total Signals | {report['total_signals']} |",
            f"| Buy Signals | {report['buy_signals']} |",
            f"| Sell Signals | {report['sell_signals']} |",
            "",
        ])
        
        # Ichimoku Component Breakdown
        ich_stats = report.get('ichimoku_stats', {})
        if ich_stats:
            lines.extend([
                "### Ichimoku Component Signals",
                "",
                f"| Component | Bullish | Bearish |",
                f"|-----------|---------|---------|",
                f"| Kumo Breakout | {ich_stats.get('kumo_bullish', 0)} | {ich_stats.get('kumo_bearish', 0)} |",
                f"| TK Cross | {ich_stats.get('tk_bullish', 0)} | {ich_stats.get('tk_bearish', 0)} |",
                "",
            ])
        
        # Trade History
        trades = report.get('trades', [])
        if trades:
            lines.extend([
                "## 📜 Trade History",
                "",
                "| Time | Side | Qty | Price | P&L | Confidence |",
                "|------|------|-----|-------|-----|------------|",
            ])
            
            for trade in trades:
                time_str = trade['timestamp'][11:19] if len(trade['timestamp']) > 11 else trade['timestamp']
                pnl_str = f"₹{trade['pnl']:+,.2f}" if trade['pnl'] != 0 else "-"
                conf_str = f"{trade['confidence']*100:.0f}%"
                lines.append(
                    f"| {time_str} | {trade['side']} | {trade['quantity']} | ₹{trade['price']:,.2f} | {pnl_str} | {conf_str} |"
                )
            
            lines.append("")
        
        return "\n".join(lines)
    
    @staticmethod
    def get_summary_metrics(report: Dict) -> Dict:
        """
        Extract key summary metrics for dashboard display.
        
        Args:
            report: Report dictionary
            
        Returns:
            Dictionary of key metrics
        """
        return {
            'total_trades': report.get('total_trades', 0),
            'win_rate': report.get('win_rate', 0),
            'total_pnl': report.get('total_pnl', 0),
            'roi_percent': report.get('roi_percent', 0),
            'profit_factor': report.get('profit_factor', 'N/A'),
            'max_drawdown': report.get('max_drawdown_pct', 0),
        }
    
    @staticmethod
    def trades_to_dataframe(report: Dict) -> pd.DataFrame:
        """
        Convert trades to pandas DataFrame for display.
        
        Args:
            report: Report dictionary
            
        Returns:
            DataFrame of trades
        """
        trades = report.get('trades', [])
        if not trades:
            return pd.DataFrame()
        
        df = pd.DataFrame(trades)
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        return df

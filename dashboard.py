"""
Trading Bot Dashboard - Indian Markets Edition
===============================================

Streamlit-based frontend for paper trading with technical analysis visualization.

Features:
- Interactive candlestick charts with indicators
- Ichimoku Cloud (一目均衡表) - Primary indicator
- Dropdown to select technical indicators
- Paper trading simulation in Indian Rupees (₹)
- Real-time P&L tracking

Usage:
    python -m streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.data.historical_loader import HistoricalLoader
from src.data.data_provider import AlphaVantageProvider
from src.data.crypto_provider import BinanceCryptoProvider, LiveCryptoFeed
from src.indicators.technical import TechnicalIndicators
from src.indicators.custom import CustomIndicators
from src.strategy.ta_strategy import TAStrategy
from src.execution.brokers.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager
from src.engine.ichimoku_auto_trader import IchimokuAutoTrader, TradingState
from src.reports.trading_report import TradingReportGenerator

# Currency symbol
CURRENCY = "₹"
INITIAL_CAPITAL = 1000000  # 10 Lakhs INR

# Page configuration
st.set_page_config(
    page_title="Trading Bot Dashboard - NSE/BSE",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #1e40af, #7c3aed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 1rem;
    }
    .ichimoku-badge {
        background: linear-gradient(135deg, #f59e0b 0%, #ef4444 100%);
        padding: 0.5rem 1rem;
        border-radius: 8px;
        color: white;
        font-weight: 600;
        display: inline-block;
        margin: 0.5rem 0;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
    }
    .profit { color: #16a34a; font-weight: 600; }
    .loss { color: #dc2626; font-weight: 600; }
    .indicator-info {
        background: #1e293b;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #f59e0b;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'paper_trader' not in st.session_state:
    st.session_state.paper_trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
    st.session_state.order_manager = OrderManager(st.session_state.paper_trader)
    st.session_state.paper_trader.set_order_manager(st.session_state.order_manager)
    st.session_state.trades = []
    st.session_state.data_loaded = False
    st.session_state.df = None
    st.session_state.show_market_depth = False
    st.session_state.timeline_filter = "All"
    # Auto-trading state
    st.session_state.auto_trader = None
    st.session_state.auto_trading_active = False
    st.session_state.auto_trade_report = None
    st.session_state.auto_trade_signals = []
    # Crypto state
    st.session_state.market_mode = "Stocks"
    st.session_state.crypto_provider = BinanceCryptoProvider()
    st.session_state.crypto_feed = None

# Indicator definitions with full names
INDICATORS = {
    # Ichimoku Cloud - THE MOST IMPORTANT (listed first)
    '🌩️ Ichimoku Kinko Hyo (一目均衡表) - Cloud Indicator': 'ichimoku',
    
    # Moving Averages
    'Simple Moving Average (SMA) - 20 Period': 'sma_20',
    'Simple Moving Average (SMA) - 50 Period': 'sma_50',
    'Exponential Moving Average (EMA) - 9 Period': 'ema_9',
    'Exponential Moving Average (EMA) - 21 Period': 'ema_21',
    
    # Volatility
    'Bollinger Bands (BB) - 20 Period, 2 StdDev': 'bb',
    'Average True Range (ATR) - 14 Period': 'atr',
    
    # Momentum
    'Relative Strength Index (RSI) - 14 Period': 'rsi',
    'Moving Average Convergence Divergence (MACD)': 'macd',
    'Stochastic Oscillator (%K, %D)': 'stoch',
    
    # Trend
    'Supertrend Indicator (ATR-Based)': 'supertrend',
    'Average Directional Index (ADX) - Trend Strength': 'adx',
    
    # Volume
    'Volume Weighted Average Price (VWAP)': 'vwap',
}


def load_data_local(symbol: str, file_path: str) -> pd.DataFrame:
    """Load data from local Excel file."""
    loader = HistoricalLoader()
    try:
        df = loader.load_excel(file_path, header_row=29, date_column="Exchange Date")
        return df
    except Exception as e:
        st.error(f"Error loading local file: {e}")
        return pd.DataFrame()


def load_data_api(symbol: str, api_key: str) -> pd.DataFrame:
    """Load data from Alpha Vantage API."""
    try:
        provider = AlphaVantageProvider(api_key)
        df = provider.get_historical_data(
            symbol=symbol,
            start_date=datetime.now() - timedelta(days=365),
            end_date=datetime.now(),
            interval="1d"
        )
        return df
    except Exception as e:
        st.error(f"Error fetching from API: {e}")
        return pd.DataFrame()


def add_indicators(df: pd.DataFrame, selected_indicators: list) -> pd.DataFrame:
    """Add selected technical indicators to the DataFrame."""
    df = df.copy()
    ti = TechnicalIndicators()
    ci = CustomIndicators()
    
    # Always add Ichimoku if selected (PRIMARY INDICATOR)
    ichimoku_key = '🌩️ Ichimoku Kinko Hyo (一目均衡表) - Cloud Indicator'
    if ichimoku_key in selected_indicators:
        df = ti.add_ichimoku(df)
    
    if 'Simple Moving Average (SMA) - 20 Period' in selected_indicators:
        df['sma_20'] = df['close'].rolling(20).mean()
    if 'Simple Moving Average (SMA) - 50 Period' in selected_indicators:
        df['sma_50'] = df['close'].rolling(50).mean()
    if 'Exponential Moving Average (EMA) - 9 Period' in selected_indicators:
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    if 'Exponential Moving Average (EMA) - 21 Period' in selected_indicators:
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    if 'Bollinger Bands (BB) - 20 Period, 2 StdDev' in selected_indicators:
        df = ti.add_bollinger_bands(df, period=20, std=2)
    if 'Relative Strength Index (RSI) - 14 Period' in selected_indicators:
        df = ti.add_rsi(df, period=14)
    if 'Moving Average Convergence Divergence (MACD)' in selected_indicators:
        df = ti.add_macd(df)
    if 'Supertrend Indicator (ATR-Based)' in selected_indicators:
        df = ci.add_supertrend(df, period=10, multiplier=3)
    if 'Average True Range (ATR) - 14 Period' in selected_indicators:
        df = ti.add_atr(df, period=14)
    if 'Stochastic Oscillator (%K, %D)' in selected_indicators:
        df = ti.add_stochastic(df)
    if 'Average Directional Index (ADX) - Trend Strength' in selected_indicators:
        df = ti.add_adx(df)
    if 'Volume Weighted Average Price (VWAP)' in selected_indicators:
        df = ti.add_vwap(df)
    
    return df


def create_chart(df: pd.DataFrame, selected_indicators: list, symbol: str) -> go.Figure:
    """Create interactive chart with indicators."""
    
    ichimoku_key = '🌩️ Ichimoku Kinko Hyo (一目均衡表) - Cloud Indicator'
    rsi_key = 'Relative Strength Index (RSI) - 14 Period'
    macd_key = 'Moving Average Convergence Divergence (MACD)'
    stoch_key = 'Stochastic Oscillator (%K, %D)'
    adx_key = 'Average Directional Index (ADX) - Trend Strength'
    
    # Determine number of rows based on indicators
    has_volume = True
    has_rsi = rsi_key in selected_indicators
    has_macd = macd_key in selected_indicators
    has_stoch = stoch_key in selected_indicators
    has_adx = adx_key in selected_indicators
    has_ichimoku = ichimoku_key in selected_indicators
    
    num_rows = 1 + int(has_volume) + int(has_rsi) + int(has_macd) + int(has_stoch) + int(has_adx)
    
    row_heights = [0.5]  # Main chart
    if has_volume:
        row_heights.append(0.1)
    if has_rsi:
        row_heights.append(0.1)
    if has_macd:
        row_heights.append(0.1)
    if has_stoch:
        row_heights.append(0.1)
    if has_adx:
        row_heights.append(0.1)
    
    # Create subplots
    fig = make_subplots(
        rows=num_rows, 
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=row_heights
    )
    
    # Candlestick chart
    fig.add_trace(
        go.Candlestick(
            x=df.index,
            open=df['open'],
            high=df['high'],
            low=df['low'],
            close=df['close'],
            name='Price (₹)',
            increasing_line_color='#16a34a',
            decreasing_line_color='#dc2626'
        ),
        row=1, col=1
    )
    
    # ===============================================
    # ICHIMOKU CLOUD - PRIMARY INDICATOR (一目均衡表)
    # ===============================================
    if has_ichimoku:
        # Tenkan-sen (Conversion Line) - 9-period
        if 'ichimoku_tenkan' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ichimoku_tenkan'], 
                name='天転換線 Tenkan-sen (Conversion Line)',
                line=dict(color='#3b82f6', width=1)
            ), row=1, col=1)
        
        # Kijun-sen (Base Line) - 26-period
        if 'ichimoku_kijun' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ichimoku_kijun'], 
                name='基準線 Kijun-sen (Base Line)',
                line=dict(color='#ef4444', width=1)
            ), row=1, col=1)
        
        # Chikou Span (Lagging Span) - 26-period lag
        if 'ichimoku_chikou' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ichimoku_chikou'], 
                name='遅行スパン Chikou Span (Lagging)',
                line=dict(color='#a855f7', width=1, dash='dot')
            ), row=1, col=1)
        
        # Senkou Span A & B (Cloud)
        if 'ichimoku_span_a' in df.columns and 'ichimoku_span_b' in df.columns:
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ichimoku_span_a'], 
                name='先行スパンA Senkou Span A',
                line=dict(color='#22c55e', width=1),
                fill=None
            ), row=1, col=1)
            
            fig.add_trace(go.Scatter(
                x=df.index, y=df['ichimoku_span_b'], 
                name='先行スパンB Senkou Span B',
                line=dict(color='#ef4444', width=1),
                fill='tonexty',
                fillcolor='rgba(34, 197, 94, 0.2)'
            ), row=1, col=1)
    
    # Add moving averages
    if 'Simple Moving Average (SMA) - 20 Period' in selected_indicators and 'sma_20' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_20'], name='SMA 20', 
                                  line=dict(color='orange', width=1)), row=1, col=1)
    if 'Simple Moving Average (SMA) - 50 Period' in selected_indicators and 'sma_50' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['sma_50'], name='SMA 50', 
                                  line=dict(color='purple', width=1)), row=1, col=1)
    if 'Exponential Moving Average (EMA) - 9 Period' in selected_indicators and 'ema_9' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_9'], name='EMA 9', 
                                  line=dict(color='cyan', width=1)), row=1, col=1)
    if 'Exponential Moving Average (EMA) - 21 Period' in selected_indicators and 'ema_21' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['ema_21'], name='EMA 21', 
                                  line=dict(color='yellow', width=1)), row=1, col=1)
    
    # Bollinger Bands
    if 'Bollinger Bands (BB) - 20 Period, 2 StdDev' in selected_indicators:
        if 'bb_upper' in df.columns:
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_upper'], name='Bollinger Upper',
                                      line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_middle'], name='Bollinger Middle',
                                      line=dict(color='gray', width=1)), row=1, col=1)
            fig.add_trace(go.Scatter(x=df.index, y=df['bb_lower'], name='Bollinger Lower',
                                      line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    
    # Supertrend
    if 'Supertrend Indicator (ATR-Based)' in selected_indicators and 'supertrend' in df.columns:
        buy_signal = df[df['supertrend_direction'] == 1]
        sell_signal = df[df['supertrend_direction'] == -1]
        fig.add_trace(go.Scatter(x=buy_signal.index, y=buy_signal['supertrend'], 
                                  name='Supertrend Bullish', mode='lines',
                                  line=dict(color='green', width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sell_signal.index, y=sell_signal['supertrend'], 
                                  name='Supertrend Bearish', mode='lines',
                                  line=dict(color='red', width=2)), row=1, col=1)
    
    current_row = 2
    
    # Volume
    if has_volume:
        colors = ['#16a34a' if c >= o else '#dc2626' for c, o in zip(df['close'], df['open'])]
        fig.add_trace(go.Bar(x=df.index, y=df['volume'], name='Volume',
                              marker_color=colors, opacity=0.7), row=current_row, col=1)
        current_row += 1
    
    # RSI
    if has_rsi and 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['rsi'], name='RSI (14)',
                                  line=dict(color='purple', width=1)), row=current_row, col=1)
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # MACD
    if has_macd and 'macd' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['macd'], name='MACD Line',
                                  line=dict(color='blue', width=1)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['macd_signal'], name='MACD Signal',
                                  line=dict(color='orange', width=1)), row=current_row, col=1)
        colors = ['green' if h >= 0 else 'red' for h in df['macd_histogram']]
        fig.add_trace(go.Bar(x=df.index, y=df['macd_histogram'], name='MACD Histogram',
                              marker_color=colors), row=current_row, col=1)
        current_row += 1
    
    # Stochastic
    if has_stoch and 'stoch_k' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_k'], name='Stochastic %K',
                                  line=dict(color='blue', width=1)), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df.index, y=df['stoch_d'], name='Stochastic %D',
                                  line=dict(color='orange', width=1)), row=current_row, col=1)
        fig.add_hline(y=80, line_dash="dash", line_color="red", row=current_row, col=1)
        fig.add_hline(y=20, line_dash="dash", line_color="green", row=current_row, col=1)
        current_row += 1
    
    # ADX
    if has_adx and 'adx' in df.columns:
        fig.add_trace(go.Scatter(x=df.index, y=df['adx'], name='ADX (Trend Strength)',
                                  line=dict(color='purple', width=2)), row=current_row, col=1)
        fig.add_hline(y=25, line_dash="dash", line_color="gray", row=current_row, col=1)
        current_row += 1
    
    # Update layout
    title = f'{symbol} - Technical Analysis'
    if has_ichimoku:
        title = f'{symbol} - 一目均衡表 Ichimoku Kinko Hyo Analysis'
    
    fig.update_layout(
        title=title,
        height=900,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        xaxis_rangeslider_visible=False,
        template='plotly_dark',
        yaxis_title=f'Price ({CURRENCY})'
    )
    
    return fig


def execute_trade(symbol: str, side: str, quantity: int, price: float):
    """Execute a paper trade."""
    paper_trader = st.session_state.paper_trader
    order_manager = st.session_state.order_manager
    
    # Set current price
    paper_trader.set_prices({symbol: price})
    
    # Create and submit order
    order = order_manager.create_order(
        symbol=symbol,
        side=side,
        quantity=quantity,
        order_type='market'
    )
    
    success = order_manager.submit_order(order)
    
    if success:
        st.session_state.trades.append({
            'time': datetime.now(),
            'symbol': symbol,
            'side': side.upper(),
            'quantity': quantity,
            'price': price,
            'value': quantity * price
        })
        return True
    return False


# ============================================================
# MAIN APP
# ============================================================

# Dynamic header based on market mode
if 'market_mode' in st.session_state and st.session_state.market_mode == "Crypto":
    st.markdown('<h1 class="main-header">🪙 Crypto Trading Bot - Binance (24/7)</h1>', unsafe_allow_html=True)
else:
    st.markdown('<h1 class="main-header">📈 Trading Bot Dashboard - NSE/BSE</h1>', unsafe_allow_html=True)

# Ichimoku Cloud info banner
st.markdown("""
<div class="indicator-info">
    <span class="ichimoku-badge">🌩️ Primary Indicator: Ichimoku Kinko Hyo (一目均衡表)</span>
    <p style="margin-top:0.5rem; color: #94a3b8;">
        The Ichimoku Cloud provides 5 key signals: Tenkan-sen, Kijun-sen, Senkou Span A/B (Cloud), and Chikou Span.
        It offers a complete view of support, resistance, momentum, and trend direction at a glance.
    </p>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.header("⚙️ Settings")

# Market Mode Selector - NEW!
st.sidebar.subheader("🌐 Market Mode")
market_mode = st.sidebar.radio(
    "Select Market",
    ["📈 Stocks (NSE/BSE)", "🪙 Crypto (24/7 Live)"],
    index=0 if st.session_state.market_mode == "Stocks" else 1
)
st.session_state.market_mode = "Stocks" if "Stocks" in market_mode else "Crypto"

# Dynamic currency based on market
if st.session_state.market_mode == "Crypto":
    CURRENCY_DISPLAY = "$"
else:
    CURRENCY_DISPLAY = CURRENCY  # INR

# Data Source (conditional based on market mode)
if st.session_state.market_mode == "Stocks":
    st.sidebar.subheader("📂 Data Source")
    data_source = st.sidebar.radio("Select Data Source", ["Local Excel Files (NSE/BSE)", "Alpha Vantage API"])
    
    if data_source == "Local Excel Files (NSE/BSE)":
        local_files = {
            "ADANIENT - Adani Enterprises (Daily)": "stock_data/Adani enterprise annual.xlsx",
            "ASIANPAINT - Asian Paints Ltd (Daily)": "stock_data/Asian Paints Annual.xlsx",
            "ADANIENT - Adani Enterprises (5-min Intraday)": "stock_data/Adani enterprise 5 min.xlsx",
            "ASIANPAINT - Asian Paints Ltd (5-min Intraday)": "stock_data/Asian paints 5 min.xlsx",
        }
        selected_file = st.sidebar.selectbox("Select Stock", list(local_files.keys()))
        symbol = selected_file.split(" - ")[0]
    else:
        symbol = st.sidebar.text_input("Stock Symbol", "RELIANCE.BSE")
        api_key = st.sidebar.text_input("Alpha Vantage API Key", "9N8KYMVUSI2VRBOZ", type="password")
    
    data_source_type = "stocks"

else:
    # CRYPTO MODE
    st.sidebar.subheader("🪙 Crypto Settings")
    st.sidebar.markdown("*Live data from Binance (24/7)*")
    
    # Crypto pair selector
    crypto_pairs = BinanceCryptoProvider.get_available_pairs()
    selected_crypto = st.sidebar.selectbox(
        "Select Crypto Pair",
        list(crypto_pairs.keys()),
        format_func=lambda x: f"{x} - {crypto_pairs[x]}"
    )
    symbol = selected_crypto
    
    # Interval selector
    intervals = BinanceCryptoProvider.get_available_intervals()
    selected_interval = st.sidebar.selectbox(
        "Candle Interval",
        list(intervals.keys()),
        index=2,  # Default to 15m
        format_func=lambda x: f"{x} ({intervals[x]})"
    )
    
    # History bars
    history_bars = st.sidebar.slider("History Bars", 100, 500, 200)
    
    data_source_type = "crypto"

# Technical Indicators - Ichimoku FIRST and DEFAULT
st.sidebar.subheader("📊 Technical Indicators")
st.sidebar.markdown("*🌩️ Ichimoku Cloud is selected by default*")

available_indicators = list(INDICATORS.keys())

# Default: Ichimoku Cloud + RSI + MACD
default_indicators = [
    '🌩️ Ichimoku Kinko Hyo (一目均衡表) - Cloud Indicator',
    'Relative Strength Index (RSI) - 14 Period',
    'Moving Average Convergence Divergence (MACD)'
]

selected_indicators = st.sidebar.multiselect(
    "Select Indicators (Ichimoku recommended)",
    available_indicators,
    default=default_indicators
)

# Load Data Button
if st.sidebar.button("📥 Load Data", type="primary"):
    with st.spinner("Loading data..."):
        if data_source_type == "stocks":
            if data_source == "Local Excel Files (NSE/BSE)":
                file_path = local_files[selected_file]
                df = load_data_local(symbol, file_path)
            else:
                df = load_data_api(symbol, api_key)
        else:
            # CRYPTO - Load from Binance
            try:
                crypto_provider = st.session_state.crypto_provider
                df = crypto_provider.get_historical_klines(
                    symbol=symbol,
                    interval=selected_interval,
                    limit=history_bars
                )
                
                # Create live feed for real-time updates
                st.session_state.crypto_feed = LiveCryptoFeed(
                    symbol=symbol,
                    interval=selected_interval,
                    history_bars=history_bars
                )
                
            except Exception as e:
                st.error(f"Error loading crypto data: {e}")
                df = pd.DataFrame()
        
        if not df.empty:
            df = add_indicators(df, selected_indicators)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.current_symbol = symbol
            st.session_state.selected_indicators = selected_indicators
            
            if data_source_type == "crypto":
                st.success(f"✅ Loaded {len(df)} candles for {symbol} (Live 24/7 data)")
            else:
                st.success(f"✅ Loaded {len(df)} data points for {symbol}")
        else:
            st.error("Failed to load data")

# Main Content
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    symbol = st.session_state.current_symbol
    
    # Create tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📈 Charts & Ichimoku", "🎯 Paper Trading", "📋 Positions & Orders", "🤖 Auto Trading", "📊 P&L Analysis", "📚 Indicator Guide"])
    
    with tab1:
        # Timeline filter
        st.markdown("### ⏱️ Timeline Filter")
        timeline_col1, timeline_col2, timeline_col3 = st.columns([2, 2, 2])
        
        with timeline_col1:
            timeline_options = ["All Data", "1 Week", "1 Month", "3 Months", "6 Months", "1 Year", "Custom"]
            timeline_filter = st.selectbox("Select Timeframe", timeline_options, index=0)
        
        with timeline_col2:
            if timeline_filter == "Custom":
                start_date = st.date_input("Start Date", df.index.min().date() if hasattr(df.index.min(), 'date') else df.index.min())
            else:
                start_date = None
        
        with timeline_col3:
            if timeline_filter == "Custom":
                end_date = st.date_input("End Date", df.index.max().date() if hasattr(df.index.max(), 'date') else df.index.max())
            else:
                end_date = None
        
        # Apply timeline filter
        if timeline_filter != "All Data":
            if timeline_filter == "1 Week":
                filter_start = df.index.max() - pd.Timedelta(days=7)
            elif timeline_filter == "1 Month":
                filter_start = df.index.max() - pd.Timedelta(days=30)
            elif timeline_filter == "3 Months":
                filter_start = df.index.max() - pd.Timedelta(days=90)
            elif timeline_filter == "6 Months":
                filter_start = df.index.max() - pd.Timedelta(days=180)
            elif timeline_filter == "1 Year":
                filter_start = df.index.max() - pd.Timedelta(days=365)
            elif timeline_filter == "Custom" and start_date and end_date:
                filter_start = pd.Timestamp(start_date)
                filter_end = pd.Timestamp(end_date)
                df_filtered = df[(df.index >= filter_start) & (df.index <= filter_end)]
            
            if timeline_filter != "Custom":
                df_filtered = df[df.index >= filter_start]
            
            if len(df_filtered) > 0:
                chart_df = df_filtered
            else:
                chart_df = df
                st.warning("No data in selected range, showing all data")
        else:
            chart_df = df
        
        # Display chart
        fig = create_chart(chart_df, selected_indicators, symbol)
        st.plotly_chart(fig, use_container_width=True)
        
        # Current price and indicators
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            current_price = df['close'].iloc[-1]
            prev_price = df['close'].iloc[-2]
            change = ((current_price - prev_price) / prev_price) * 100
            st.metric("Current Price", f"{CURRENCY}{current_price:,.2f}", f"{change:+.2f}%")
        
        with col2:
            if 'rsi' in df.columns:
                rsi = df['rsi'].iloc[-1]
                st.metric("RSI (14)", f"{rsi:.1f}", 
                         "Overbought" if rsi > 70 else "Oversold" if rsi < 30 else "Neutral")
        
        with col3:
            if 'macd' in df.columns:
                macd = df['macd'].iloc[-1]
                signal = df['macd_signal'].iloc[-1]
                st.metric("MACD", f"{macd:.2f}", "Bullish" if macd > signal else "Bearish")
        
        with col4:
            # Open Interest (simulated for derivatives)
            import random
            simulated_oi = random.randint(50000, 200000)
            oi_change = random.uniform(-5, 5)
            st.metric("Open Interest (OI)", f"{simulated_oi:,}", f"{oi_change:+.2f}%")
        
        # Additional row for ATR if available
        if 'atr' in df.columns:
            st.markdown("---")
            atr_col1, atr_col2, atr_col3, atr_col4 = st.columns(4)
            with atr_col1:
                atr = df['atr'].iloc[-1]
                st.metric("ATR (Volatility)", f"{CURRENCY}{atr:.2f}")
        
        # Ichimoku Analysis Summary
        ichimoku_key = '🌩️ Ichimoku Kinko Hyo (一目均衡表) - Cloud Indicator'
        if ichimoku_key in selected_indicators:
            st.subheader("🌩️ Ichimoku Cloud Analysis")
            
            col1, col2, col3, col4, col5 = st.columns(5)
            
            with col1:
                if 'ichimoku_tenkan' in df.columns:
                    tenkan = df['ichimoku_tenkan'].iloc[-1]
                    st.metric("天転換線 Tenkan-sen", f"{CURRENCY}{tenkan:,.2f}")
            
            with col2:
                if 'ichimoku_kijun' in df.columns:
                    kijun = df['ichimoku_kijun'].iloc[-1]
                    st.metric("基準線 Kijun-sen", f"{CURRENCY}{kijun:,.2f}")
            
            with col3:
                if 'ichimoku_span_a' in df.columns:
                    span_a = df['ichimoku_span_a'].iloc[-1]
                    st.metric("先行スパンA Span A", f"{CURRENCY}{span_a:,.2f}")
            
            with col4:
                if 'ichimoku_span_b' in df.columns:
                    span_b = df['ichimoku_span_b'].iloc[-1]
                    st.metric("先行スパンB Span B", f"{CURRENCY}{span_b:,.2f}")
            
            with col5:
                # Cloud Signal
                if 'ichimoku_span_a' in df.columns and 'ichimoku_span_b' in df.columns:
                    current_price = df['close'].iloc[-1]
                    span_a = df['ichimoku_span_a'].iloc[-1]
                    span_b = df['ichimoku_span_b'].iloc[-1]
                    cloud_top = max(span_a, span_b)
                    cloud_bottom = min(span_a, span_b)
                    
                    if current_price > cloud_top:
                        signal = "🟢 BULLISH"
                    elif current_price < cloud_bottom:
                        signal = "🔴 BEARISH"
                    else:
                        signal = "🟡 IN CLOUD"
                    st.metric("Cloud Signal", signal)
    
    with tab2:
        st.subheader("🎯 Paper Trading")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Trading form
            current_price = df['close'].iloc[-1]
            st.write(f"**Current Price:** {CURRENCY}{current_price:,.2f}")
            
            trade_side = st.radio("Trade Side", ["BUY", "SELL"], horizontal=True, key="trade_side_radio")
            
            # Market Depth display when side is selected
            st.session_state.paper_trader.set_prices({symbol: current_price})
            market_depth = st.session_state.paper_trader.get_market_depth(symbol)
            
            st.markdown("#### 📊 Market Depth (Order Book)")
            depth_col1, depth_col2 = st.columns(2)
            
            with depth_col1:
                st.markdown("**🟢 Bids (Buy Orders)**")
                bids_df = pd.DataFrame(market_depth['bids'])
                bids_df.columns = ['Price (₹)', 'Volume']
                st.dataframe(bids_df, use_container_width=True, hide_index=True)
            
            with depth_col2:
                st.markdown("**🔴 Asks (Sell Orders)**")
                asks_df = pd.DataFrame(market_depth['asks'])
                asks_df.columns = ['Price (₹)', 'Volume']
                st.dataframe(asks_df, use_container_width=True, hide_index=True)
            
            st.markdown("---")
            
            account = st.session_state.paper_trader.get_account_info()
            max_shares = int(account['cash'] / current_price) if current_price > 0 else 1
            
            quantity = st.number_input(
                "Quantity (Shares)", 
                min_value=1, 
                max_value=max(max_shares, 1), 
                value=min(10, max(max_shares, 1))
            )
            
            trade_value = quantity * current_price
            st.write(f"**Trade Value:** {CURRENCY}{trade_value:,.2f}")
            
            if st.button(f"Execute {trade_side}", type="primary"):
                success = execute_trade(
                    symbol, 
                    trade_side.lower(), 
                    quantity, 
                    current_price
                )
                if success:
                    st.success(f"✅ {trade_side} {quantity} shares of {symbol} @ {CURRENCY}{current_price:,.2f}")
                    st.rerun()
                else:
                    st.error("Trade execution failed")
        
        with col2:
            # Account summary
            st.subheader("💰 Account Summary (INR)")
            account = st.session_state.paper_trader.get_account_info()
            
            st.metric("Cash Available", f"{CURRENCY}{account['cash']:,.2f}")
            st.metric("Positions Value", f"{CURRENCY}{account['positions_value']:,.2f}")
            st.metric("Total Equity", f"{CURRENCY}{account['total_value']:,.2f}")
            
            pnl = account['pnl']
            pnl_pct = (pnl / INITIAL_CAPITAL) * 100
            st.metric("P&L", f"{CURRENCY}{pnl:+,.2f}", f"{pnl_pct:+.2f}%")
            
            # Calculate and show Avg Buying Price for current positions
            positions = st.session_state.paper_trader.get_positions()
            if positions:
                total_invested = sum(p['avg_price'] * p['quantity'] for p in positions)
                total_qty = sum(p['quantity'] for p in positions)
                if total_qty > 0:
                    avg_buying = total_invested / total_qty
                    st.metric("Avg Buying Price", f"{CURRENCY}{avg_buying:,.2f}")
    
    with tab3:
        st.subheader("📋 Positions & Orders")
        
        # Sub-tabs for Open Positions, Closed Positions, Trade History, Pending Orders
        pos_tab1, pos_tab2, pos_tab3, pos_tab4 = st.tabs(["📈 Open Positions", "✅ Closed Positions", "📜 Trade History", "⏳ Pending Orders"])
        
        with pos_tab1:
            st.markdown("### Open Positions")
            positions = st.session_state.paper_trader.get_positions()
            
            if positions:
                pos_df = pd.DataFrame(positions)
                # Rename unrealized_pnl to net_position
                if 'unrealized_pnl' in pos_df.columns:
                    pos_df = pos_df.rename(columns={
                        'unrealized_pnl': 'net_position',
                        'unrealized_pnl_pct': 'net_position_pct'
                    })
                
                # Display with custom formatting
                st.dataframe(
                    pos_df,
                    column_config={
                        "symbol": "Symbol",
                        "quantity": "Qty",
                        "avg_price": st.column_config.NumberColumn("Avg Buying Price", format=f"{CURRENCY}%.2f"),
                        "current_price": st.column_config.NumberColumn("Current Price", format=f"{CURRENCY}%.2f"),
                        "market_value": st.column_config.NumberColumn("Market Value", format=f"{CURRENCY}%.2f"),
                        "net_position": st.column_config.NumberColumn("Net Position (P&L)", format=f"{CURRENCY}%.2f"),
                        "net_position_pct": st.column_config.NumberColumn("Net Position %", format="%.2f%%"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary
                total_invested = sum(p['avg_price'] * p['quantity'] for p in positions)
                total_market_value = sum(p['market_value'] for p in positions)
                total_net_position = sum(p.get('net_position', p.get('unrealized_pnl', 0)) for p in positions)
                
                st.markdown("---")
                summary_col1, summary_col2, summary_col3 = st.columns(3)
                with summary_col1:
                    st.metric("Total Invested", f"{CURRENCY}{total_invested:,.2f}")
                with summary_col2:
                    st.metric("Total Market Value", f"{CURRENCY}{total_market_value:,.2f}")
                with summary_col3:
                    st.metric("Total Net Position", f"{CURRENCY}{total_net_position:+,.2f}")
            else:
                st.info("No open positions. Start trading in the Paper Trading tab!")
        
        with pos_tab2:
            st.markdown("### Closed Positions")
            closed_positions = st.session_state.paper_trader.get_closed_positions()
            
            if closed_positions:
                closed_df = pd.DataFrame(closed_positions)
                closed_df['closed_at'] = pd.to_datetime(closed_df['closed_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    closed_df,
                    column_config={
                        "symbol": "Symbol",
                        "quantity": "Qty Sold",
                        "entry_price": st.column_config.NumberColumn("Entry Price", format=f"{CURRENCY}%.2f"),
                        "exit_price": st.column_config.NumberColumn("Exit Price", format=f"{CURRENCY}%.2f"),
                        "realized_pnl": st.column_config.NumberColumn("Realized P&L", format=f"{CURRENCY}%.2f"),
                        "commission": st.column_config.NumberColumn("Commission", format=f"{CURRENCY}%.2f"),
                        "closed_at": "Closed At",
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Summary
                total_realized = sum(p['realized_pnl'] for p in closed_positions)
                total_commission = sum(p['commission'] for p in closed_positions)
                
                st.markdown("---")
                closed_summary_col1, closed_summary_col2 = st.columns(2)
                with closed_summary_col1:
                    st.metric("Total Realized P&L", f"{CURRENCY}{total_realized:+,.2f}")
                with closed_summary_col2:
                    st.metric("Total Commissions", f"{CURRENCY}{total_commission:,.2f}")
            else:
                st.info("No closed positions yet. Sell some positions to see them here.")
        
        with pos_tab3:
            st.markdown("### Trade History")
            
            if st.session_state.trades:
                trades_df = pd.DataFrame(st.session_state.trades)
                trades_df['time'] = trades_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    trades_df,
                    column_config={
                        "time": "Time",
                        "symbol": "Symbol",
                        "side": st.column_config.TextColumn("Side"),
                        "quantity": "Qty",
                        "price": st.column_config.NumberColumn("Price", format=f"{CURRENCY}%.2f"),
                        "value": st.column_config.NumberColumn("Value", format=f"{CURRENCY}%.2f"),
                    },
                    use_container_width=True,
                    hide_index=True
                )
            else:
                st.info("No trades executed yet.")
        
        with pos_tab4:
            st.markdown("### Pending Orders")
            pending_orders = st.session_state.paper_trader.get_pending_orders()
            
            if pending_orders:
                pending_df = pd.DataFrame(pending_orders)
                pending_df['created_at'] = pd.to_datetime(pending_df['created_at']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(
                    pending_df,
                    column_config={
                        "order_id": "Order ID",
                        "symbol": "Symbol",
                        "side": "Side",
                        "order_type": "Type",
                        "quantity": "Qty",
                        "price": st.column_config.NumberColumn("Limit Price", format=f"{CURRENCY}%.2f"),
                        "stop_price": st.column_config.NumberColumn("Stop Price", format=f"{CURRENCY}%.2f"),
                        "status": "Status",
                        "created_at": "Created At",
                    },
                    use_container_width=True,
                    hide_index=True
                )
                
                # Cancel order functionality
                st.markdown("---")
                order_to_cancel = st.selectbox(
                    "Select Order to Cancel",
                    [o['order_id'] for o in pending_orders],
                    key="cancel_order_select"
                )
                
                if st.button("❌ Cancel Selected Order", type="secondary"):
                    # Find the order and cancel it
                    for order_id, order in st.session_state.paper_trader.pending_orders.items():
                        if order_id == order_to_cancel:
                            st.session_state.paper_trader.cancel_order(order)
                            st.success(f"Order {order_to_cancel} cancelled!")
                            st.rerun()
                            break
            else:
                st.info("No pending orders. Place a limit or stop order to see them here.")
                st.markdown("*Note: Market orders are executed immediately and won't appear here.*")
    
    with tab4:
        st.subheader("🤖 Automated Trading with Ichimoku Cloud")
        
        # Information banner
        st.markdown("""
        <div class="indicator-info">
            <span class="ichimoku-badge">🌩️ Ichimoku Cloud Strategy</span>
            <p style="margin-top:0.5rem; color: #94a3b8;">
                Uses 4-signal confluence: Kumo Breakout, TK Cross, Cloud Twist, Chikou Confirmation.
                Trades are executed automatically when 3+ signals align.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Control columns
        ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 1, 2])
        
        with ctrl_col1:
            # Initialize auto-trader if not exists
            if st.session_state.auto_trader is None:
                st.session_state.auto_trader = IchimokuAutoTrader(
                    paper_trader=st.session_state.paper_trader,
                    order_manager=st.session_state.order_manager,
                    check_interval_seconds=0.5,
                    min_confluence=3,
                    position_size_pct=0.10
                )
            
            # Start/Stop Button
            is_running = st.session_state.auto_trader.is_running()
            
            if is_running:
                if st.button("🛑 STOP Auto-Trading", type="primary", use_container_width=True):
                    report = st.session_state.auto_trader.stop()
                    st.session_state.auto_trade_report = report
                    st.session_state.auto_trading_active = False
                    st.rerun()
            else:
                if st.button("🚀 START Auto-Trading", type="primary", use_container_width=True):
                    # Set data and start
                    st.session_state.auto_trader.set_data(df, symbol)
                    if st.session_state.auto_trader.start():
                        st.session_state.auto_trading_active = True
                        st.session_state.auto_trade_report = None
                        st.rerun()
                    else:
                        st.error("Failed to start auto-trading. Ensure data is loaded.")
        
        with ctrl_col2:
            # Status indicator
            if is_running:
                st.markdown("### 🟢 RUNNING")
                status = st.session_state.auto_trader.get_status()
                st.caption(f"Progress: {status['progress_pct']:.1f}%")
                st.caption(f"Trades: {status['total_trades']}")
            else:
                st.markdown("### 🔴 STOPPED")
                st.caption("Click START to begin")
        
        with ctrl_col3:
            # Configuration
            with st.expander("⚙️ Strategy Settings"):
                confluence = st.slider("Min Confluence (signals)", 2, 4, 3, key="auto_confluence")
                position_size = st.slider("Position Size (%)", 5, 25, 10, key="auto_pos_size")
                check_interval = st.slider("Check Interval (sec)", 0.1, 2.0, 0.5, key="auto_interval")
                
                if st.button("Apply Settings"):
                    st.session_state.auto_trader.min_confluence = confluence
                    st.session_state.auto_trader.position_size_pct = position_size / 100
                    st.session_state.auto_trader.check_interval = check_interval
                    st.success("Settings applied!")
        
        st.markdown("---")
        
        # Live feed or Report display
        if is_running:
            # Show live status
            st.subheader("📊 Live Trading Status")
            
            status = st.session_state.auto_trader.get_status()
            
            # Progress bar
            st.progress(status['progress_pct'] / 100)
            
            # Current stats
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            with stat_col1:
                st.metric("Total Trades", status['total_trades'])
            with stat_col2:
                st.metric("Current P&L", f"{CURRENCY}{status['current_pnl']:+,.2f}")
            with stat_col3:
                st.metric("Current Bar", f"{status['current_bar']}/{status['total_bars']}")
            with stat_col4:
                if status['latest_signal']:
                    signal = status['latest_signal']
                    st.metric("Latest Signal", signal.signal_type, f"{signal.confidence*100:.0f}%")
            
            # Recent signals
            st.subheader("📡 Recent Signals")
            recent_signals = st.session_state.auto_trader.signal_history[-10:]
            if recent_signals:
                for sig in reversed(recent_signals):
                    if sig.signal_type == "BUY":
                        st.success(f"🟢 BUY @ ₹{sig.price:,.2f} | Confidence: {sig.confidence*100:.0f}% | {', '.join(sig.reasons[:2])}")
                    elif sig.signal_type == "SELL":
                        st.error(f"🔴 SELL @ ₹{sig.price:,.2f} | Confidence: {sig.confidence*100:.0f}% | {', '.join(sig.reasons[:2])}")
                    else:
                        st.info(f"⚪ HOLD @ ₹{sig.price:,.2f}")
            
            # Auto-refresh hint
            st.caption("💡 The dashboard updates when you interact with it. Click anywhere to refresh.")
            
        elif st.session_state.auto_trade_report:
            # Show report
            st.subheader("📋 Trading Session Report")
            
            report = st.session_state.auto_trade_report
            report_md = TradingReportGenerator.format_report_for_display(report)
            st.markdown(report_md)
            
            # Download button
            if st.button("📥 Download Report as CSV"):
                trades_df = TradingReportGenerator.trades_to_dataframe(report)
                if not trades_df.empty:
                    csv = trades_df.to_csv(index=False)
                    st.download_button(
                        label="Download",
                        data=csv,
                        file_name=f"trading_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
        else:
            # Instructions
            st.info("""
            ### How it works:
            1. **Load data** from the sidebar
            2. Click **START** to begin automated trading
            3. The bot uses Ichimoku Cloud strategy with 4-signal confluence:
               - 🌩️ **Kumo Breakout**: Price above/below cloud
               - 📈 **TK Cross**: Tenkan crosses Kijun
               - 🔄 **Cloud Twist**: Span A vs Span B
               - 👁️ **Chikou Confirmation**: Lagging span position
            4. Trades execute when 3+ signals align
            5. Click **STOP** to end and generate report
            """)
    
    with tab5:
        st.subheader("📊 P&L Analysis")
        
        # Account metrics
        account = st.session_state.paper_trader.get_account_info()
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Starting Capital", f"{CURRENCY}{INITIAL_CAPITAL:,.2f}")
        with col2:
            st.metric("Current Equity", f"{CURRENCY}{account['total_value']:,.2f}")
        with col3:
            pnl = account['pnl']
            st.metric("Total P&L", f"{CURRENCY}{pnl:+,.2f}")
        with col4:
            pnl_pct = (pnl / INITIAL_CAPITAL) * 100
            st.metric("Return %", f"{pnl_pct:+.2f}%")
        
        # Trade history
        st.subheader("📜 Trade History")
        
        if st.session_state.trades:
            trades_df = pd.DataFrame(st.session_state.trades)
            trades_df['time'] = trades_df['time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            
            # Style the dataframe
            st.dataframe(
                trades_df,
                column_config={
                    "time": "Time",
                    "symbol": "Symbol",
                    "side": st.column_config.TextColumn("Side"),
                    "quantity": "Qty",
                    "price": st.column_config.NumberColumn("Price", format=f"{CURRENCY}%.2f"),
                    "value": st.column_config.NumberColumn("Value", format=f"{CURRENCY}%.2f"),
                },
                use_container_width=True,
                hide_index=True
            )
            
            # Summary stats
            if len(st.session_state.trades) > 0:
                st.subheader("📈 Trade Statistics")
                total_buy = sum(t['value'] for t in st.session_state.trades if t['side'] == 'BUY')
                total_sell = sum(t['value'] for t in st.session_state.trades if t['side'] == 'SELL')
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Trades", len(st.session_state.trades))
                with col2:
                    st.metric("Total Bought", f"{CURRENCY}{total_buy:,.2f}")
                with col3:
                    st.metric("Total Sold", f"{CURRENCY}{total_sell:,.2f}")
        else:
            st.info("No trades executed yet. Start trading in the Paper Trading tab!")
        
        # Reset button
        if st.button("🔄 Reset Account"):
            st.session_state.paper_trader = PaperTrader(initial_capital=INITIAL_CAPITAL)
            st.session_state.order_manager = OrderManager(st.session_state.paper_trader)
            st.session_state.paper_trader.set_order_manager(st.session_state.order_manager)
            st.session_state.trades = []
            st.success(f"Account reset to {CURRENCY}{INITIAL_CAPITAL:,.2f}")
            st.rerun()
    
    with tab6:
        st.subheader("📚 Technical Indicator Guide")
        
        st.markdown("""
        ### 🌩️ Ichimoku Kinko Hyo (一目均衡表) - PRIMARY INDICATOR
        
        The **Ichimoku Cloud** (一目均衡表, literally "one glance equilibrium chart") is the most comprehensive 
        technical indicator, developed by Japanese journalist Goichi Hosoda in 1936.
        
        #### Components:
        
        | Component | Japanese | Description |
        |-----------|----------|-------------|
        | **Tenkan-sen** | 天転換線 | Conversion Line (9-period high+low)/2 |
        | **Kijun-sen** | 基準線 | Base Line (26-period high+low)/2 |
        | **Senkou Span A** | 先行スパンA | Leading Span A (Tenkan + Kijun)/2, plotted 26 periods ahead |
        | **Senkou Span B** | 先行スパンB | Leading Span B (52-period high+low)/2, plotted 26 periods ahead |
        | **Chikou Span** | 遅行スパン | Lagging Span (Close price plotted 26 periods behind) |
        
        #### Trading Signals:
        - **Bullish**: Price above cloud, Tenkan > Kijun
        - **Bearish**: Price below cloud, Tenkan < Kijun
        - **Strong Trend**: Thick cloud in trend direction
        - **TK Cross**: Tenkan crosses Kijun (golden/death cross)
        
        ---
        
        ### Other Indicators Available:
        
        | Indicator | Full Name | Purpose |
        |-----------|-----------|---------|
        | SMA | Simple Moving Average | Trend identification |
        | EMA | Exponential Moving Average | Faster trend response |
        | BB | Bollinger Bands | Volatility measurement |
        | RSI | Relative Strength Index | Overbought/oversold |
        | MACD | Moving Average Convergence Divergence | Momentum |
        | Stochastic | Stochastic Oscillator | Momentum reversals |
        | ADX | Average Directional Index | Trend strength |
        | ATR | Average True Range | Volatility |
        | VWAP | Volume Weighted Average Price | Fair value |
        | Supertrend | ATR-based Trend | Trend following |
        """)

else:
    # Welcome screen
    st.info("👈 Select a data source and click 'Load Data' to get started!")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        ### 🌩️ Ichimoku Cloud
        - Primary indicator (一目均衡表)
        - Complete market view
        - Support/Resistance levels
        - Trend direction at a glance
        """)
    
    with col2:
        st.markdown("""
        ### 🎯 Paper Trading
        - Risk-free simulation
        - Indian Rupee (₹) currency
        - Track your trades
        - Real NIFTY 50 data
        """)
    
    with col3:
        st.markdown("""
        ### 📈 P&L Tracking
        - Live profit/loss in ₹
        - Trade history
        - Performance metrics
        - Starting capital: ₹10 Lakhs
        """)
    
    # Comparison with TradingView
    st.markdown("---")
    st.subheader("📊 How does this compare to TradingView Paper Trading?")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        #### ✅ Advantages over TradingView:
        - **Free & Unlimited** - No subscription needed
        - **Indian Market Focus** - NSE/BSE stocks with ₹ currency
        - **Custom Strategies** - Use your own indicators
        - **Local Data** - Works offline with Excel data
        - **Ichimoku Emphasis** - Prominent cloud visualization
        - **Full Source Code** - Customize anything
        - **No Account Required** - Instant access
        """)
    
    with col2:
        st.markdown("""
        #### 🔄 TradingView has:
        - Real-time streaming data
        - More chart drawing tools
        - Social features
        - Mobile app
        - Pine Script
        
        *Our dashboard focuses on serious technical analysis
        with Ichimoku Cloud as the primary indicator.*
        """)

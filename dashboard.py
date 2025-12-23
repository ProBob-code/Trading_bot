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
from src.indicators.technical import TechnicalIndicators
from src.indicators.custom import CustomIndicators
from src.strategy.ta_strategy import TAStrategy
from src.execution.brokers.paper_trader import PaperTrader
from src.execution.order_manager import OrderManager

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

# Data Source
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
        if data_source == "Local Excel Files (NSE/BSE)":
            file_path = local_files[selected_file]
            df = load_data_local(symbol, file_path)
        else:
            df = load_data_api(symbol, api_key)
        
        if not df.empty:
            df = add_indicators(df, selected_indicators)
            st.session_state.df = df
            st.session_state.data_loaded = True
            st.session_state.current_symbol = symbol
            st.session_state.selected_indicators = selected_indicators
            st.success(f"✅ Loaded {len(df)} data points for {symbol}")
        else:
            st.error("Failed to load data")

# Main Content
if st.session_state.data_loaded and st.session_state.df is not None:
    df = st.session_state.df
    symbol = st.session_state.current_symbol
    
    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Charts & Ichimoku", "🎯 Paper Trading", "📊 P&L Analysis", "📚 Indicator Guide"])
    
    with tab1:
        # Display chart
        fig = create_chart(df, selected_indicators, symbol)
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
            if 'atr' in df.columns:
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
            
            trade_side = st.radio("Trade Side", ["BUY", "SELL"], horizontal=True)
            
            account = st.session_state.paper_trader.get_account_info()
            max_shares = int(account['cash'] / current_price)
            
            quantity = st.number_input(
                "Quantity (Shares)", 
                min_value=1, 
                max_value=max(max_shares, 1), 
                value=min(10, max_shares)
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
        
        # Current positions
        st.subheader("📋 Current Positions")
        positions = st.session_state.paper_trader.get_positions()
        
        if positions:
            pos_df = pd.DataFrame(positions)
            st.dataframe(pos_df, use_container_width=True)
        else:
            st.info("No open positions")
    
    with tab3:
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
    
    with tab4:
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

# Trading Bot Dashboard - NSE/BSE 📈

A professional trading dashboard with **Ichimoku Cloud (一目均衡表)** technical analysis and paper trading simulation for Indian stock markets.

## 🌐 Live Demo
[**Click here to access the Dashboard**](https://your-app-name.streamlit.app)

## ✨ Features

### 🌩️ Ichimoku Kinko Hyo (一目均衡表) - Primary Indicator
- **Tenkan-sen** (天転換線) - Conversion Line
- **Kijun-sen** (基準線) - Base Line
- **Senkou Span A/B** (先行スパン) - Leading Cloud
- **Chikou Span** (遅行スパン) - Lagging Span

### 📊 13+ Technical Indicators
- Moving Averages (SMA, EMA)
- Bollinger Bands
- RSI, MACD, Stochastic
- Supertrend, ADX, ATR
- VWAP

### 🎯 Paper Trading
- Risk-free simulation in Indian Rupees (₹)
- Starting capital: ₹10,00,000 (10 Lakhs)
- Real-time P&L tracking
- Trade history

### 📈 NSE/BSE Stock Data
- Adani Enterprises (ADANIENT)
- Asian Paints (ASIANPAINT)
- Daily and 5-minute intraday data

## 🚀 Quick Start

### Local Development
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/trading-bot-dashboard.git
cd trading-bot-dashboard

# Install dependencies
pip install -r requirements.txt

# Run dashboard
streamlit run dashboard.py
```

### Deployment
Deployed on [Streamlit Community Cloud](https://streamlit.io/cloud) for free hosting.

## 📁 Project Structure
```
Trading_bot/
├── dashboard.py           # Main Streamlit app
├── requirements.txt       # Dependencies
├── config.yaml           # Trading configuration
├── stock_data/           # Local Excel data files
│   ├── Adani enterprise annual.xlsx
│   ├── Asian Paints Annual.xlsx
│   └── ...
└── src/
    ├── data/             # Data providers
    ├── indicators/       # Technical indicators
    ├── strategy/         # Trading strategies
    ├── execution/        # Order execution
    └── risk/             # Risk management
```

## 🛠️ Tech Stack
- **Frontend**: Streamlit + Plotly
- **Backend**: Python 3.11+
- **Data**: Alpha Vantage API, Local Excel
- **Indicators**: ta library, custom implementation

## 📜 License
MIT License - Free for commercial and personal use

## 🤝 Contributing
Pull requests welcome! Please open an issue first for major changes.

---
Made with ❤️ for Indian Traders

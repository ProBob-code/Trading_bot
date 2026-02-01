# GodBotTrade - Multi-Asset Trading Platform ⚡

A professional multi-asset trading platform supporting **Crypto (24/7)** and **Stocks** with real-time charts, auto-trading bots, news sentiment analysis, and paper trading simulation.

![Trading Platform](https://img.shields.io/badge/Platform-Web-blue) ![Python](https://img.shields.io/badge/Python-3.12+-green) ![Flask](https://img.shields.io/badge/Backend-Flask-red) ![JavaScript](https://img.shields.io/badge/Frontend-JavaScript-yellow)

---

## 🌟 Features

### 📊 Multi-Asset Trading
- **Crypto Markets**: BTC, ETH, BNB, SOL, XRP (24/7 trading via Binance)
- **Stock Markets**: AAPL, TSLA, GOOGL, MSFT, AMZN (US markets)
- **Indian Stocks**: TCS, INFOSYS, RELIANCE, and more

### 🤖 Auto-Trading Bots
- **Multiple Strategies**:
  - 🌩️ Ichimoku Cloud
  - 📊 Bollinger Bands
  - 📈 MACD + RSI (73-77% success rate)
  - 🤖 ML Forecast
  - ⚡ Combined (consensus voting)
- Start/Stop bots per symbol
- Hot-swap strategies on running bots
- Configurable stop-loss/take-profit

### � News Sentiment Analysis
- Live news from RSS feeds (CoinTelegraph, Yahoo Finance)
- Keyword-based sentiment scoring (BULLISH/BEARISH/NEUTRAL)
- Top gainers/losers from news data
- Symbol-specific sentiment analysis

### 📈 Real-Time Charts
- Candlestick charts via Lightweight Charts
- Multiple timeframes: 1m, 5m, 15m, 1H, 1D
- Live price updates via WebSocket
- 24h stats: High, Low, Volume

### 💰 Paper Trading
- $100,000 starting balance
- Full trade execution simulation
- P&L tracking with history
- Positions management (Open/Closed/History/Orders)

### 🎨 Dynamic Themes
- **Crypto Theme**: Dark futuristic with cyan/purple accents
- **Stocks Theme**: Classic parchment with deep green

---

## 🚀 Quick Start

### Prerequisites
- Python 3.12+ installed
- Internet connection for live data

### Installation

```powershell
# 1. Clone or navigate to the project
cd "C:\Users\bajacob\OneDrive - Tecnicas Reunidas, S.A\sandbox\project_2\Trading_bot"

# 2. Install dependencies (one-time setup)
pip install -r requirements.txt
```

### Running the Application

```powershell
# Start the server
python api_server.py
```

Then open your browser to: **http://localhost:5050**

### Stopping the Application

Press `Ctrl+C` in the terminal where the server is running, or:

```powershell
# Find and stop Python processes
Get-Process python | Stop-Process -Force
```

---

## 📁 Project Structure

```
Trading_bot/
├── api_server.py          # Flask + WebSocket backend server
├── requirements.txt       # Python dependencies
├── config.yaml           # Trading configuration
├── web/                  # JavaScript Frontend
│   ├── index.html        # Main trading dashboard
│   ├── app.js            # Frontend logic (1698 lines)
│   ├── styles.css        # Complete styling (2600+ lines)
│   ├── login.html        # Authentication page
│   ├── report.html       # Trading reports
│   └── live-settings.html # Live trading API configuration
├── src/
│   ├── data/             # Data providers (Binance, AlphaVantage)
│   ├── indicators/       # Technical indicators (Ichimoku, MACD, etc.)
│   ├── strategy/         # Trading strategies
│   ├── execution/        # Paper/Live trading execution
│   ├── sentiment/        # News sentiment analysis
│   └── risk/             # Risk management
├── data/                 # Local data files
├── models/               # ML models
└── logs/                 # Application logs
```

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Backend** | Python 3.12, Flask, Flask-SocketIO |
| **Frontend** | Vanilla JavaScript, HTML5, CSS3 |
| **Charts** | Lightweight Charts (TradingView) |
| **Real-time** | Socket.IO (WebSocket) |
| **Data** | Binance API, Alpha Vantage, RSS Feeds |
| **Sentiment** | Keyword-based NLP analysis |

---

## 📊 Trading Strategies

| Strategy | Success Rate | Best Timeframe | Description |
|----------|--------------|----------------|-------------|
| Ichimoku Cloud | 50-75% | 1H, 4H, 1D | Trend following with cloud support/resistance |
| Bollinger Bands | 55-65% | 15m, 1H | Mean reversion in ranging markets |
| MACD + RSI | 73-77% | 1H, 4H, 1D | Momentum + overbought/oversold |
| ML Forecast | 50-60% | 1H+ | Linear regression trend prediction |
| Combined | 65-80% | 1H, 4H | Consensus voting (highest quality) |

---

## 🎯 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/klines/{symbol}` | GET | Crypto candlestick data |
| `/api/stocks/klines/{symbol}` | GET | Stock candlestick data |
| `/api/trade` | POST | Execute a trade |
| `/api/bots/start` | POST | Start auto-trading bot |
| `/api/bots/stop-all` | POST | Stop all bots |
| `/api/news` | GET | News with sentiment |
| `/api/sentiment/{symbol}` | GET | Symbol-specific sentiment |
| `/api/positions` | GET | Current positions |
| `/api/panic-sell` | POST | Close all positions |

---

## 💡 Usage Tips

1. **Start Conservative**: Use 5% position size and 3+ confluence signals
2. **Crypto Timeframes**: 15m-1H for day trading, 4H for swing
3. **Stocks Timeframes**: 1H-1D (market hours only)
4. **Trending Markets**: Use Ichimoku or MACD+RSI
5. **Ranging Markets**: Use Bollinger Bands
6. **Unsure**: Use Combined strategy for consensus

---

## 🔧 Configuration

Edit `config.yaml` to customize:
- Initial capital
- Default stop-loss/take-profit
- API keys (for live trading)
- Strategy parameters

---

## 📜 License

MIT License - Free for commercial and personal use

---

## 🤝 Contributing

Pull requests welcome! Please open an issue first for major changes.

---

Made with ⚡ for Traders worldwide

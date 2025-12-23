# Trading Bot - Quick Start Guide

## Prerequisites

1. **Activate Virtual Environment**
   ```powershell
   cd "c:\Users\bajacob\OneDrive - Tecnicas Reunidas, S.A\sandbox\project_2\Trading_bot"
   .\TB\Scripts\Activate.ps1
   ```

---

## Step 1: Verify Installation

Run the demo to ensure all components work:

```powershell
python demo.py
```

**Expected Output**: Should load Adani Enterprise data, compute 41 indicators, generate a SELL signal.

---

## Step 2: Run Daily Strategy Backtest

Test the strategy on historical data (2020-2025):

```powershell
python run_bot.py --mode backtest
```

**Or run directly in Python:**
```python
from src.data.historical_loader import HistoricalLoader
from src.strategy.ta_strategy import TAStrategy

loader = HistoricalLoader()
df = loader.load_excel('stock_data/Adani enterprise annual.xlsx', header_row=29, date_column='Exchange Date')

strategy = TAStrategy(ma_fast=9, ma_slow=21, ma_signal=44, min_confluence=3)
trades = strategy.backtest(df, initial_capital=10000)
```

**Expected Result**: ~370% return on Adani Enterprise over 5 years.

---

## Step 3: Run Intraday Simulation

Simulate a trading day with auto buy/sell:

```powershell
# Run on latest date
python intraday_simulation.py

# Run on specific date
python intraday_simulation.py --date 2025-07-28

# Use 30-minute candles
python intraday_simulation.py --interval 30m

# Custom capital
python intraday_simulation.py --capital 50000
```

**Expected Output**: Trades with automatic SL/TP execution, EOD summary.

---

## Step 4: Paper Trading (Daily Signals)

Check current signals across stocks:

```powershell
python paper_trade.py
```

---

## Key Configuration Files

| File | Purpose |
|------|---------|
| `config.yaml` | Main configuration (symbols, strategy params, risk limits) |
| `demo.py` | Quick verification script |
| `intraday_simulation.py` | Intraday auto-trading simulation |
| `paper_trade.py` | Daily paper trading signals |

---

## Available Stock Data Files

| File | Interval |
|------|----------|
| `stock_data/Adani enterprise annual.xlsx` | Daily (2020-2025) |
| `stock_data/Adani enterprise 5 min.xlsx` | 5-minute intraday |
| `stock_data/Adani enterprise 30 min.xlsx` | 30-minute intraday |
| `stock_data/Asian Paints Annual.xlsx` | Daily (2020-2025) |
| `stock_data/Asian paints 5 min.xlsx` | 5-minute intraday |
| `stock_data/Asian paints 30 min.xlsx` | 30-minute intraday |

---

## Troubleshooting

### SSL Certificate Error
Use trusted hosts for pip:
```powershell
python -m pip install --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt
```

### Missing Module Error
Install dependencies:
```powershell
python -m pip install loguru openpyxl pandas numpy ta scikit-learn xgboost
```

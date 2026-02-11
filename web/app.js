/**
 * GodBotTrade - Multi-Asset Trading Platform
 * ==========================================
 * Trading bot for everything: Crypto, Stocks, Forex, Options
 */

// ============================================================
// STATE
// ============================================================

const state = {
    currentMarket: 'crypto',
    currentSymbol: 'BTCUSDT',
    currentInterval: '1m',
    currentStrategy: 'ichimoku',
    chart: null,
    socket: null,
    tradingEnabled: false,
    positions: {},
    strategies: {
        'ichimoku': 'Strategy 1',
        'bollinger': 'Strategy 2',
        'macd_rsi': 'Strategy 3',
        'ml_forecast': 'Strategy 4',
        'combined': 'Strategy 5 (Combined)'
    },
    isAutoTrading: false,
    lastPrice: 0,
    tradeCount: 0,
    buyCount: 0,
    sellCount: 0,
    // Position tracking for P&L
    // positions: {},  // { symbol: { qty, entryPrice, side } } - This is now part of the new state structure
    settings: {
        confluence: 3,
        positionSize: 10,
        checkInterval: 5,
        stopLoss: 5,
        takeProfit: 10
    },
    // Currency settings
    currency: 'USD',
    currencySymbol: '$',
    currencyRates: {
        USD: 1,
        EUR: 0.92,
        INR: 83.12,
        GBP: 0.79
    },
    paperBalance: 100000  // Editable paper trading balance
};

function getStrategyName(slug) {
    return state.strategies[slug] || slug;
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', () => {
    initChart();
    initSocket();
    initEventListeners();
    loadInitialData();
    updateUI();
    loadNews();  // Load news on startup

    // Auto-refresh data every 3 seconds
    setInterval(() => {
        loadPositions();
        loadBots();
    }, 3000);

    // Refresh news every 60 seconds
    setInterval(() => {
        loadNews();
    }, 60000);

    initClock();
    // Check server status periodically
    setInterval(checkServerStatus, 10000);
});

function initClock() {
    setInterval(() => {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const timeEl = document.getElementById('footerTime');
        if (timeEl) timeEl.textContent = timeStr;
    }, 1000);
}

async function checkServerStatus() {
    try {
        const res = await fetch('/api/stats');
        const statusEl = document.getElementById('serverStatusText');
        if (res.ok && statusEl) {
            statusEl.textContent = 'Online';
            statusEl.style.color = 'var(--green)';
        } else if (statusEl) {
            statusEl.textContent = 'Degraded';
            statusEl.style.color = 'var(--orange)';
        }
    } catch (e) {
        const statusEl = document.getElementById('serverStatusText');
        if (statusEl) {
            statusEl.textContent = 'Offline';
            statusEl.style.color = 'var(--red)';
        }
    }
}

// ============================================================
// CHART
// ============================================================

function initChart() {
    const container = document.getElementById('chart');

    state.chart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: container.clientHeight || 350,
        layout: {
            background: { type: 'solid', color: '#16161f' },
            textColor: '#8888aa',
        },
        grid: {
            vertLines: { color: '#2a2a3a' },
            horzLines: { color: '#2a2a3a' },
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
        },
        rightPriceScale: {
            borderColor: '#2a2a3a',
        },
        timeScale: {
            borderColor: '#2a2a3a',
            timeVisible: true,
            secondsVisible: false,
        },
    });

    state.candleSeries = state.chart.addCandlestickSeries({
        upColor: '#10b981',
        downColor: '#ef4444',
        borderUpColor: '#10b981',
        borderDownColor: '#ef4444',
        wickUpColor: '#10b981',
        wickDownColor: '#ef4444',
    });

    // Resize handler
    window.addEventListener('resize', () => {
        state.chart.applyOptions({
            width: container.clientWidth,
            height: container.clientHeight || 350,
        });
    });
}

async function loadChartData() {
    try {
        const endpoint = state.currentMarket === 'crypto'
            ? `/api/klines/${state.currentSymbol}?interval=${state.currentInterval}&limit=200`
            : `/api/stocks/klines/${state.currentSymbol}?interval=${state.currentInterval}&limit=200`;

        const response = await fetch(endpoint);
        const data = await response.json();

        if (data && data.length > 0) {
            const rate = state.currencyRates[state.currency] || 1;
            const convertedData = data.map(d => ({
                ...d,
                open: d.open * rate,
                high: d.high * rate,
                low: d.low * rate,
                close: d.close * rate
            }));
            state.candleSeries.setData(convertedData);
            state.chart.timeScale().fitContent();
        }
    } catch (error) {
        console.error('Error loading chart data:', error);
    }
}

// ============================================================
// WEBSOCKET - REAL-TIME DATA
// ============================================================

function initSocket() {
    state.socket = io('http://localhost:5050', {
        transports: ['websocket', 'polling']
    });

    state.socket.on('connect', () => {
        console.log('Connected to GodBotTrade server');
        updateConnectionStatus(true);
    });

    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        updateConnectionStatus(false);
    });

    state.socket.on('price_update', (data) => {
        updatePrice(data);
        updateAccount(data.account);
        updateChart(data);
    });

    state.socket.on('connected', (data) => {
        console.log('Server acknowledged connection:', data);
    });

    // Live auto-trading events
    state.socket.on('auto_trade_signal', (data) => {
        console.log('Signal:', data.signal, '@', data.price);
        addSignalToFeed(data);
    });

    state.socket.on('auto_trade_executed', (data) => {
        console.log('Trade executed:', data);
        addTradeToFeed(data);
        showTradeNotification(data);
        updateTradeCount();
    });

    state.socket.on('market_intel', (data) => {
        renderPulseGauge(data);
        renderAIInsights(data);
    });
}

function updateConnectionStatus(connected) {
    const statusDot = document.querySelector('.status-dot');
    const statusText = document.querySelector('.status-text');

    if (connected) {
        statusDot.classList.add('connected');
        statusText.textContent = 'Live';
    } else {
        statusDot.classList.remove('connected');
        statusText.textContent = 'Disconnected';
    }
}

function updatePrice(data) {
    const priceEl = document.getElementById('currentPrice');
    const changeEl = document.getElementById('priceChange');
    const high24hEl = document.getElementById('high24h');
    const low24hEl = document.getElementById('low24h');
    const volume24hEl = document.getElementById('volume24h');

    const price = parseFloat(data.price);
    const previousPrice = state.lastPrice;
    state.lastPrice = price;

    priceEl.textContent = formatPrice(price);

    // Flash effect
    if (price > previousPrice) {
        priceEl.style.color = '#10b981';
    } else if (price < previousPrice) {
        priceEl.style.color = '#ef4444';
    }
    setTimeout(() => {
        priceEl.style.color = '#ffffff';
    }, 200);

    // Change percentage
    const changePct = parseFloat(data.change_pct) || 0;
    changeEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
    changeEl.className = `price-change ${changePct >= 0 ? 'positive' : 'negative'}`;

    // Stats
    high24hEl.textContent = formatPrice(data.high_24h);
    low24hEl.textContent = formatPrice(data.low_24h);
    volume24hEl.textContent = formatVolume(data.volume_24h);

    // Update auto quantity badge
    updateAutoQuantityBadge(price);
}

// Update the Ready badge to show calculated auto quantity
function updateAutoQuantityBadge(currentPrice) {
    const badge = document.getElementById('botBadge');
    if (!badge || !currentPrice) return;

    // Get current settings
    const positionSizePct = parseFloat(document.getElementById('settingPositionSize')?.value) || 10;
    const maxQty = parseFloat(document.getElementById('maxQuantity')?.value) || 10;
    const balance = state.balance || 100000;

    // Calculate auto quantity: (balance × position_size%) / price
    const tradeValue = balance * (positionSizePct / 100);
    let autoQty = tradeValue / currentPrice;

    // Apply max quantity cap
    autoQty = Math.min(autoQty, maxQty);

    // Format and display
    const formattedQty = autoQty < 1 ? autoQty.toFixed(4) : autoQty.toFixed(2);
    badge.textContent = `Auto: ${formattedQty}`;
    badge.title = `Auto quantity: ${formattedQty} (${positionSizePct}% of $${balance.toLocaleString()} = $${tradeValue.toFixed(2)} ÷ $${currentPrice.toFixed(2)}, max: ${maxQty})`;
}

function updateChart(data) {
    if (!state.candleSeries || !data.price) return;

    const now = Math.floor(Date.now() / 1000);
    const rate = state.currencyRates[state.currency] || 1;
    const price = data.price * rate;

    state.candleSeries.update({
        time: now,
        open: price,
        high: price,
        low: price,
        close: price
    });
}

function updateAccount(account) {
    if (!account) return;

    const balanceEl = document.getElementById('accountBalance');
    const pnlEl = document.getElementById('accountPnl');

    balanceEl.textContent = formatCurrencyValue(account.total_value);

    const pnl = account.pnl;
    pnlEl.textContent = (pnl >= 0 ? '+' : '') + formatCurrencyValue(pnl);
    pnlEl.className = `account-value pnl ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
}

function updateTradeCount() {
    fetch('/api/auto-trade/status')
        .then(res => res.json())
        .then(data => {
            const countEl = document.getElementById('tradeCount');
            if (countEl) countEl.textContent = data.total_trades || 0;

            // Also update total trades in summary if it exists
            const totalTradesEl = document.getElementById('totalTrades');
            if (totalTradesEl) totalTradesEl.textContent = data.total_trades || 0;
        });
}

// ============================================================
// MARKET SWITCHING
// ============================================================

function switchMarket(market) {
    state.currentMarket = market;

    // Update tabs
    document.querySelectorAll('.market-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.market === market);
    });

    // Update symbol options
    const cryptoGroup = document.getElementById('cryptoSymbols');
    const stockGroup = document.getElementById('stockSymbols');

    if (market === 'crypto') {
        cryptoGroup.style.display = '';
        stockGroup.style.display = 'none';
        state.currentSymbol = 'BTCUSDT';
        document.getElementById('marketBadge').textContent = 'CRYPTO';
        document.getElementById('marketBadge').style.background = '#7c3aed';
    } else {
        cryptoGroup.style.display = 'none';
        stockGroup.style.display = '';
        state.currentSymbol = 'AAPL';
        document.getElementById('marketBadge').textContent = 'STOCKS';
        document.getElementById('marketBadge').style.background = '#10b981';
    }

    document.getElementById('symbolSelect').value = state.currentSymbol;

    // Tell server about market change
    state.socket.emit('change_market', {
        market: market,
        symbol: state.currentSymbol
    });

    loadChartData();
}

// ============================================================
// TRADING
// ============================================================

async function executeTrade(side) {
    const quantity = parseFloat(document.getElementById('tradeQuantity').value);

    if (!quantity || quantity <= 0) {
        alert('Please enter a valid quantity');
        return;
    }

    try {
        const response = await fetch('/api/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                side: side,
                quantity: quantity,
                market: state.currentMarket,
                user: document.getElementById('dropdownUsername')?.textContent || 'admin'
            })
        });

        const result = await response.json();

        if (result.success) {
            // Track position for P&L calculation
            if (side === 'buy') {
                state.positions[state.currentSymbol] = {
                    qty: quantity,
                    entryPrice: result.price,
                    side: 'BUY'
                };
            } else {
                // Selling - calculate realized P&L
                if (state.positions[state.currentSymbol]) {
                    const pos = state.positions[state.currentSymbol];
                    const pnl = (result.price - pos.entryPrice) * quantity;
                    const pnlPct = ((result.price / pos.entryPrice) - 1) * 100;
                    result.pnl = pnl;
                    result.pnlPct = pnlPct;
                    delete state.positions[state.currentSymbol];
                }
            }

            // Update trade count display
            state.tradeCount = result.total_trades || state.tradeCount + 1;
            if (side === 'buy') state.buyCount++;
            else state.sellCount++;

            document.getElementById('tradeCount').textContent = state.tradeCount;

            addTradeToFeed(result);
            console.log('Trade executed:', result);
        } else {
            alert('Trade failed: ' + result.error);
        }
    } catch (error) {
        console.error('Trade error:', error);
        alert('Trade failed: ' + error.message);
    }
}

function addTradeToFeed(trade) {
    const feed = document.getElementById('tradesFeed');
    const item = document.createElement('div');
    item.className = `trade-item ${trade.side.toLowerCase().includes('buy') ? 'buy' : 'sell'}`;

    // Show P&L % if available (on SELL), otherwise show price
    let displayValue;
    if (trade.pnlPct !== undefined) {
        const pnlClass = trade.pnlPct >= 0 ? 'positive' : 'negative';
        displayValue = `<span class="price pnl ${pnlClass}">${trade.pnlPct >= 0 ? '+' : ''}${trade.pnlPct.toFixed(2)}%</span>`;
    } else {
        displayValue = `<span class="price">${formatPrice(trade.price)}</span>`;
    }

    item.innerHTML = `
        <span class="side">${trade.side === 'BUY' || trade.side.includes('BUY') ? '🟢' : '🔴'} ${trade.side.split(' ')[0]}</span>
        <span class="symbol">${trade.symbol}</span>
        <div class="price-qty-container">
            ${displayValue}
            <span class="qty">×${trade.quantity.toFixed(4)}</span>
        </div>
        <span class="time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
    `;
    feed.insertBefore(item, feed.firstChild);

    while (feed.children.length > 30) {
        feed.removeChild(feed.lastChild);
    }
}

function addSignalToFeed(signal) {
    const feed = document.getElementById('tradesFeed');
    const item = document.createElement('div');

    const signalIcon = signal.signal === 'BUY' ? '🟢' :
        signal.signal === 'SELL' ? '🔴' : '⚪';
    const signalClass = signal.signal === 'BUY' ? 'buy' :
        signal.signal === 'SELL' ? 'sell' : 'hold';

    item.className = `trade-item ${signalClass}`;
    item.style.opacity = '0.7';

    item.innerHTML = `
        <span class="side">${signalIcon} ${signal.signal}</span>
        <span class="symbol">${signal.symbol}</span>
        <div class="price-qty-container">
            <span class="price">${formatPrice(signal.price)}</span>
            <span class="qty">Signal</span>
        </div>
        <span class="time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
    `;
    feed.insertBefore(item, feed.firstChild);

    while (feed.children.length > 30) {
        feed.removeChild(feed.lastChild);
    }
}

function showTradeNotification(trade) {
    const status = document.getElementById('autoTradeStatus');
    const originalText = status.textContent;

    status.textContent = `${trade.side} @ ${formatPrice(trade.price)}`;
    status.style.background = trade.side === 'BUY' ? '#10b981' : '#ef4444';
    status.style.color = '#fff';

    setTimeout(() => {
        status.textContent = originalText;
        status.style.background = '';
        status.style.color = '';
    }, 2000);
}

// ============================================================
// AUTO TRADING
// ============================================================

async function startBot() {
    const btn = document.getElementById('btnAutoTrade');
    const status = document.getElementById('autoTradeStatus');

    // Gather settings from MAIN UI inputs for WYSIWYG experience
    const uiSettings = {
        stopLoss: parseFloat(document.getElementById('stopLoss').value) || state.settings.stopLoss,
        takeProfit: parseFloat(document.getElementById('takeProfit').value) || state.settings.takeProfit,
        positionSize: parseFloat(document.getElementById('positionSize').value) || state.settings.positionSize,
        maxQuantity: parseFloat(document.getElementById('maxQuantity').value) || 1.0,
        confluence: state.settings.confluence || 3,
        checkInterval: state.settings.checkInterval || 5
    };

    try {
        console.log('Starting bot with settings:', uiSettings);

        const response = await fetch('/api/bots/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                interval: state.currentInterval,
                strategy: state.currentStrategy,
                market: state.currentMarket,
                settings: uiSettings,
                mode: state.tradingMode || 'paper'
            })
        });

        const result = await response.json();

        if (result.success) {
            showNotification(`🚀 Bot started for ${state.currentSymbol}`);
            status.textContent = 'Launching...';
            setTimeout(() => {
                status.textContent = 'Running';
                loadBots();
            }, 1000);
        } else {
            alert('Failed to start: ' + result.error);
        }
    } catch (error) {
        alert('Error: ' + error.message);
    }
}

async function showReportModal() {
    // Navigate to the professional report page
    window.location.href = '/report.html';
}

function showReport(report) {
    const msg = `📊 Trading Session Complete
━━━━━━━━━━━━━━━━━━━━━━
Total Trades: ${report.total_trades || 0}
  • Buy Orders: ${report.buy_trades || 0}
  • Sell Orders: ${report.sell_trades || 0}
━━━━━━━━━━━━━━━━━━━━━━
Total P&L: $${(report.total_pnl || 0).toFixed(2)}
ROI: ${(report.roi_percent || 0).toFixed(2)}%
Final Balance: $${(report.final_balance || 0).toLocaleString()}
Signals Generated: ${report.signals_generated || 0}`;
    alert(msg);
}

// ============================================================
// SETTINGS MODAL
// ============================================================

function openSettings() {
    document.getElementById('settingsModal').classList.add('open');

    // Load current settings
    document.getElementById('settingConfluence').value = state.settings.confluence;
    document.getElementById('confluenceValue').textContent = state.settings.confluence;

    document.getElementById('settingPositionSize').value = state.settings.positionSize;
    document.getElementById('positionSizeValue').textContent = state.settings.positionSize + '%';

    document.getElementById('settingInterval').value = state.settings.checkInterval;
    document.getElementById('intervalValue').textContent = state.settings.checkInterval + 's';

    document.getElementById('settingStopLoss').value = state.settings.stopLoss;
    document.getElementById('settingTakeProfit').value = state.settings.takeProfit;
}

function closeSettings() {
    document.getElementById('settingsModal').classList.remove('open');
}

function saveSettings() {
    state.settings = {
        confluence: parseInt(document.getElementById('settingConfluence').value),
        positionSize: parseInt(document.getElementById('settingPositionSize').value),
        checkInterval: parseInt(document.getElementById('settingInterval').value),
        stopLoss: parseFloat(document.getElementById('settingStopLoss').value),
        takeProfit: parseFloat(document.getElementById('settingTakeProfit').value)
    };

    console.log('Settings saved:', state.settings);
    closeSettings();

    // Notify user
    alert('✅ Settings saved successfully!');
}

// ============================================================
// EVENT LISTENERS
// ============================================================

function initEventListeners() {
    // Market tabs
    document.querySelectorAll('.market-tab').forEach(tab => {
        tab.addEventListener('click', () => switchMarket(tab.dataset.market));
    });

    // If market switching logic needs to be initialized
    applyMarketTheme(state.currentMarket);

    // Symbol selector
    document.getElementById('symbolSelect').addEventListener('change', (e) => {
        const symbol = e.target.value;
        state.currentSymbol = symbol;
        const market = state.currentMarket;

        // Update badge
        const marketBadge = document.getElementById('marketBadge');
        if (marketBadge) marketBadge.textContent = market.toUpperCase();

        // Sync both market and symbol to backend
        if (state.socket) {
            state.socket.emit('change_market', {
                market: market,
                symbol: symbol
            });
        }

        loadChartData();
    });

    // Strategy selector
    document.getElementById('strategySelect').addEventListener('change', async (e) => {
        const oldStrategy = state.currentStrategy;
        state.currentStrategy = e.target.value;
        const strategyName = getStrategyName(state.currentStrategy);
        document.getElementById('activeStrategy').textContent = strategyName;

        // Check if there's a bot running for this symbol/market to hot-swap strategy
        const botId = `${state.currentMarket}_${state.currentSymbol}`.toLowerCase();
        try {
            const response = await fetch(`/api/bots/${botId}/strategy`, {
                method: 'PUT',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ strategy: state.currentStrategy })
            });
            const result = await response.json();
            if (result.success) {
                showNotification(`🔄 Strategy updated: ${getStrategyName(oldStrategy)} → ${getStrategyName(state.currentStrategy)}`);
            }
        } catch (err) {
            // No running bot for this symbol, that's fine
        }
    });

    // Interval buttons
    document.querySelectorAll('.interval-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            document.querySelectorAll('.interval-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentInterval = btn.dataset.interval;
            loadChartData();
        });
    });

    // Trade buttons
    document.getElementById('btnBuy').addEventListener('click', () => executeTrade('buy'));
    document.getElementById('btnSell').addEventListener('click', () => executeTrade('sell'));

    // Auto trading - START button starts a bot for the current symbol
    document.getElementById('btnAutoTrade').addEventListener('click', () => {
        startBot();
    });

    // Dedicated STOP button for the CURRENT symbol
    document.getElementById('btnStopBot').addEventListener('click', () => {
        const botId = `${state.currentMarket}_${state.currentSymbol}`.toLowerCase();
        stopBot(botId);
    });

    // STOP ALL button - stops all running bots
    document.getElementById('btnStopAll').addEventListener('click', () => {
        fetch('/api/bots/stop-all', { method: 'POST' })
            .then(() => {
                showNotification('🛑 All bots stopping...');
                loadBots();
                loadPositions(); // Refresh positions immediately to see closures
            });
    });

    document.getElementById('btnPanicSell').addEventListener('click', async () => {
        if (confirm('🚨 EMERGENCY: Close ALL open positions immediately?')) {
            try {
                const response = await fetch('/api/panic-sell', { method: 'POST' });
                const result = await response.json();
                if (result.success) {
                    showNotification(`✅ Panic Sell Executed: ${result.message}`, 'warning');
                    loadPositions();
                }
            } catch (err) {
                alert('Panic sell failed: ' + err.message);
            }
        }
    });

    document.getElementById('btnMarketAnalysis').addEventListener('click', async () => {
        showNotification('🔍 Analyzing market sentiment...', 'info');

        try {
            // Fetch news data only (calculate sentiment locally)
            const newsRes = await fetch(`/api/news?market=${state.currentMarket}&limit=10`);
            const newsData = await newsRes.json();
            const news = newsData.news || [];

            // Calculate sentiment from news items
            let bullish = 0, bearish = 0, neutral = 0;
            news.forEach(article => {
                if (article.sentiment === 'bullish') bullish++;
                else if (article.sentiment === 'bearish') bearish++;
                else neutral++;
            });

            const total = bullish + bearish + neutral || 1;
            const score = (bullish - bearish) / total;

            const sentiment = {
                score: score,
                bullish_count: bullish,
                bearish_count: bearish,
                neutral_count: neutral
            };

            // Show persistent modal with analysis
            showNewsAnalysisModal(sentiment, news);

        } catch (error) {
            console.error('News analysis error:', error);
            // Show modal with empty data instead of failing
            showNewsAnalysisModal({ score: 0, bullish_count: 0, bearish_count: 0, neutral_count: 0 }, []);
        }
    });

    document.getElementById('btnBacktest').addEventListener('click', () => {
        alert('📈 Strategy Backtester module coming soon! Use the Report section to see historical bot performance.');
    });

    document.getElementById('btnReport').addEventListener('click', showReportModal);

    // Settings modal
    document.getElementById('btnSettings').addEventListener('click', openSettings);
    document.getElementById('modalClose').addEventListener('click', closeSettings);
    document.getElementById('btnSaveSettings').addEventListener('click', saveSettings);

    // Tips modal
    document.getElementById('btnTips').addEventListener('click', () => {
        document.getElementById('tipsModal').classList.add('open');
    });
    document.getElementById('tipsModalClose').addEventListener('click', () => {
        document.getElementById('tipsModal').classList.remove('open');
    });

    // Update auto quantity badge when settings change
    document.getElementById('settingPositionSize')?.addEventListener('input', () => {
        if (state.lastPrice) updateAutoQuantityBadge(state.lastPrice);
    });
    document.getElementById('maxQuantity')?.addEventListener('input', () => {
        if (state.lastPrice) updateAutoQuantityBadge(state.lastPrice);
    });

    // Settings sliders
    document.getElementById('settingConfluence').addEventListener('input', (e) => {
        document.getElementById('confluenceValue').textContent = e.target.value;
    });
    document.getElementById('settingPositionSize').addEventListener('input', (e) => {
        document.getElementById('positionSizeValue').textContent = e.target.value + '%';
    });
    document.getElementById('settingInterval').addEventListener('input', (e) => {
        document.getElementById('intervalValue').textContent = e.target.value + 's';
    });

    // Clear feed
    document.getElementById('btnClearFeed').addEventListener('click', () => {
        document.getElementById('tradesFeed').innerHTML = '';
    });
}

// ============================================================
// UI UPDATES
// ============================================================

function updateUI() {
    document.getElementById('activeStrategy').textContent = 'Ichimoku';
}

async function loadInitialData() {
    await loadChartData();
}

// ============================================================
// UTILITIES
// ============================================================

function formatPrice(price) {
    return formatCurrencyValue(price);
}

function formatVolume(volume) {
    if (!volume) return '0';

    const num = parseFloat(volume);
    if (num >= 1000000000) {
        return (num / 1000000000).toFixed(2) + 'B';
    } else if (num >= 1000000) {
        return (num / 1000000).toFixed(2) + 'M';
    } else if (num >= 1000) {
        return (num / 1000).toFixed(2) + 'K';
    }
    return num.toFixed(2);
}

// ============================================================
// MULTI-BOT MANAGEMENT
// ============================================================

// Load and display all bots
async function loadBots() {
    try {
        const response = await fetch('/api/bots');
        const data = await response.json();

        if (data.success) {
            renderBots(data.bots);
            document.getElementById('botCount').textContent = data.running_count;

            // Update Control Buttons (START/STOP) for current symbol
            const currentBotId = `${state.currentMarket}_${state.currentSymbol}`.toLowerCase();
            const isBotRunning = data.bots.some(b => b.bot_id === currentBotId);

            const startBtn = document.getElementById('btnAutoTrade');
            const stopBtn = document.getElementById('btnStopBot');
            const status = document.getElementById('autoTradeStatus');

            if (isBotRunning) {
                if (startBtn) startBtn.style.display = 'none';
                if (stopBtn) stopBtn.style.display = 'inline-flex';
                status.textContent = 'Running';
            } else {
                if (startBtn) startBtn.style.display = 'inline-flex';
                if (stopBtn) stopBtn.style.display = 'none';
                status.textContent = data.running_count > 0 ? 'Watching' : 'Stopped';
            }

            // Show/hide STOP ALL button based on running bots
            const stopAllBtn = document.getElementById('btnStopAll');
            if (data.running_count > 0) {
                stopAllBtn.classList.add('active');
            } else {
                stopAllBtn.classList.remove('active');
            }
        }
    } catch (error) {
        console.error('Error loading bots:', error);
    }
}

// Render bots list
function renderBots(bots) {
    const container = document.getElementById('botsList');

    if (!bots || bots.length === 0) {
        container.innerHTML = '<div class="no-bots">No active bots</div>';
        return;
    }

    container.innerHTML = bots.map(bot => {
        const pnl = bot.stats.total_pnl || 0;
        const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const pnlSign = pnl >= 0 ? '+' : '';

        return `
            <div class="bot-card ${bot.mode}">
                <div class="bot-info">
                    <span class="bot-symbol">${bot.symbol}</span>
                    <span class="bot-strategy">${getStrategyName(bot.strategy)} • ${bot.market}</span>
                </div>
                <div class="bot-stats-row">
                     <span class="bot-pnl ${pnlClass}">${pnlSign}${formatCurrencyValue(pnl)}</span>
                     <span class="bot-trade-count">${bot.stats.total_trades || 0} trades</span>
                </div>
                <button class="btn-stop-bot" onclick="stopBot('${bot.bot_id}')">STOP</button>
            </div>
        `;
    }).join('');
}

// Stop a specific bot
async function stopBot(botId) {
    try {
        const response = await fetch(`/api/bots/${botId}/stop`, { method: 'POST' });
        const result = await response.json();

        if (result.success) {
            showNotification(`Bot ${botId} stopped`);
            loadBots();
        }
    } catch (error) {
        console.error('Error stopping bot:', error);
    }
}

// Show notification
function showNotification(message) {
    const status = document.getElementById('autoTradeStatus');
    const originalText = status.textContent;

    // Create toast alert if possible or use status bar
    status.textContent = message;
    status.style.background = '#7c3aed';
    status.style.color = '#fff';

    console.log('📢', message);

    setTimeout(() => {
        status.textContent = 'Running';
        status.style.background = '';
        status.style.color = '';
    }, 3000);
}

// Show News Analysis Modal
function showNewsAnalysisModal(sentiment, news) {
    // Remove existing modal if any
    const existingModal = document.getElementById('newsAnalysisModal');
    if (existingModal) existingModal.remove();

    // Determine overall sentiment
    const score = sentiment.score || 0;
    const overallSentiment = score > 0.1 ? 'BULLISH' : (score < -0.1 ? 'BEARISH' : 'NEUTRAL');
    const sentimentColor = score > 0.1 ? '#10b981' : (score < -0.1 ? '#ef4444' : '#f59e0b');
    const sentimentIcon = score > 0.1 ? '📈' : (score < -0.1 ? '📉' : '➡️');

    // Create modal HTML
    const modalHTML = `
        <div id="newsAnalysisModal" class="news-modal-overlay">
            <div class="news-modal">
                <div class="news-modal-header">
                    <h2>🔍 Market Analysis: ${state.currentSymbol}</h2>
                    <button class="news-modal-close" onclick="document.getElementById('newsAnalysisModal').remove()">&times;</button>
                </div>
                <div class="news-modal-body">
                    <div class="sentiment-summary">
                        <div class="sentiment-score" style="background: ${sentimentColor}">
                            <span class="sentiment-icon">${sentimentIcon}</span>
                            <span class="sentiment-label">${overallSentiment}</span>
                            <span class="sentiment-value">${(score * 100).toFixed(1)}%</span>
                        </div>
                        <div class="sentiment-stats">
                            <div class="stat"><span class="label">Bullish News:</span><span class="value green">${sentiment.bullish_count || 0}</span></div>
                            <div class="stat"><span class="label">Bearish News:</span><span class="value red">${sentiment.bearish_count || 0}</span></div>
                            <div class="stat"><span class="label">Neutral News:</span><span class="value">${sentiment.neutral_count || 0}</span></div>
                        </div>
                    </div>
                    <div class="news-headlines">
                        <h3>📰 Latest Headlines</h3>
                        ${news.length > 0 ? news.map(article => `
                            <div class="news-item ${article.sentiment || 'neutral'}">
                                <span class="news-sentiment-badge">${article.sentiment?.toUpperCase() || 'NEWS'}</span>
                                <span class="news-title">${article.title}</span>
                            </div>
                        `).join('') : '<p class="no-news">No recent news available</p>'}
                    </div>
                </div>
                <div class="news-modal-footer">
                    <button class="btn btn-secondary" onclick="document.getElementById('newsAnalysisModal').remove()">Close</button>
                </div>
            </div>
        </div>
    `;

    // Insert modal into body
    document.body.insertAdjacentHTML('beforeend', modalHTML);
}

// Trading mode toggle
function setTradingMode(mode) {
    state.tradingMode = mode;

    // Update buttons
    document.querySelectorAll('.mode-btn').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.mode === mode);
    });

    // Update badge
    const badge = document.getElementById('tradingModeBadge');
    badge.textContent = mode === 'paper' ? 'PAPER MODE' : '⚠️ LIVE MODE';
    badge.className = `mode-badge ${mode}`;

    // Show warning for live mode
    if (mode === 'live') {
        alert('⚠️ LIVE MODE ENABLED\n\nThis will use REAL MONEY.\nMake sure your API keys are configured correctly.');
    }
}

// Initialize bot management
function initBotManagement() {
    // Mode toggle buttons
    document.getElementById('btnPaperMode')?.addEventListener('click', () => setTradingMode('paper'));
    document.getElementById('btnLiveMode')?.addEventListener('click', () => setTradingMode('live'));

    // WebSocket events for bots
    if (state.socket) {
        state.socket.on('bot_started', (data) => {
            showNotification(`🚀 ${data.message}`);
            loadBots();
        });

        state.socket.on('bot_stopped', (data) => {
            showNotification(`🛑 ${data.message}`);
            loadBots();
        });

        state.socket.on('strategy_changed', (data) => {
            showNotification(`🔄 Strategy changed: ${getStrategyName(data.old_strategy)} → ${getStrategyName(data.new_strategy)}`);
        });

        // Live News Update
        state.socket.on('news_update', (news) => {
            if (news && news.length > 0) {
                updateNewsFeed(news);
                showNotification(`📰 New market update received`, 'info');
            }
        });
    }

    // Load bots on init
    loadBots();

    // Refresh bots every 5 seconds
    setInterval(loadBots, 5000);
}

// Add to DOMContentLoaded
document.addEventListener('DOMContentLoaded', () => {
    setTimeout(initBotManagement, 1000);
    initPositionsPanel();
    initMarketDepth();
});

// ============================================================
// POSITIONS PANEL MANAGEMENT
// ============================================================

// Position data storage
const positionsData = {
    open: [],
    closed: [],
    history: [],
    orders: []
};

function initPositionsPanel() {
    // Tab switching
    document.querySelectorAll('.pos-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            document.querySelectorAll('.pos-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.pos-content').forEach(c => c.classList.remove('active'));

            tab.classList.add('active');
            const tabId = tab.dataset.tab;
            document.getElementById(`tab-${tabId}`).classList.add('active');

            // Load specific data if needed
            if (tabId === 'reports') {
                loadReports();
            }
        });
    });

    // Export CSV button
    document.getElementById('btnExportCSV')?.addEventListener('click', exportToCSV);

    // Load initial positions
    loadPositions();

    // Refresh every 10 seconds
    setInterval(() => {
        loadPositions();
        const activeTab = document.querySelector('.pos-tab.active');
        if (activeTab && activeTab.dataset.tab === 'reports') {
            loadReports();
        }
    }, 10000);
}

async function loadPositions() {
    try {
        const response = await fetch('/api/positions');
        if (!response.ok) return;

        const data = await response.json();
        if (data.success) {
            positionsData.open = data.open_positions || [];
            positionsData.closed = data.closed_positions || [];
            positionsData.history = data.trade_history || [];
            positionsData.orders = data.pending_orders || [];

            renderOpenPositions();
            renderClosedPositions();
            renderTradeHistory();
            renderPendingOrders();
        }
    } catch (error) {
        // API not yet implemented, use local tracking
        renderFromLocalState();
    }
}

function renderFromLocalState() {
    // Render from local state.positions
    const openBody = document.getElementById('openPositionsBody');
    if (!openBody) return;

    if (Object.keys(state.positions).length === 0) {
        openBody.innerHTML = '<tr class="empty-row"><td colspan="8">No open positions</td></tr>';
    } else {
        openBody.innerHTML = Object.entries(state.positions).map(([symbol, pos]) => {
            const currentPrice = state.currentPrice || pos.entryPrice;
            const netPnl = (currentPrice - pos.entryPrice) * pos.qty;
            const netPnlPct = ((currentPrice / pos.entryPrice) - 1) * 100;
            const pnlClass = netPnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const sideClass = pos.side === 'BUY' ? 'side-long' : 'side-short';

            return `
                <tr>
                    <td><strong>${symbol}</strong></td>
                    <td class="${sideClass}">${pos.side === 'BUY' ? 'LONG' : 'SHORT'}</td>
                    <td>${pos.qty.toFixed(4)}</td>
                    <td>$${pos.entryPrice.toLocaleString()}</td>
                    <td>$${currentPrice.toLocaleString()}</td>
                    <td class="${pnlClass}">$${netPnl.toFixed(2)} (${netPnlPct >= 0 ? '+' : ''}${netPnlPct.toFixed(2)}%)</td>
                    <td>-</td>
                    <td><button class="btn-close-pos" onclick="closePosition('${symbol}')">Close</button></td>
                </tr>
            `;
        }).join('');
    }
}

function renderOpenPositions() {
    const openBody = document.getElementById('openPositionsBody');
    if (!openBody) return;

    if (positionsData.open.length === 0) {
        openBody.innerHTML = '<tr class="empty-row"><td colspan="8">No open positions</td></tr>';
    } else {
        openBody.innerHTML = positionsData.open.map(pos => {
            const currentPrice = pos.current_price || pos.avg_price;
            const netPnl = pos.net_pnl;
            const netPnlPct = pos.avg_price > 0 ? ((currentPrice / pos.avg_price) - 1) * 100 : 0;
            const pnlClass = netPnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const sideClass = pos.side === 'BUY' || pos.side === 'LONG' ? 'side-long' : 'side-short';

            return `
                <tr>
                    <td><strong>${pos.symbol}</strong></td>
                    <td class="${sideClass}">${pos.side}</td>
                    <td>${pos.qty.toFixed(4)}</td>
                    <td>$${pos.avg_price.toLocaleString()}</td>
                    <td>$${currentPrice.toLocaleString()}</td>
                    <td class="${pnlClass}">$${netPnl.toFixed(2)} (${netPnlPct >= 0 ? '+' : ''}${netPnlPct.toFixed(2)}%)</td>
                    <td>${pos.open_interest || 0}</td>
                    <td><button class="btn-close-pos" onclick="closePosition('${pos.symbol}')">Close</button></td>
                </tr>
            `;
        }).join('');
    }
}

function renderClosedPositions() {
    const body = document.getElementById('closedPositionsBody');
    if (!body) return;

    if (positionsData.closed.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="7">No closed positions</td></tr>';
    } else {
        body.innerHTML = positionsData.closed.map(pos => {
            const pnl = pos.pnl || (pos.exit - pos.entry) * pos.qty;
            const pnlPct = ((pos.exit / pos.entry) - 1) * 100;
            const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const sideClass = pos.side === 'BUY' ? 'side-long' : 'side-short';

            return `
                <tr class="closed-trade-row">
                    <td>
                        <div class="trade-symbol-meta">
                            <strong>${pos.symbol}</strong>, 
                            <span class="${sideClass}">${pos.side === 'BUY' ? 'buy' : 'sell'} ${pos.qty}</span>
                        </div>
                    </td>
                    <td colspan="3">
                        <div class="trade-price-flow">
                            $${parseFloat(pos.entry).toLocaleString()} &rarr; $${parseFloat(pos.exit).toLocaleString()}
                        </div>
                    </td>
                    <td class="${pnlClass}">
                        <div class="trade-pnl-cell">
                            <strong>${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(2)}</strong>
                            <small>(${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</small>
                        </div>
                    </td>
                    <td colspan="2">
                        <div class="trade-time-cell">
                            ${new Date(pos.closed_at).toLocaleString()}
                        </div>
                    </td>
                </tr>
            `;
        }).join('');
    }
}

function renderAuditJournal() {
    const body = document.getElementById('auditJournalBody');
    if (!body) return;

    if (!positionsData.journal || positionsData.journal.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="6">No recent audit logs</td></tr>';
    } else {
        body.innerHTML = positionsData.journal.slice(-30).reverse().map(log => {
            const sideClass = log.side === 'BUY' ? 'side-long' : 'side-short';
            return `
                <tr>
                    <td>${new Date(log.time).toLocaleTimeString()}</td>
                    <td><strong>${log.symbol}</strong></td>
                    <td class="${sideClass}">${log.side}</td>
                    <td>$${parseFloat(log.price).toLocaleString()}</td>
                    <td>${log.qty}</td>
                    <td class="strategy-reason-cell">
                        <span class="strategy-badge">${getStrategyName(log.strategy)}</span>
                        <div class="reason-text">${Array.isArray(log.reasons) ? log.reasons.join(', ') : log.reasons}</div>
                    </td>
                </tr>
            `;
        }).join('');
    }
}

function renderTradeHistory() {
    const body = document.getElementById('tradeHistoryBody');
    if (!body) return;

    // Use auto_trade_stats trades_log if available
    if (positionsData.history.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="8">No trade history</td></tr>';
    } else {
        body.innerHTML = positionsData.history.slice(-20).reverse().map(trade => `
            <tr>
                <td>${new Date(trade.time).toLocaleTimeString()}</td>
                <td><strong>${trade.symbol}</strong></td>
                <td>MARKET</td>
                <td class="${trade.side === 'BUY' ? 'side-long' : 'side-short'}">${trade.side}</td>
                <td>${trade.quantity}</td>
                <td>$${trade.price.toLocaleString()}</td>
                <td>$${(trade.quantity * trade.price).toLocaleString()}</td>
                <td>✓ Filled</td>
            </tr>
        `).join('');
    }

    // Update Trade Count in UI
    const totalTrades = positionsData.history.length;
    document.querySelectorAll('.trade-count-badge').forEach(el => {
        el.textContent = totalTrades;
    });
}

function renderPendingOrders() {
    const body = document.getElementById('pendingOrdersBody');
    if (!body) return;

    if (positionsData.orders.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="9">No pending orders</td></tr>';
    }
}

async function loadReports() {
    try {
        const response = await fetch('/api/reports');
        if (!response.ok) return;

        const data = await response.json();
        renderReports(data);
    } catch (error) {
        console.error('Error loading reports:', error);
    }
}

function renderReports(reports) {
    const body = document.getElementById('dailyReportsBody');
    if (!body) return;

    if (!reports || reports.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="7">No reports generated yet</td></tr>';
    } else {
        body.innerHTML = reports.map(report => `
            <tr>
                <td>${report.date}</td>
                <td><strong>${report.user}</strong></td>
                <td>${report.total_trades}</td>
                <td>${report.win_loss}</td>
                <td class="${parseFloat(report.win_rate) >= 50 ? 'pnl-positive' : 'pnl-negative'}">${report.win_rate}</td>
                <td class="${report.total_pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}">$${report.total_pnl.toLocaleString()}</td>
                <td class="${report.avg_profit >= 0 ? 'pnl-positive' : 'pnl-negative'}">$${report.avg_profit.toLocaleString()}</td>
            </tr>
        `).join('');
    }
}

async function closePosition(symbol) {
    if (confirm(`Close position for ${symbol}?`)) {
        // Execute opposite trade
        const pos = state.positions[symbol];
        if (pos) {
            const side = pos.side === 'BUY' ? 'sell' : 'buy';
            await executeTrade(side);
        }
    }
}

// ============================================================
// MARKET DEPTH
// ============================================================

function initMarketDepth() {
    // Show depth modal on Buy/Sell hover (optional)
    document.getElementById('btnBuy')?.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        openDepthModal();
    });
    document.getElementById('btnSell')?.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        openDepthModal();
    });
}

function openDepthModal() {
    document.getElementById('depthModal')?.classList.add('open');
    document.getElementById('depthSymbol').textContent = state.currentSymbol;
    loadMarketDepth();
}

function closeDepthModal() {
    document.getElementById('depthModal')?.classList.remove('open');
}

async function loadMarketDepth() {
    try {
        const response = await fetch(`/api/crypto/depth/${state.currentSymbol}`);
        const data = await response.json();

        if (data.bids && data.asks) {
            renderDepth(data);
        }
    } catch (error) {
        // Use placeholder data if API not available
        console.log('Market depth: Using placeholder data');
    }
}

function renderDepth(data) {
    const bidsContainer = document.getElementById('depthBids');
    const asksContainer = document.getElementById('depthAsks');

    if (data.bids) {
        bidsContainer.innerHTML = data.bids.slice(0, 5).map(bid => `
            <div class="depth-row bid-row">
                <span class="depth-price">$${parseFloat(bid[0]).toLocaleString()}</span>
                <span class="depth-volume">${parseFloat(bid[1]).toFixed(4)}</span>
            </div>
        `).join('');
    }

    if (data.asks) {
        asksContainer.innerHTML = data.asks.slice(0, 5).map(ask => `
            <div class="depth-row ask-row">
                <span class="depth-price">$${parseFloat(ask[0]).toLocaleString()}</span>
                <span class="depth-volume">${parseFloat(ask[1]).toFixed(4)}</span>
            </div>
        `).join('');
    }
}

// ============================================================
// CSV EXPORT
// ============================================================

function exportToCSV() {
    const trades = positionsData.history;
    if (trades.length === 0) {
        alert('No trades to export');
        return;
    }

    const headers = ['Time', 'Symbol', 'Side', 'Quantity', 'Price', 'Value'];
    const rows = trades.map(t => [
        t.time,
        t.symbol,
        t.side,
        t.quantity,
        t.price,
        (t.quantity * t.price).toFixed(2)
    ]);

    const csv = [headers, ...rows].map(row => row.join(',')).join('\n');
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);

    const a = document.createElement('a');
    a.href = url;
    a.download = `godbottrade_history_${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
}

// ============================================================
// PORTFOLIO ANALYTICS
// ============================================================

function calculateAnalytics() {
    const trades = positionsData.history;
    if (trades.length === 0) return {};

    const pnls = [];
    let wins = 0;
    let losses = 0;

    // Calculate win rate and returns
    for (let i = 1; i < trades.length; i += 2) {
        const entry = trades[i - 1];
        const exit = trades[i];
        if (entry && exit) {
            const pnl = (exit.price - entry.price) * entry.quantity;
            pnls.push(pnl);
            if (pnl > 0) wins++;
            else losses++;
        }
    }

    const totalPnL = pnls.reduce((a, b) => a + b, 0);
    const avgReturn = pnls.length > 0 ? totalPnL / pnls.length : 0;
    const winRate = (wins / (wins + losses)) * 100 || 0;

    // Sharpe ratio (simplified)
    const mean = avgReturn;
    const variance = pnls.reduce((sum, pnl) => sum + Math.pow(pnl - mean, 2), 0) / pnls.length;
    const stdDev = Math.sqrt(variance);
    const sharpe = stdDev > 0 ? mean / stdDev : 0;

    return {
        totalPnL,
        winRate,
        sharpe,
        totalTrades: trades.length
    };
}

// ============================================================
// NEWS FEED
// ============================================================

async function loadNews() {
    try {
        // Get current market type from state
        const market = state.currentMarket || 'crypto';

        // Fetch news and movers in parallel
        const [newsResponse, moversResponse, sentimentResponse] = await Promise.all([
            fetch(`/api/news?limit=10&market=${market}`),
            fetch('/api/news/movers'),
            fetch(`/api/sentiment/${state.currentSymbol}`)
        ]);

        const newsData = await newsResponse.json();
        const moversData = await moversResponse.json();
        const sentimentData = await sentimentResponse.json();

        // Update news feed
        updateNewsFeed(newsData.news || []);

        // Update sentiment badge
        updateSentimentBadge(sentimentData);

        // Update top movers
        updateTopMovers(moversData);

    } catch (error) {
        console.error('Error loading news:', error);
        const newsFeed = document.getElementById('newsFeed');
        if (newsFeed) {
            newsFeed.innerHTML = '<div class="news-loading">Unable to load news</div>';
        }
    }
}

function updateNewsFeed(news) {
    const container = document.getElementById('newsFeed');
    if (!container) return;

    if (!news || news.length === 0) {
        container.innerHTML = '<div class="news-loading">No news available</div>';
        return;
    }

    container.innerHTML = news.map(item => {
        const sentimentClass = item.sentiment_label === 'BULLISH' ? 'bullish' :
            item.sentiment_label === 'BEARISH' ? 'bearish' : '';

        // Clean up title
        let title = item.title || 'News headline';
        title = title.replace(/-/g, ' ').replace(/_/g, ' ');
        if (title.length > 80) title = title.substring(0, 77) + '...';

        return `
            <div class="news-item ${sentimentClass}" onclick="window.open('${item.url}', '_blank')">
                <span class="news-title">${title}</span>
                <div class="news-meta">
                    <span class="news-source">${item.source || 'News'}</span>
                    <span class="news-sentiment">${item.sentiment_label || 'NEUTRAL'}</span>
                </div>
            </div>
        `;
    }).join('');
}

function updateSentimentBadge(data) {
    const badge = document.getElementById('overallSentiment');
    if (!badge) return;

    const dot = badge.querySelector('.sentiment-dot');
    const label = badge.querySelector('.sentiment-label');

    if (dot && label) {
        // Remove old classes
        dot.classList.remove('bullish', 'bearish', 'neutral');
        badge.classList.remove('bullish', 'bearish');

        // Add new class based on sentiment
        const sentimentLabel = data.label || 'NEUTRAL';
        if (sentimentLabel === 'BULLISH') {
            dot.classList.add('bullish');
            badge.classList.add('bullish');
        } else if (sentimentLabel === 'BEARISH') {
            dot.classList.add('bearish');
            badge.classList.add('bearish');
        } else {
            dot.classList.add('neutral');
        }

        label.textContent = sentimentLabel;
    }
}

// ============================================================
// MARKET & THEME SWITCHING
// ============================================================

function switchMarket(market) {
    if (state.currentMarket === market) return;

    state.currentMarket = market;
    console.log(`Switching to ${market} market...`);

    // Update active tab UI
    document.querySelectorAll('.market-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.market === market);
    });

    // Update symbol select options (optgroups)
    const cryptoGroup = document.getElementById('cryptoSymbols');
    const stockGroup = document.getElementById('stockSymbols');
    const symbolSelect = document.getElementById('symbolSelect');

    if (market === 'crypto') {
        cryptoGroup.style.display = 'block';
        stockGroup.style.display = 'none';
        state.currentSymbol = 'BTCUSDT';
    } else {
        cryptoGroup.style.display = 'none';
        stockGroup.style.display = 'block';
        state.currentSymbol = 'AAPL';
    }

    symbolSelect.value = state.currentSymbol;

    // Notify server of market change
    if (state.socket) {
        state.socket.emit('change_market', {
            market: state.currentMarket,
            symbol: state.currentSymbol
        });
    }

    // Update news heading
    const newsHeading = document.getElementById('newsHeading');
    if (newsHeading) {
        newsHeading.textContent = market === 'crypto' ? '🪙 Crypto Market News' : '📉 Stock Market News';
    }

    // Apply visual theme
    applyMarketTheme(market);

    // Reload data
    loadChartData();
    loadNews();
    showNotification(`Market switched to ${market.toUpperCase()}`, 'info');
}

function applyMarketTheme(market) {
    const body = document.body;
    body.classList.remove('theme-crypto', 'theme-stocks');
    body.classList.add(`theme-${market}`);

    // Update chart colors based on theme if needed
    if (state.chart) {
        const themeColors = market === 'crypto'
            ? { bg: '#0a0a0f', grid: '#1a1a2a', text: '#8888aa' }
            : { bg: '#f4f4e4', grid: '#d0d0c0', text: '#4a4a3a' };

        state.chart.applyOptions({
            layout: {
                background: { type: 'solid', color: themeColors.bg },
                textColor: themeColors.text,
            },
            grid: {
                vertLines: { color: themeColors.grid },
                horzLines: { color: themeColors.grid },
            }
        });
    }
}

function updateTopMovers(data) {
    const gainersEl = document.getElementById('topGainers');
    const losersEl = document.getElementById('topLosers');

    if (gainersEl) {
        const gainers = data.top_gainers || [];
        gainersEl.textContent = gainers.length > 0 ? gainers.slice(0, 3).join(', ') : '--';
    }

    if (losersEl) {
        const losers = data.top_losers || [];
        losersEl.textContent = losers.length > 0 ? losers.slice(0, 3).join(', ') : '--';
    }
}

// ============================================================
// CURRENCY & BALANCE SETTINGS
// ============================================================

// Format currency with selected symbol
function formatCurrencyValue(amount) {
    const rate = state.currencyRates[state.currency] || 1;
    const converted = amount * rate;
    return `${state.currencySymbol}${converted.toLocaleString(undefined, {
        minimumFractionDigits: 2,
        maximumFractionDigits: 2
    })}`;
}

// Initialize currency selector
function initCurrencySelector() {
    const currencySelect = document.getElementById('currencySelect');
    if (!currencySelect) return;

    currencySelect.addEventListener('change', (e) => {
        const option = e.target.selectedOptions[0];
        state.currency = option.value;
        state.currencySymbol = option.dataset.symbol || '$';

        // Update all displayed values
        updateCurrencyDisplay();

        console.log(`Currency changed to: ${state.currency} (${state.currencySymbol})`);
    });
}

// Update all currency displays
function updateCurrencyDisplay() {
    // Refresh prices and positions with new currency
    loadPositions();
    loadBots();

    // Force chart refresh if possible or just wait for next update
    if (state.candleSeries) {
        loadChartData();
    }
}

// Initialize balance editor
function initBalanceEditor() {
    const btnEdit = document.getElementById('btnEditBalance');
    if (!btnEdit) return;

    btnEdit.addEventListener('click', () => {
        const currentCurrency = state.currency || 'USD';
        const rate = state.currencyRates[currentCurrency] || 1;

        // Show current balance in selected currency as default
        const currentDisplayBalance = (state.paperBalance * rate).toFixed(2);

        const newBalanceStr = prompt(`Enter new balance in ${currentCurrency}:`, currentDisplayBalance);

        if (newBalanceStr !== null && !isNaN(parseFloat(newBalanceStr))) {
            const inputAmount = parseFloat(newBalanceStr);

            // Convert BACK to USD for backend storage
            const amountInUSD = inputAmount / rate;

            // Validate
            if (amountInUSD < 0) {
                alert('Balance cannot be negative');
                return;
            }

            state.paperBalance = amountInUSD;

            // Update backend balance
            fetch('/api/balance', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ cash: amountInUSD })
            }).then(res => res.json())
                .then(data => {
                    if (data.success) {
                        console.log(`Balance updated: ${inputAmount} ${currentCurrency} -> ${amountInUSD.toFixed(2)} USD`);
                        state.paperBalance = amountInUSD;
                        loadPositions(); // Updates UI
                        showNotification(`✅ Balance updated to ${formatCurrencyValue(amountInUSD)}`);
                    }
                }).catch(err => console.error('Failed to update balance:', err));
        }
    });
}

// Show News Analysis Modal
async function showNewsAnalysisModal(sentiment, news) {
    const modal = document.getElementById('newsModal');
    const overlay = document.getElementById('newsModalOverlay');
    if (!modal || !overlay) return;

    modal.style.display = 'flex';
    overlay.style.display = 'block';

    updateNewsModal(news);
}

function updateNewsModal(news) {
    const container = document.getElementById('modalNewsList');
    if (!container) return;

    if (!news || news.length === 0) {
        container.innerHTML = '<div class="no-news">No recent news found for this market.</div>';
        return;
    }

    const itemsHtml = news.map(item => {
        const score = item.sentiment_score !== undefined ? item.sentiment_score : 0;
        let sentiment = 'neutral';
        if (score > 0.1) sentiment = 'bullish';
        else if (score < -0.1) sentiment = 'bearish';

        return `
            <div class="news-item ${sentiment}">
                <div class="news-sentiment-badge">${sentiment.toUpperCase()}</div>
                <div class="news-content">
                    <div class="news-title">${item.title}</div>
                    <div class="news-meta">${item.source} • ${new Date(item.published_at).toLocaleTimeString()}</div>
                </div>
            </div>
        `;
    }).join('');

    container.innerHTML = itemsHtml;
}

function closeNewsModal() {
    const modal = document.getElementById('newsModal');
    const overlay = document.getElementById('newsModalOverlay');
    if (modal) modal.style.display = 'none';
    if (overlay) overlay.style.display = 'none';
}

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initCurrencySelector();
    initBalanceEditor();
    initMarketTabs();
    initEventListeners();

    // Initial data load
    loadChartData();
    loadPositions();
    loadBots();
    loadNews();

    // Start bot list refresher
    setInterval(loadBots, 5000);
});

// ============================================================
// MARKET COMMAND CENTER (STAR FEATURES)
// ============================================================

function renderPulseGauge(data) {
    const score = data.pulse_score;
    const label = data.pulse_label;

    // Update Value Text
    const valEl = document.getElementById('pulseValue');
    if (valEl) valEl.textContent = score;

    // Update Label
    const labelEl = document.querySelector('.pulse-label');
    if (labelEl) labelEl.textContent = label;

    // Animate Gauge Fill
    const fill = document.getElementById('pulseGaugeFill');
    if (fill) {
        const offset = 251.2 - (score / 100 * 251.2);
        fill.style.strokeDashoffset = offset;
    }

    // Rotate Needle
    const needleLine = document.getElementById('pulseNeedleLine');
    if (needleLine) {
        const rotation = (score / 100 * 180) - 90;
        needleLine.setAttribute('transform', `rotate(${rotation}, 100, 100)`);
    }
}

let aiTypeInterval = null;
function renderAIInsights(data) {
    const textEl = document.getElementById('aiInsightText');
    const timeEl = document.getElementById('aiTimestamp');
    if (timeEl) timeEl.textContent = data.timestamp;

    if (!textEl) return;
    if (aiTypeInterval) clearInterval(aiTypeInterval);

    const fullText = data.ai_thought;
    textEl.textContent = '';
    let i = 0;

    aiTypeInterval = setInterval(() => {
        if (i < fullText.length) {
            textEl.textContent += fullText.charAt(i);
            i++;
        } else {
            clearInterval(aiTypeInterval);
        }
    }, 40);
}

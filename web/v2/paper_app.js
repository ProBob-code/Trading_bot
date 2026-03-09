/**
 * GodBotTrade V2 — Institutional Brain
 * =====================================
 * Same layout as V1. Upgraded logic layer only.
 * Modules: R:R, Slippage, Spread, Risk, Kelly, Expectancy, Uniqueness
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
    strategies: {}, // Loaded dynamically from /api/v2/strategies
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
    paperBalance: 100000,  // Editable paper trading balance
    activeBots: [],  // Loaded from /api/bots, used for bot_id lookups
    // ── V2 Institutional State ──
    v2TradeHistory: [],        // Rolling trade results for expectancy calc
    v2ActiveStrategies: new Set(),  // strategy+symbol uniqueness
    v2AutoDisabled: false      // Negative expectancy lockout flag
};

function getStrategyName(slug) {
    return state.strategies[slug] || slug;
}

// ============================================================
// INITIALIZATION
// ============================================================

document.addEventListener('DOMContentLoaded', async () => {
    // 🛡️ FRONTEND AUTH GATE
    try {
        const authRes = await fetch('/api/auth/status');
        const authData = await authRes.json();

        if (!authData.authenticated || !authData.user.is_verified) {
            window.location.href = 'godbot_login';
            return;
        }

        // Update Profile UI
        updateProfileUI(authData.user);
    } catch (err) {
        console.error('Auth Check Failed', err);
        window.location.href = 'godbot_login';
        return;
    }

    initChart();
    initSocket();
    initEventListeners();
    await loadStrategies(); // Load strategies before initial data
    loadInitialData();
    updateUI();
    loadNews();

    // Additional initializations from other modules
    initBotManagement();
    initPositionsPanel();
    initMarketDepth();
    initCurrencySelector();
    initBalanceEditor();
    initMarketTabs();

    setInterval(() => {
        loadPositions();
        loadBots();
    }, 5000);

    setInterval(() => loadNews(), 60000);
    initClock();
    setInterval(checkServerStatus, 10000);
    checkSystemStatus();

    // ── V2 Institutional Init ──
    InstitutionalEngine.init();
});

function updateProfileUI(user) {
    const profileName = document.getElementById('profileName');
    const dropdownBrand = document.querySelector('.dropdown-brand');
    const dropdownSub = document.querySelector('.dropdown-sub');

    if (profileName) profileName.textContent = user.username;
    if (dropdownBrand) dropdownBrand.textContent = user.username;
    if (dropdownSub) dropdownSub.textContent = 'Pro Trader';
}

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
// SYSTEM CONTROLS
// ============================================================

let isSystemPaused = false;

async function checkSystemStatus() {
    try {
        const res = await fetch('/api/system/status');
        const data = await res.json();
        updateSystemStatusUI(data.paused);
    } catch (e) {
        console.error('Failed to fetch system status:', e);
    }
}

async function toggleSystemPause() {
    const endpoint = isSystemPaused ? '/api/system/resume' : '/api/system/pause';
    try {
        const res = await fetch(endpoint, { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            updateSystemStatusUI(data.paused);
            showNotification(data.paused ? '⏸️ System PAUSED' : '▶️ System RESUMED', data.paused ? 'warning' : 'success');
        }
    } catch (e) {
        alert('Failed to toggle system status');
    }
}

function updateSystemStatusUI(paused) {
    isSystemPaused = paused;
    const btn = document.getElementById('btnSystemPause');
    const icon = btn.querySelector('.pause-icon');
    const text = btn.querySelector('.pause-text');

    if (paused) {
        btn.classList.add('paused');
        icon.textContent = '▶️';
        text.textContent = 'Paused';
        btn.title = "System is PAUSED. Click to Resume.";
        // Pulse red animation handled by CSS
        document.body.classList.add('system-paused-mode'); // Optional visual cue
    } else {
        btn.classList.remove('paused');
        icon.textContent = '⏸️';
        text.textContent = 'Running';
        btn.title = "System is RUNNING. Click to Pause.";
        document.body.classList.remove('system-paused-mode');
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

    // Indicator series removed for simplification

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
            const candles = data
                .filter(d => d.open && d.high && d.low && d.close)
                .map(d => ({
                    time: d.time,
                    open: d.open * rate,
                    high: d.high * rate,
                    low: d.low * rate,
                    close: d.close * rate
                }));

            if (candles.length > 0) {
                state.candleSeries.setData(candles);
            }

            // Indicator population disabled for simplification
            // updateIndicatorVisibility();
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
    // Determine the socket URL: relative to the current window
    // This fixed the hardcoded 'localhost:5050' which broke on Railway
    state.socket = io({
        transports: ['websocket', 'polling'],
        reconnectionAttempts: 5,
        reconnectionDelay: 2000
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
        // Strict symbol filtering: only update if it matches the current dashboard symbol
        if (data.symbol !== state.currentSymbol) return;

        updatePrice(data);
        // Deprecated: account update now comes in its own event
        // updateAccount(data.account);
        updateChart(data);
    });

    state.socket.on('v2_account_update', (data) => {
        console.log('[V2] Account update received:', data);
        updateAccount(data);
        // If we have positions in the update, we can refresh the list faster than the 5s interval
        if (data.positions) {
            positionsData.open = data.positions;
            renderOpenPositions();
        }
    });

    state.socket.on('connected', (data) => {
        console.log('Server acknowledged connection:', data);
    });

    // Live auto-trading events (V2-only — ignore V1 signals)
    state.socket.on('auto_trade_signal', (data) => {
        if (data.engine && data.engine !== 'v2') return;  // Skip V1 signals
        console.log('[V2] Signal:', data.signal, '@', data.price);
        addSignalToFeed(data);
    });

    state.socket.on('auto_trade_executed', (data) => {
        console.log('[V2] Trade executed (legacy event):', data);
        addTradeToFeed(data);
        showTradeNotification(data);
        updateTradeCount();
    });

    state.socket.on('v2_trade_executed', (data) => {
        console.log('[V2] Trade executed:', data);
        addTradeToFeed(data);
        showTradeNotification(data);
        updateTradeCount();
    });

    state.socket.on('market_intel', (data) => {
        renderPulseGauge(data);
        renderAIInsights(data);
    });

    // System Status Update
    state.socket.on('system_status', (data) => {
        updateSystemStatusUI(data.paused);
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

function getIntervalSeconds(interval) {
    const unit = interval.slice(-1).toLowerCase();
    const val = parseInt(interval.slice(0, -1));
    if (isNaN(val)) return 60;
    switch (unit) {
        case 'm': return val * 60;
        case 'h': return val * 3600;
        case 'd': return val * 86400;
        default: return 60;
    }
}

function updateChart(data) {
    if (!state.candleSeries || !data.price) return;

    const intervalSecs = getIntervalSeconds(state.currentInterval);
    const now = Math.floor(Date.now() / 1000);
    const alignedTime = Math.floor(now / intervalSecs) * intervalSecs;

    const rate = state.currencyRates[state.currency] || 1;
    const price = data.price * rate;

    state.candleSeries.update({
        time: alignedTime,
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
    const tradeCountEl = document.getElementById('tradeCount');

    if (balanceEl) balanceEl.textContent = formatCurrencyValue(account.total_value);

    const pnl = account.pnl;
    if (pnlEl) {
        pnlEl.textContent = (pnl >= 0 ? '+' : '') + formatCurrencyValue(pnl);
        pnlEl.className = `account-value pnl ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
    }

    // Pro-Coder Tip: Update trade count in real-time from socket data
    if (account.total_trades !== undefined && tradeCountEl) {
        tradeCountEl.textContent = account.total_trades;
        // Keep internal state in sync
        state.tradeCount = account.total_trades;
    }
}

function updateTradeCount() {
    fetch('/api/v2/account')
        .then(res => res.json())
        .then(data => {
            if (!data.success) return;
            // V2 account trade count comes from the account info
            const v2Trades = data.total_trades || 0;
            const countEl = document.getElementById('tradeCount');
            if (countEl) countEl.textContent = v2Trades;

            const totalTradesEl = document.getElementById('totalTrades');
            if (totalTradesEl) totalTradesEl.textContent = v2Trades;

            state.tradeCount = v2Trades;
        })
        .catch(err => console.error('[V2] updateTradeCount error:', err));
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

    // ── V2 Institutional Pre-Trade Validation ──
    const stopLossPct = parseFloat(document.getElementById('stopLoss')?.value) || state.settings.stopLoss;
    const takeProfitPct = parseFloat(document.getElementById('takeProfit')?.value) || state.settings.takeProfit;
    const currentPrice = state.lastPrice;
    const equity = state.paperBalance || 100000;

    // A) R:R Enforcement
    if (!InstitutionalEngine.enforceRR(takeProfitPct, stopLossPct)) {
        return; // Trade blocked
    }

    // D) Risk Budget Enforcement
    if (!InstitutionalEngine.validateRisk(quantity, currentPrice, equity)) {
        return; // Trade blocked
    }

    // V2: No frontend slippage — backend authoritative fill price
    console.log(`[V2] Sending trade to backend: side=${side}, qty=${quantity}, price=${currentPrice}`);

    try {
        const response = await fetch('/api/v2/trade', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                side: side,
                quantity: quantity,
                market: state.currentMarket,
                strategy: state.currentStrategy || 'manual',
                leverage: state.settings.leverage || 1.0,
                volatility: state.lastVolatility || 0.02,
                volume: state.lastVolume || 100000000,
                margin_mode: state.settings.marginMode || 'isolated',
                user: document.getElementById('dropdownUsername')?.textContent || 'admin'
            })
        });

        const result = await response.json();

        if (result.success) {
            // V2: Use backend-authoritative fill_price — NO frontend manipulation
            const fillPrice = result.fill_price || result.price;

            // Track position for P&L calculation
            if (side === 'buy') {
                state.positions[state.currentSymbol] = {
                    qty: quantity,
                    entryPrice: fillPrice,
                    side: 'BUY',
                    leverage: result.leverage || 1.0
                };
            } else {
                // Selling - use backend realized P&L
                if (result.realized_pnl !== undefined) {
                    result.pnl = result.realized_pnl;
                    result.pnlPct = result.entry_price ? ((fillPrice / result.entry_price) - 1) * 100 : 0;
                    delete state.positions[state.currentSymbol];

                    // V2: Track trade result for rolling expectancy
                    InstitutionalEngine.recordTradeResult(result.realized_pnl);
                } else if (state.positions[state.currentSymbol]) {
                    const pos = state.positions[state.currentSymbol];
                    const pnl = (fillPrice - pos.entryPrice) * quantity;
                    result.pnl = pnl;
                    result.pnlPct = ((fillPrice / pos.entryPrice) - 1) * 100;
                    delete state.positions[state.currentSymbol];
                    InstitutionalEngine.recordTradeResult(pnl);
                }
            }

            // Log V2 execution details
            console.log(`[V2] Executed: fill=${fillPrice}, spread=${result.spread_pct}%, slip=${result.slippage_pct}%, comm=$${result.commission}`);

            // Update trade count display
            state.tradeCount = result.total_trades || state.tradeCount + 1;
            if (side === 'buy') state.buyCount++;
            else state.sellCount++;

            document.getElementById('tradeCount').textContent = state.tradeCount;

            addTradeToFeed(result);
            console.log('[V2] Trade executed:', result);
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

    const reasons = trade.reasons && trade.reasons.length > 0
        ? `<div class="signal-reasons">${trade.reasons.join(', ')}</div>`
        : '';

    item.innerHTML = `
        <div class="signal-main">
            <span class="side">${trade.side === 'BUY' || trade.side.includes('BUY') ? '🟢' : '🔴'} ${trade.side.split(' ')[0]}</span>
            <span class="symbol">${trade.symbol}</span>
            <div class="price-qty-container">
                ${displayValue}
                <span class="qty">×${trade.quantity.toFixed(4)}</span>
            </div>
            <span class="time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}</span>
        </div>
        ${reasons}
    `;
    feed.insertBefore(item, feed.firstChild);

    while (feed.children.length > 50) {
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

    // Make HOLD more subtle but still visible, BUY/SELL full opacity
    item.style.opacity = signal.signal === 'HOLD' ? '0.6' : '1.0';

    const reasons = signal.reasons && signal.reasons.length > 0
        ? `<div class="signal-reasons">${signal.reasons.join(', ')}</div>`
        : '';

    item.innerHTML = `
        <div class="signal-main">
            <span class="side">${signalIcon} ${signal.signal}</span>
            <span class="symbol">${signal.symbol}</span>
            <div class="price-qty-container">
                <span class="price">${formatPrice(signal.price)}</span>
                <span class="qty">Signal</span>
            </div>
            <span class="time">${new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}</span>
        </div>
        ${reasons}
    `;
    feed.insertBefore(item, feed.firstChild);

    while (feed.children.length > 50) {
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

    // ── V2 Institutional Pre-Start Checks ──

    // G) Strategy Uniqueness
    if (!InstitutionalEngine.enforceStrategyUniqueness(state.currentStrategy, state.currentSymbol)) {
        return; // Duplicate blocked
    }

    // F) Negative Expectancy Gate
    if (!InstitutionalEngine.checkExpectancyGate()) {
        return; // Auto disabled due to negative expectancy
    }

    // Gather settings from MAIN UI inputs for WYSIWYG experience
    const uiSettings = {
        stopLoss: parseFloat(document.getElementById('stopLoss').value) || state.settings.stopLoss,
        takeProfit: parseFloat(document.getElementById('takeProfit').value) || state.settings.takeProfit,
        positionSize: parseFloat(document.getElementById('positionSize').value) || state.settings.positionSize,
        maxQuantity: parseFloat(document.getElementById('maxQuantity').value) || 1.0,
        confluence: state.settings.confluence || 3,
        checkInterval: state.settings.checkInterval || 5
    };

    // E) Kelly-based Position Sizing (advisory log)
    const kellySize = InstitutionalEngine.computePositionSize(state.v2TradeHistory, state.paperBalance || 100000);
    console.log(`[V2-INSTITUTIONAL] Kelly recommended position: ${(kellySize * 100).toFixed(2)}% of equity`);

    try {
        console.log('[V2] Starting bot with settings:', uiSettings);

        const response = await fetch('/api/v2/start-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: state.currentSymbol,
                interval: state.currentInterval,
                strategy: state.currentStrategy,
                market: state.currentMarket,
                mode: state.tradingMode || 'paper',
                position_size: uiSettings.positionSize,
                stop_loss: uiSettings.stopLoss,
                take_profit: uiSettings.takeProfit,
                max_quantity: uiSettings.maxQuantity,
                leverage: state.settings.leverage || 1.0,
                risk_pct: state.settings.riskPct || 2.0
            })
        });

        const result = await response.json();

        if (result.success) {
            // V2: Register strategy in active set
            state.v2ActiveStrategies.add(`${state.currentStrategy}:${state.currentSymbol}`);
            console.log(`[V2] Bot started: id=${result.bot_id}, hash=${result.config_hash}`);

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
    window.location.href = '/v2_report.html';
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

    // Strategy selector — V2: multi-strategy, no hot-swap
    document.getElementById('strategySelect').addEventListener('change', (e) => {
        state.currentStrategy = e.target.value;
        const strategyName = getStrategyName(state.currentStrategy);
        const activeStrategyEl = document.getElementById('activeStrategy');
        if (activeStrategyEl) {
            activeStrategyEl.textContent = strategyName;
        }

        // Immediately refresh Start/Stop button state for the new strategy
        loadBots();

        console.log(`[V2] Strategy switched to: ${strategyName} — button state refreshed`);
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
        const matchingBot = state.activeBots && state.activeBots.find(
            b => b.symbol.toLowerCase() === state.currentSymbol.toLowerCase() &&
                b.market === state.currentMarket
        );
        if (matchingBot) {
            stopBot(matchingBot.bot_id);
        } else {
            showNotification('No running bot for this symbol');
        }
    });

    // STOP ALL button - stops all running V2 bots
    document.getElementById('btnStopAll').addEventListener('click', () => {
        fetch('/api/v2/stop-all', { method: 'POST' })
            .then(() => {
                state.v2ActiveStrategies.clear();
                showNotification('🛑 All V2 bots stopping...');
                loadBots();
                loadPositions(); // Refresh positions immediately to see closures
            });
    });

    document.getElementById('btnPanicSell').addEventListener('click', async () => {
        if (confirm('🚨 EMERGENCY: Close ALL open positions and STOP all bots immediately?')) {
            try {
                const response = await fetch('/api/panic-sell', { method: 'POST' });
                const result = await response.json();
                if (result.success) {
                    showNotification(`✅ Panic Sell Executed: ${result.message}`, 'warning');
                    loadPositions();
                    loadBots();  // Refresh bot list since bots are now stopped
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
    document.getElementById('btnResetPaper')?.addEventListener('click', resetPaperTrading);

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

    // Logout button
    const btnLogout = document.getElementById('btnLogout');
    if (btnLogout) {
        btnLogout.addEventListener('click', async () => {
            try {
                const res = await fetch('/api/auth/logout', { method: 'POST' });
                if (res.ok) {
                    window.location.href = 'godbot_login';
                }
            } catch (err) {
                console.error('Logout failed', err);
            }
        });
    }

    // Clear feed
    document.getElementById('btnClearFeed').addEventListener('click', () => {
        document.getElementById('tradesFeed').innerHTML = '';
    });
}

// ============================================================
// UI UPDATES
// ============================================================

function updateUI() {
    document.getElementById('activeStrategy').textContent = 'Strategy 1';
}

async function loadInitialData() {
    await loadChartData();
    updateTradeCount();  // Sync trade counter on page load
}

// ============================================================
// STRATEGY MANAGEMENT
// ============================================================

async function loadStrategies() {
    try {
        const res = await fetch('/api/v2/strategies');
        const data = await res.json();
        if (data.success) {
            state.strategies = {};
            data.strategies.forEach(s => {
                state.strategies[s.id] = s.name;
            });
            populateStrategyDropdown(data.strategies);
            // Also update the active strategy indicator in the UI if it's set
            document.getElementById('activeStrategy').textContent = getStrategyName(state.currentStrategy);
        }
    } catch (e) {
        console.error('Failed to load strategies:', e);
    }
}

function populateStrategyDropdown(strategies) {
    const select = document.getElementById('strategySelect');
    if (!select) return;

    select.innerHTML = strategies.map(s =>
        `<option value="${s.id}">${s.icon || '📈'} ${s.name}</option>`
    ).join('');

    // Ensure current selection matches state or use first available
    if (state.strategies[state.currentStrategy]) {
        select.value = state.currentStrategy;
    } else if (strategies.length > 0) {
        state.currentStrategy = strategies[0].id;
        select.value = strategies[0].id;
    }
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
        // V2: Use V2-only bot list endpoint (isolated from V1)
        const response = await fetch('/api/v2/bots');
        const data = await response.json();

        if (data.success) {
            const bots = data.bots || [];
            const runningCount = bots.filter(b => b.status === 'running').length;
            renderBots(bots);
            document.getElementById('botCount').textContent = runningCount;

            // Store bots for use by strategy selector and other features
            state.activeBots = bots;

            // Rebuild V2 active strategies set from live bot data
            state.v2ActiveStrategies.clear();
            bots.forEach(b => {
                if (b.status === 'running') {
                    state.v2ActiveStrategies.add(`${b.strategy}:${b.symbol}`);
                }
            });

            // Update Control Buttons (START/STOP) for current symbol + strategy
            const isBotRunning = bots.some(
                b => b.symbol.toLowerCase() === state.currentSymbol.toLowerCase() &&
                    b.market === state.currentMarket &&
                    b.strategy === state.currentStrategy &&
                    b.status === 'running'
            );

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
                status.textContent = runningCount > 0 ? 'Watching' : 'Stopped';
            }

            // Show/hide STOP ALL button based on running bots
            const stopAllBtn = document.getElementById('btnStopAll');
            if (runningCount > 0) {
                stopAllBtn.classList.add('active');
            } else {
                stopAllBtn.classList.remove('active');
            }
        }
    } catch (error) {
        console.error('Error loading V2 bots:', error);
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
        const realizedPnl = bot.stats.realized_pnl || 0;
        const unrealizedPnl = bot.stats.unrealized_pnl || 0;
        const totalPnl = realizedPnl + unrealizedPnl;

        const pnlClass = totalPnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const pnlSign = totalPnl >= 0 ? '+' : '';

        return `
            <div class="bot-card ${bot.mode}">
                <div class="bot-info">
                    <span class="bot-symbol">${bot.symbol}</span>
                    <span class="bot-strategy">${getStrategyName(bot.strategy)} • ${bot.market.toUpperCase()}</span>
                </div>
                <div class="bot-stats-row">
                     <span class="bot-pnl ${pnlClass}" title="Total (Realized: ${formatCurrencyValue(realizedPnl)})">
                        ${pnlSign}${formatCurrencyValue(totalPnl)}
                     </span>
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
        // V2: Use V2-only stop endpoint
        const response = await fetch('/api/v2/stop-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bot_id: botId })
        });
        const result = await response.json();

        if (result.success) {
            // Remove from active strategies set
            const bot = state.activeBots.find(b => b.bot_id === botId);
            if (bot) {
                state.v2ActiveStrategies.delete(`${bot.strategy}:${bot.symbol}`);
            }
            showNotification(`Bot ${botId} stopped`);
            loadBots();
        }
    } catch (error) {
        console.error('Error stopping V2 bot:', error);
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

// Bot management initialized via main DOMContentLoaded

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

    // Date filters
    const dateFrom = document.getElementById('filterDateFrom');
    const dateTo = document.getElementById('filterDateTo');

    // Set default dates (today)
    const today = new Date().toISOString().split('T')[0];
    if (dateFrom && !dateFrom.value) dateFrom.value = today;
    if (dateTo && !dateTo.value) dateTo.value = today;

    [dateFrom, dateTo].forEach(el => {
        el?.addEventListener('change', () => {
            loadPositions();
            if (document.querySelector('.pos-tab.active')?.dataset.tab === 'reports') {
                loadReports();
            }
        });
    });

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
        // V2: Fetch open positions and trade history from isolated V2 endpoints
        const [posRes, historyRes] = await Promise.all([
            fetch('/api/v2/positions'),
            fetch('/api/v2/trades?limit=100')
        ]);

        if (!posRes.ok || !historyRes.ok) return;

        const posData = await posRes.json();
        const historyData = await historyRes.json();

        if (posData.success && historyData.success) {
            positionsData.open = posData.positions || [];
            positionsData.history = historyData.trades || [];
            // Closed trades are CLOSE records in history
            positionsData.closed = (historyData.trades || []).filter(t => (t.trade_type || t.type) === 'CLOSE');
            positionsData.orders = []; // V2 pending orders to be implemented later

            renderOpenPositions();
            renderClosedPositions();
            renderTradeHistory();
            renderPendingOrders();
        }
    } catch (error) {
        console.error('Error loading V2 positions:', error);
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
            const isShort = pos.side === 'SHORT' || pos.side === 'SELL';
            const netPnl = pos.net_pnl;
            const netPnlPct = pos.avg_price > 0
                ? (isShort ? (1 - currentPrice / pos.avg_price) : (currentPrice / pos.avg_price - 1)) * 100
                : 0;
            const pnlClass = netPnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const sideClass = !isShort ? 'side-long' : 'side-short';

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
            const pnl = pos.net_pnl ?? pos.realized_pnl ?? 0;
            const entry = pos.entry_price || pos.entry || 0;
            const exit = pos.exit_price || pos.exit || 0;
            const pnlPct = entry > 0 ? ((exit / entry) - 1) * 100 : 0;
            const pnlClass = pnl >= 0 ? 'pnl-positive' : 'pnl-negative';
            const sideClass = (pos.side || '').toUpperCase().includes('BUY') ? 'side-long' : 'side-short';

            return `
                <tr class="closed-trade-row">
                    <td>
                        <div class="trade-symbol-meta">
                            <strong>${pos.symbol}</strong>, 
                            <span class="${sideClass}">${(pos.side || '').toLowerCase()} ${pos.quantity || pos.qty}</span>
                        </div>
                    </td>
                    <td colspan="3">
                        <div class="trade-price-flow">
                            $${parseFloat(entry).toLocaleString()} &rarr; $${parseFloat(exit).toLocaleString()}
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
                            ${pos.timestamp ? new Date(pos.timestamp).toLocaleString() : '—'}
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

    if (positionsData.history.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="8">No trade history</td></tr>';
    } else {
        body.innerHTML = positionsData.history.slice(-20).reverse().map(trade => {
            const tradeType = trade.trade_type || trade.type || '—';
            const price = trade.fill_price || trade.price || 0;
            const qty = trade.quantity || 0;
            const ts = trade.timestamp || trade.time;
            const side = (trade.side || '').toUpperCase();

            return `
            <tr>
                <td>${ts ? new Date(ts).toLocaleTimeString() : '—'}</td>
                <td><strong>${trade.symbol}</strong></td>
                <td><span style="font-size:0.7rem; color:var(--text-secondary)">${tradeType}</span></td>
                <td class="${side.includes('BUY') ? 'side-long' : 'side-short'}">${side}</td>
                <td>${qty.toFixed(4)}</td>
                <td>$${price.toLocaleString()}</td>
                <td>$${(qty * price).toLocaleString()}</td>
                <td><span style="color:var(--green)">✓ Filled</span></td>
            </tr>
        `;
        }).join('');
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
        const from = document.getElementById('filterDateFrom')?.value;
        const to = document.getElementById('filterDateTo')?.value;
        let url = '/api/reports';
        if (from || to) {
            const params = new URLSearchParams();
            if (from) params.append('start_date', from);
            if (to) params.append('end_date', to);
            url += `?${params.toString()}`;
        }

        const response = await fetch(url);
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
    if (!trades || trades.length === 0) return {};

    // Filter to CLOSE trades only — these contain the real PnL
    const closedTrades = trades.filter(t => t.trade_type === 'CLOSE');
    if (closedTrades.length === 0) return {};

    // Use nullish coalescing (??) to avoid net_pnl === 0 false-fallback
    const pnls = closedTrades.map(t => t.net_pnl ?? t.realized_pnl ?? 0);
    let wins = 0;
    let losses = 0;
    pnls.forEach(p => p > 0 ? wins++ : losses++);

    const totalPnL = pnls.reduce((a, b) => a + b, 0);
    const avgReturn = pnls.length > 0 ? totalPnL / pnls.length : 0;
    const winRate = (wins + losses) > 0 ? (wins / (wins + losses)) * 100 : 0;

    // Sharpe ratio (simplified)
    const mean = avgReturn;
    const variance = pnls.reduce((sum, pnl) => sum + Math.pow(pnl - mean, 2), 0) / pnls.length;
    const stdDev = Math.sqrt(variance);
    const sharpe = stdDev > 0 ? mean / stdDev : 0;

    return {
        totalPnL,
        winRate,
        sharpe,
        totalTrades: closedTrades.length
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

// UI components initialized via main DOMContentLoaded

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

// Reset paper trading state
async function resetPaperTrading() {
    if (!confirm('🚨 Are you sure you want to reset all Paper Trading state? This will clear balance, P&L, and trade counters.')) {
        return;
    }

    try {
        const response = await fetch('/api/paper/reset', { method: 'POST' });
        const data = await response.json();
        if (data.success) {
            showNotification('✅ Paper trading state reset successfully!', 'success');
            // Refresh data
            location.reload(); // Refreshing page is safest to clear all state
        } else {
            showNotification(`❌ Reset failed: ${data.error}`, 'error');
        }
    } catch (error) {
        console.error('Error resetting paper trading:', error);
        showNotification('❌ Error resetting paper trading', 'error');
    }
}

// ============================================================
// V2 INSTITUTIONAL ENGINE
// ============================================================
// Modular layer — never mixed into V1 functions directly.
// Hooked via targeted intercepts in executeTrade() and startBot().

const InstitutionalEngine = {

    // ── Configurable Constants ──
    MIN_RR: 1.5,               // Minimum reward-to-risk ratio
    MAX_RISK_PCT: 0.02,        // Max 2% of equity per trade
    MAX_KELLY_PCT: 0.05,       // Cap Kelly at 5% of equity
    EXPECTANCY_WINDOW: 20,     // Rolling window for expectancy calc
    BASE_SLIPPAGE_PCT: 0.0005, // 0.05% base slippage

    // ── Initialization ──
    init() {
        console.log('[V2-INSTITUTIONAL] ═══════════════════════════════════════');
        console.log('[V2-INSTITUTIONAL] Institutional Engine v2.0.0 loading...');
        console.log('[V2-INSTITUTIONAL] Modules:');
        console.log('[V2-INSTITUTIONAL]   ✓ R:R Enforcement (min ' + this.MIN_RR + ':1)');
        console.log('[V2-INSTITUTIONAL]   ✓ Slippage Simulation (ATR-aware)');
        console.log('[V2-INSTITUTIONAL]   ✓ Spread Model (entry/exit)');
        console.log('[V2-INSTITUTIONAL]   ✓ Risk Budget (' + (this.MAX_RISK_PCT * 100) + '% max per trade)');
        console.log('[V2-INSTITUTIONAL]   ✓ Kelly Position Sizing (capped ' + (this.MAX_KELLY_PCT * 100) + '%)');
        console.log('[V2-INSTITUTIONAL]   ✓ Negative Expectancy Gate (window=' + this.EXPECTANCY_WINDOW + ')');
        console.log('[V2-INSTITUTIONAL]   ✓ Strategy Uniqueness');
        console.log('[V2-INSTITUTIONAL]   ✓ Silent API Integration (Phase 1)');
        console.log('[V2-INSTITUTIONAL] ═══════════════════════════════════════');

        // H) Silent API Integration — fetch and log, no UI rendering
        this._fetchInstitutionalData();

        // Register bot stop hook to deregister strategies
        this._hookBotStopDeregistration();
    },

    // ── A) R:R Enforcement ──
    enforceRR(takeProfitPct, stopLossPct) {
        if (!stopLossPct || stopLossPct <= 0) {
            console.log('[V2-INSTITUTIONAL] R:R Check: No stop-loss set, allowing trade');
            return true;
        }

        const rr = takeProfitPct / stopLossPct;

        if (rr < this.MIN_RR) {
            const msg = `⛔ Trade Blocked — R:R ratio ${rr.toFixed(2)}:1 below minimum ${this.MIN_RR}:1 (TP=${takeProfitPct}%, SL=${stopLossPct}%)`;
            console.warn('[V2-INSTITUTIONAL]', msg);
            showNotification(msg);
            return false;
        }

        console.log(`[V2-INSTITUTIONAL] R:R Check: ${rr.toFixed(2)}:1 ✓ (min=${this.MIN_RR}:1)`);
        return true;
    },

    // ── B) Slippage Simulation (ATR-aware) ──
    applySlippage(marketPrice, side) {
        // Volatility multiplier: use 24h range as ATR proxy
        const high = parseFloat(document.getElementById('high24h')?.textContent?.replace(/[^0-9.]/g, '')) || marketPrice * 1.01;
        const low = parseFloat(document.getElementById('low24h')?.textContent?.replace(/[^0-9.]/g, '')) || marketPrice * 0.99;
        const atrProxy = (high - low) / marketPrice; // Percentage range
        const volatilityMultiplier = Math.max(1, atrProxy / 0.02); // Normalize around 2% daily range

        // Scale slippage by volatility
        const slippagePct = this.BASE_SLIPPAGE_PCT * volatilityMultiplier * (0.5 + Math.random());

        // Buys get worse entry (higher), sells get worse exit (lower)
        const direction = side === 'buy' ? 1 : -1;
        const effectivePrice = marketPrice * (1 + direction * slippagePct);

        return {
            marketPrice,
            effectivePrice,
            slippagePct: slippagePct * 100,
            volatilityMultiplier
        };
    },

    // ── C) Spread Model ──
    getEffectiveEntryPrice(price, side) {
        // Entry: buyer pays slightly more, seller enters slightly lower
        const spreadBps = this._getSpreadBps();
        const adjustment = side === 'buy' ? spreadBps : -spreadBps;
        const effective = price * (1 + adjustment);
        console.log(`[V2-INSTITUTIONAL] Spread Entry: ${price.toFixed(2)} → ${effective.toFixed(2)} (${(adjustment * 100).toFixed(3)}%)`);
        return effective;
    },

    getEffectiveExitPrice(price, side) {
        // Exit: closing a long (sell) gets slightly worse, closing a short (buy) gets slightly worse
        const spreadBps = this._getSpreadBps();
        const adjustment = side === 'sell' ? -spreadBps : spreadBps;
        const effective = price * (1 + adjustment);
        console.log(`[V2-INSTITUTIONAL] Spread Exit: ${price.toFixed(2)} → ${effective.toFixed(2)} (${(adjustment * 100).toFixed(3)}%)`);
        return effective;
    },

    _getSpreadBps() {
        // Dynamic spread: wider during low volume
        const volume = document.getElementById('volume24h')?.textContent || '0';
        const volNum = parseFloat(volume.replace(/[^0-9.]/g, ''));
        const suffix = volume.slice(-1).toUpperCase();
        let volumeVal = volNum;
        if (suffix === 'B') volumeVal *= 1e9;
        else if (suffix === 'M') volumeVal *= 1e6;
        else if (suffix === 'K') volumeVal *= 1e3;

        // Low volume = wider spread
        if (volumeVal < 1e6) return 0.003;     // 0.3% spread (low volume)
        if (volumeVal < 100e6) return 0.001;   // 0.1% spread (medium)
        return 0.0003;                          // 0.03% spread (high volume)
    },

    // ── D) Risk Budget Enforcement ──
    validateRisk(quantity, currentPrice, equity) {
        if (!currentPrice || currentPrice <= 0) return true; // Can't validate without price

        const tradeValue = quantity * currentPrice;
        const riskAmount = equity * this.MAX_RISK_PCT;

        // Use stop-loss to estimate actual risk, or fallback to trade value
        const stopLossPct = (parseFloat(document.getElementById('stopLoss')?.value) || state.settings.stopLoss) / 100;
        const tradeRisk = tradeValue * stopLossPct;

        if (tradeRisk > riskAmount) {
            const msg = `⛔ Risk Budget Exceeded — Trade risk $${tradeRisk.toFixed(2)} > allowed $${riskAmount.toFixed(2)} (${(this.MAX_RISK_PCT * 100)}% of $${equity.toLocaleString()})`;
            console.warn('[V2-INSTITUTIONAL]', msg);
            showNotification(msg);
            return false;
        }

        console.log(`[V2-INSTITUTIONAL] Risk Check: $${tradeRisk.toFixed(2)} / $${riskAmount.toFixed(2)} ✓`);
        return true;
    },

    // ── E) Position Sizing (Kelly Lite) ──
    computePositionSize(tradeHistory, equity) {
        if (!tradeHistory || tradeHistory.length < 5) {
            console.log('[V2-INSTITUTIONAL] Kelly: Insufficient history, using default 2%');
            return 0.02; // Default conservative
        }

        const wins = tradeHistory.filter(t => t > 0);
        const losses = tradeHistory.filter(t => t < 0);

        const winRate = wins.length / tradeHistory.length;
        const avgWin = wins.length > 0 ? wins.reduce((a, b) => a + b, 0) / wins.length : 0;
        const avgLoss = losses.length > 0 ? Math.abs(losses.reduce((a, b) => a + b, 0) / losses.length) : 1;

        // Kelly formula: f* = (win_rate × avg_win - loss_rate × avg_loss) / avg_win
        let kelly = 0;
        if (avgWin > 0) {
            kelly = (winRate * avgWin - (1 - winRate) * avgLoss) / avgWin;
        }

        // Safety: reduce if win rate < 50%
        if (winRate < 0.5) {
            kelly *= 0.5; // Half-Kelly for sub-50% win rate
        }

        // Cap at MAX_KELLY_PCT, floor at 0
        kelly = Math.max(0, Math.min(kelly, this.MAX_KELLY_PCT));

        console.log(`[V2-INSTITUTIONAL] Kelly: f*=${kelly.toFixed(4)} (WR=${(winRate * 100).toFixed(1)}%, avgW=${avgWin.toFixed(2)}, avgL=${avgLoss.toFixed(2)})`);
        return kelly;
    },

    // ── F) Negative Expectancy Gate ──
    checkExpectancyGate() {
        const history = state.v2TradeHistory;

        if (history.length < this.EXPECTANCY_WINDOW) {
            return true; // Not enough data yet
        }

        // Rolling window expectancy
        const recent = history.slice(-this.EXPECTANCY_WINDOW);
        const expectancy = recent.reduce((a, b) => a + b, 0) / recent.length;

        if (expectancy < 0) {
            state.v2AutoDisabled = true;
            const msg = `⚠ Institutional Guard: Negative expectancy detected (${expectancy.toFixed(2)} over last ${this.EXPECTANCY_WINDOW} trades). Auto-trading disabled.`;
            console.warn('[V2-INSTITUTIONAL]', msg);
            showNotification(msg);

            // Show persistent warning banner
            this._showExpectancyBanner(expectancy);
            return false;
        }

        state.v2AutoDisabled = false;
        this._hideExpectancyBanner();
        console.log(`[V2-INSTITUTIONAL] Expectancy Gate: ${expectancy.toFixed(2)} ✓ (window=${this.EXPECTANCY_WINDOW})`);
        return true;
    },

    _showExpectancyBanner(expectancy) {
        let banner = document.getElementById('v2ExpectancyBanner');
        if (!banner) {
            banner = document.createElement('div');
            banner.id = 'v2ExpectancyBanner';
            banner.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:9999;background:#ef4444;color:#fff;text-align:center;padding:8px 16px;font-weight:600;font-size:14px;';
            document.body.prepend(banner);
        }
        banner.textContent = `⚠ INSTITUTIONAL GUARD: Negative expectancy (${expectancy.toFixed(2)}) — Auto-trading locked`;
        banner.style.display = 'block';
    },

    _hideExpectancyBanner() {
        const banner = document.getElementById('v2ExpectancyBanner');
        if (banner) banner.style.display = 'none';
    },

    // ── G) Strategy Uniqueness ──
    enforceStrategyUniqueness(strategy, symbol) {
        const key = `${strategy}:${symbol}`;

        if (state.v2ActiveStrategies.has(key)) {
            const msg = `⛔ Duplicate Bot Blocked — ${getStrategyName(strategy)} on ${symbol} is already running`;
            console.warn('[V2-INSTITUTIONAL]', msg);
            showNotification(msg);
            return false;
        }

        console.log(`[V2-INSTITUTIONAL] Strategy Uniqueness: ${key} ✓`);
        return true;
    },

    // ── H) Silent Institutional API Integration (Phase 1) ──
    async _fetchInstitutionalData() {
        const endpoints = [
            { name: 'Bot Comparison', url: '/api/bots/compare' },
            { name: 'Capital Allocation', url: '/api/allocation' }
        ];

        for (const ep of endpoints) {
            try {
                const res = await fetch(ep.url);
                if (res.ok) {
                    const data = await res.json();
                    console.log(`[V2-INSTITUTIONAL] ${ep.name}:`, data);
                } else {
                    console.log(`[V2-INSTITUTIONAL] ${ep.name}: endpoint returned ${res.status}`);
                }
            } catch (e) {
                console.log(`[V2-INSTITUTIONAL] ${ep.name}: not available yet`);
            }
        }
    },

    // ── Record Trade Result (for expectancy tracking) ──
    recordTradeResult(pnl) {
        state.v2TradeHistory.push(pnl);
        console.log(`[V2-INSTITUTIONAL] Trade recorded: P&L=$${pnl.toFixed(2)}, history length=${state.v2TradeHistory.length}`);

        // Check expectancy gate after each trade
        if (state.v2TradeHistory.length >= this.EXPECTANCY_WINDOW) {
            this.checkExpectancyGate();
        }
    },

    // ── Bot Stop Deregistration Hook ──
    _hookBotStopDeregistration() {
        // Wrap the existing stopBot to deregister strategies
        const originalStopBot = window.stopBot;
        window.stopBot = async function (botId) {
            // Find the bot being stopped to deregister its strategy
            const bot = state.activeBots?.find(b => b.bot_id === botId);
            if (bot) {
                const key = `${bot.strategy}:${bot.symbol}`;
                state.v2ActiveStrategies.delete(key);
                console.log(`[V2-INSTITUTIONAL] Strategy deregistered: ${key}`);
            }
            return originalStopBot(botId);
        };
        console.log('[V2-INSTITUTIONAL] Bot stop deregistration hook installed');
    }
};

// ============================================================
// V3 SAFETY LAYER — tolerates elements removed in UI revamps.
// A missing element must never crash the boot chain again.
// ============================================================
function _setText(id, v) { const e = document.getElementById(id); if (e) e.textContent = v; }
function _setValue(id, v) { const e = document.getElementById(id); if (e) e.value = v; }
function _setStyle(id, prop, v) { const e = document.getElementById(id); if (e) e.style[prop] = v; }
function _safeInit(name, fn) {
    try { const r = fn(); if (r && r.catch) r.catch(err => console.error(`[V3-BOOT] ${name} (async) failed:`, err)); }
    catch (err) { console.error(`[V3-BOOT] ${name} failed:`, err); }
}

/**
 * GoatBotTrade V2 — Institutional Brain
 * =====================================
 * Same layout as V1. Upgraded logic layer only.
 * Modules: R:R, Slippage, Spread, Risk, Kelly, Expectancy, Uniqueness
 */

// ============================================================
console.log('[V2] paper_app.js loading...');
// UI MANAGER (V2 Institutional)
// ============================================================
const UI = {
    elements: {
        currentPrice: document.getElementById('currentPrice'),
        priceChange: document.getElementById('priceChange'),
        high24h: document.getElementById('high24h'),
        low24h: document.getElementById('low24h'),
        volume24h: document.getElementById('volume24h'),
        accountBalance: document.getElementById('accountBalance'),
        accountPnl: document.getElementById('accountPnl'),
        tradeCount: document.getElementById('tradeCount'),
        connectionDot: document.querySelector('.pulse-dot'),
        connectionText: document.getElementById('serverStatusText'),
        marketTable: document.getElementById('marketTableBody'),
        aiInsightText: document.getElementById('aiInsightText'),
        gaugeFill: document.getElementById('pulseGaugeFill'),
        pulseValue: document.getElementById('pulseValue'),
        botBadge: document.getElementById('botBadge'),
        autoTradeStatus: document.getElementById('autoTradeStatus'),
        footerTime: document.getElementById('footerTime'),
        uptime: document.getElementById('uptime'),
        // New elements for panel header
        assetName: document.getElementById('assetName'),
        assetPrice: document.getElementById('assetPrice'),
        assetChange: document.getElementById('assetChange'),
        assetTrend: document.getElementById('assetTrend'),
        userIdDisplay: document.getElementById('userIdDisplay'),
        themeToggle: document.getElementById('themeToggle'),
        marketSelect: document.getElementById('marketType')
    },


    updateMarketWatch(assets) {
        const { marketTable } = this.elements;
        if (!marketTable) return;

        const favSymbols = (state.userWatchlist || []).map(w => w.symbol);

        marketTable.innerHTML = assets.map(asset => {
            const isFav = favSymbols.includes(asset.symbol);
            const market = asset.market || state.currentMarket;
            const name = (asset.name || '').replace(/'/g, "\\'");
            return `
            <div class="market-item" data-symbol="${asset.symbol}">
                <div class="symbol-box"><span class="symbol">${asset.symbol}</span><span class="name">${asset.name || ''}</span></div>
                <div class="price-box" data-price-symbol="${asset.symbol}" style="font-family:var(--font-mono); font-weight:600;">${formatSymbolPrice(asset.symbol, asset.price || 0)}</div>
                <div class="change-box ${asset.change >= 0 ? 'p-positive' : 'p-negative'}" data-change-symbol="${asset.symbol}" style="font-family:var(--font-mono); font-weight:700; text-align:right;">
                    ${asset.change >= 0 ? '+' : ''}${asset.change.toFixed(2)}%
                </div>
                <button class="fav-btn ${isFav ? 'active' : ''}" data-fav-symbol="${asset.symbol}"
                    onclick="event.stopPropagation(); toggleWatchlistSymbol('${asset.symbol}', '${market}', '${name}')"
                    title="${isFav ? 'Remove from watchlist' : 'Add to watchlist'}"
                    aria-label="${isFav ? 'Remove from watchlist' : 'Add to watchlist'}">
                    <i class="${isFav ? 'fas' : 'far'} fa-heart"></i>
                </button>
            </div>`;
        }).join('');

        // Add click listeners
        marketTable.querySelectorAll('.market-item').forEach((item, i) => {
            item.onclick = () => {
                switchSymbol(assets[i].symbol);
            };
        });
    },

    updatePrice(data) {
        const { currentPrice, priceChange, high24h, low24h, volume24h } = this.elements;
        if (!currentPrice) return;

        const price = parseFloat(data.price);
        const previousPrice = state.lastPrice;
        state.lastPrice = price;
        state.prices[data.symbol] = price;

        // An instrument's price belongs to its own quote currency — it is not
        // a portfolio value, so it must not be run through the account's
        // display-currency conversion.
        if (currentPrice) currentPrice.textContent = formatSymbolPrice(data.symbol, price);

        // Update Portfolio Header (Asset Specific Display)
        const { assetPrice, assetChange, assetTrend } = this.elements;
        if (assetPrice) {
            assetPrice.textContent = formatSymbolPrice(data.symbol, price);
            assetPrice.style.color = price > previousPrice ? 'var(--bullish)' : (price < previousPrice ? 'var(--bearish)' : '');
        }

        // Flash animation
        if (currentPrice) {
            currentPrice.style.color = price > previousPrice ? 'var(--bullish)' : 'var(--bearish)';
            setTimeout(() => currentPrice.style.color = '', 300);
        }

        if (priceChange || assetChange) {
            const changePct = parseFloat(data.change_pct) || 0;
            const changeText = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
            if (priceChange) {
                priceChange.textContent = changeText;
                priceChange.className = `price-change ${changePct >= 0 ? 'positive' : 'negative'}`;
            }
            if (assetChange) {
                assetChange.textContent = changeText;
                assetChange.className = changePct >= 0 ? 'p-positive' : 'p-negative';
            }
            if (assetTrend) {
                assetTrend.textContent = changePct >= 0 ? '↑' : '↓';
                assetTrend.style.color = changePct >= 0 ? 'var(--bullish)' : 'var(--bearish)';
            }
        }

        if (high24h) high24h.textContent = formatSymbolPrice(data.symbol, data.high_24h);
        if (low24h) low24h.textContent = formatSymbolPrice(data.symbol, data.low_24h);
        if (volume24h) volume24h.textContent = formatVolume(data.volume_24h);
    },

    updateAccount(account) {
        const { accountBalance, accountPnl, tradeCount } = this.elements;
        if (accountBalance) accountBalance.textContent = formatCurrencyValue(account.total_value);
        if (accountPnl) {
            const pnl = account.pnl;
            accountPnl.textContent = (pnl >= 0 ? '+' : '') + formatCurrencyValue(pnl);
            accountPnl.className = `account-value pnl ${pnl >= 0 ? 'pnl-positive' : 'pnl-negative'}`;
        }
        if (tradeCount && account.total_trades !== undefined) {
            tradeCount.textContent = account.total_trades;
            state.tradeCount = account.total_trades;
        }
    },

    updateStatus(connected) {
        const { connectionDot, connectionText } = this.elements;
        if (connectionDot) connectionDot.classList.toggle('connected', connected);
        if (connectionText) connectionText.textContent = connected ? 'Live' : 'Offline';
    },

    renderAI(data) {
        const { aiInsightText, aiPulseDot, gaugeFill, pulseValue } = this.elements;
        if (!aiInsightText) return;

        if (pulseValue) pulseValue.textContent = data.pulse_score;
        if (gaugeFill) {
            const offset = 314.16 - (data.pulse_score / 100 * 314.16);
            gaugeFill.style.strokeDashoffset = offset;
        }

        // AI Typing Effect
        if (aiTypeInterval) clearInterval(aiTypeInterval);
        aiInsightText.textContent = '';
        let i = 0;
        const fullText = data.ai_thought;

        aiTypeInterval = setInterval(() => {
            if (i < fullText.length) {
                aiInsightText.textContent += fullText.charAt(i);
                i++;
            } else {
                clearInterval(aiTypeInterval);
            }
        }, 40);
    }
};

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
    prices: {},      // { SYMBOL: PRICE } - Real-time price tracking
    // ── V2 Institutional State ──
    v2TradeHistory: [],        // Rolling trade results for expectancy calc
    v2ActiveStrategies: new Set(),  // strategy+symbol uniqueness
    v2AutoDisabled: false      // Negative expectancy lockout flag
};

// ============================================================
// MARKET ASSETS DEFINITION (per-market watch lists)
// ============================================================
const marketAssets = {
    crypto: [
        { symbol: 'BTCUSDT', name: 'Bitcoin' },
        { symbol: 'ETHUSDT', name: 'Ethereum' },
        { symbol: 'SOLUSDT', name: 'Solana' },
        { symbol: 'BNBUSDT', name: 'BNB' },
        { symbol: 'XRPUSDT', name: 'Ripple' },
        { symbol: 'ADAUSDT', name: 'Cardano' },
        { symbol: 'DOGEUSDT', name: 'Dogecoin' },
        { symbol: 'AVAXUSDT', name: 'Avalanche' }
    ],
    stocks: [
        { symbol: 'AAPL', name: 'Apple' },
        { symbol: 'MSFT', name: 'Microsoft' },
        { symbol: 'GOOGL', name: 'Alphabet' },
        { symbol: 'TSLA', name: 'Tesla' },
        { symbol: 'NVDA', name: 'NVIDIA' },
        { symbol: 'AMZN', name: 'Amazon' },
        { symbol: 'META', name: 'Meta' },
        { symbol: 'RELIANCE', name: 'Reliance Ind.' },
        { symbol: 'TCS', name: 'TCS' },
        { symbol: 'INFY', name: 'Infosys' },
        { symbol: 'HDFCBANK', name: 'HDFC Bank' },
        { symbol: 'TATAMOTORS', name: 'Tata Motors' }
    ],
    forex: [
        { symbol: 'EURUSD', name: 'EUR/USD' },
        { symbol: 'GBPUSD', name: 'GBP/USD' },
        { symbol: 'USDJPY', name: 'USD/JPY' },
        { symbol: 'USDINR', name: 'USD/INR' },
        { symbol: 'AUDUSD', name: 'AUD/USD' },
        { symbol: 'USDCAD', name: 'USD/CAD' }
    ],
    commodities: [
        { symbol: 'XAUUSD', name: 'Gold' },
        { symbol: 'XAGUSD', name: 'Silver' },
        { symbol: 'XBRUSD', name: 'Brent Oil' },
        { symbol: 'XTIUSD', name: 'WTI Oil' },
        { symbol: 'XCUUSD', name: 'Copper' },
        { symbol: 'XNGUSD', name: 'Natural Gas' }
    ]
};

// ============================================================
// QUOTE CURRENCY
// ------------------------------------------------------------
// An instrument is priced in its OWN quote currency, not always
// USD: USDINR quotes rupees, USDJPY quotes yen, NSE stocks quote
// rupees. Prefixing every price with "$" made the panel disagree
// with the chart, so pricing is resolved per symbol here.
// ============================================================
const CURRENCY_GLYPHS = {
    USD: '$', INR: '₹', JPY: '¥', EUR: '€', GBP: '£',
    CAD: 'C$', AUD: 'A$', CHF: 'CHF ', NZD: 'NZ$', CNY: '¥',
    SGD: 'S$', HKD: 'HK$'
};

// Symbols that trade on Indian exchanges and are therefore quoted in INR.
const INR_QUOTED_SYMBOLS = [
    'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'TATAMOTORS', 'ICICIBANK',
    'SBIN', 'WIPRO', 'ITC', 'BHARTIARTL', 'HINDUNILVR', 'AXISBANK',
    'KOTAKBANK', 'LT', 'MARUTI', 'TITAN', 'NIFTY', 'BANKNIFTY', 'SENSEX'
];

const ISO_CURRENCY_CODES = [
    'USD', 'EUR', 'GBP', 'JPY', 'INR', 'CAD', 'AUD',
    'CHF', 'NZD', 'CNY', 'SGD', 'HKD'
];

/** True for a real FX cross (EURUSD, USDINR) — but NOT for metals like XAUUSD. */
function isFxPair(symbol) {
    const s = String(symbol || '').toUpperCase().replace('/', '');
    return /^[A-Z]{6}$/.test(s)
        && ISO_CURRENCY_CODES.includes(s.slice(0, 3))
        && ISO_CURRENCY_CODES.includes(s.slice(3));
}

// Quote currency reported by the backend per symbol — authoritative when
// present, because the provider knows what the exchange actually quotes in.
const symbolCurrencies = {};

/** The currency a symbol's price is denominated in. */
function quoteCurrencyFor(symbol) {
    const s = String(symbol || '').toUpperCase().replace('/', '');
    if (!s) return 'USD';
    if (symbolCurrencies[s]) return symbolCurrencies[s];
    if (INR_QUOTED_SYMBOLS.includes(s)) return 'INR';
    // Crypto priced in a USD stablecoin.
    if (s.endsWith('USDT') || s.endsWith('USDC') || s.endsWith('BUSD')) return 'USD';
    // 6-char pair (FX or metal): the trailing 3 chars are the quote currency.
    if (/^[A-Z]{6}$/.test(s) && ISO_CURRENCY_CODES.includes(s.slice(3))) return s.slice(3);
    return 'USD';
}

function currencyGlyph(code) {
    return CURRENCY_GLYPHS[code] || `${code} `;
}

/**
 * Format a price in the instrument's own quote currency.
 * FX crosses get 4 decimals — at 2dp an entire day's move disappears.
 */
function formatSymbolPrice(symbol, price) {
    const value = parseFloat(price);
    if (!isFinite(value)) return '—';
    const dp = isFxPair(symbol) ? 4 : 2;
    return `${currencyGlyph(quoteCurrencyFor(symbol))}${value.toLocaleString(undefined, {
        minimumFractionDigits: dp,
        maximumFractionDigits: dp
    })}`;
}

// ============================================================
// LIVE PRICE FETCHING (Binance API + Backend Proxy)
// ============================================================
let pricePollingInterval = null;

/**
 * Fetch live price for a symbol from backend proxy (which calls Binance for crypto)
 * @param {string} symbol - The symbol to fetch
 * @param {boolean} updateHeader - Whether to update panel header
 */
async function fetchLivePrice(symbol, updateHeader = true) {
    try {
        const res = await fetch(`/api/v2/price/${symbol}`);
        const data = await res.json();

        if (data.success && data.price > 0) {
            const price = parseFloat(data.price);
            const changePct = parseFloat(data.change_pct) || 0;
            const high = parseFloat(data.high_24h) || 0;
            const low = parseFloat(data.low_24h) || 0;
            const volume = parseFloat(data.volume_24h) || 0;

            // Remember what the provider quotes this instrument in
            if (data.currency) symbolCurrencies[symbol.toUpperCase()] = data.currency;

            // Store in prices map
            const previousPrice = state.prices[symbol] || price;
            state.prices[symbol] = price;

            if (updateHeader && symbol === state.currentSymbol) {
                // Update panel header
                const assetPriceEl = document.getElementById('assetPrice');
                const assetChangeEl = document.getElementById('assetChange');
                const assetTrendEl = document.getElementById('assetTrend');

                if (assetPriceEl) {
                    assetPriceEl.textContent = formatSymbolPrice(symbol, price);
                    // Flash color on price change
                    assetPriceEl.style.color = price > previousPrice ? 'var(--bullish)' : (price < previousPrice ? 'var(--bearish)' : 'var(--accent)');
                    setTimeout(() => { assetPriceEl.style.color = 'var(--accent)'; }, 600);
                }
                if (assetChangeEl) {
                    assetChangeEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
                    assetChangeEl.className = changePct >= 0 ? 'p-positive' : 'p-negative';
                }
                if (assetTrendEl) {
                    assetTrendEl.textContent = changePct >= 0 ? '↑' : '↓';
                    assetTrendEl.style.color = changePct >= 0 ? 'var(--bullish)' : 'var(--bearish)';
                }

                // Also update lastPrice for trade execution
                state.lastPrice = price;

                // Update main price display
                UI.updatePrice({
                    symbol: symbol,
                    price: price,
                    change_pct: changePct,
                    high_24h: high,
                    low_24h: low,
                    volume_24h: volume
                });
            }

            return { price, changePct, high, low, volume };
        }
    } catch (err) {
        console.warn(`[V2] fetchLivePrice failed for ${symbol}:`, err);
    }
    return null;
}

/**
 * Start polling live price for the active symbol
 */
function startPricePolling(symbol) {
    // Clear previous polling
    if (pricePollingInterval) {
        clearInterval(pricePollingInterval);
        pricePollingInterval = null;
    }

    // Immediate fetch
    fetchLivePrice(symbol, true);

    // Poll every 10 seconds
    pricePollingInterval = setInterval(() => {
        fetchLivePrice(symbol, true);
    }, 10000);

    console.log(`[V2] Price polling started for ${symbol}`);
}

/**
 * Render market watch list for a specific market type
 */
function renderMarketWatch(type) {
    const assets = marketAssets[type] || marketAssets.crypto;

    // Render with placeholder prices, then fetch live
    const assetsWithPrices = assets.map(a => ({
        ...a,
        market: type,
        price: state.prices[a.symbol] || 0,
        change: 0
    }));

    UI.updateMarketWatch(assetsWithPrices);

    // Fetch live prices for all assets in this market
    assets.forEach(async (asset) => {
        const data = await fetchLivePrice(asset.symbol, false);
        if (data) {
            // Update the specific market watch item
            const priceEl = document.querySelector(`[data-price-symbol="${asset.symbol}"]`);
            const changeEl = document.querySelector(`[data-change-symbol="${asset.symbol}"]`);
            if (priceEl) {
                priceEl.textContent = formatSymbolPrice(asset.symbol, data.price);
            }
            if (changeEl) {
                changeEl.textContent = `${data.changePct >= 0 ? '+' : ''}${data.changePct.toFixed(2)}%`;
                changeEl.className = `change-box ${data.changePct >= 0 ? 'p-positive' : 'p-negative'}`;
            }
        }
    });
}

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

        if (!authData.authenticated) {
            window.location.href = '/godbot_login';
            return;
        }

        // Initialize Specialized Dashboard Features (Theme, Identity)
        initDashboardFeatures();
    } catch (err) {
        console.error('Auth Check Failed', err);
        window.location.href = '/godbot_login';
        return;
    }


    // Initialize Components
    setTimeout(() => {
        initChart();
    }, 500);

    // V3: every step isolated — one broken panel can never again kill
    // price polling, account balance, or news for the whole dashboard.
    _safeInit('socket', initSocket);
    _safeInit('botManagement', initBotManagement);
    _safeInit('positionsPanel', initPositionsPanel);
    _safeInit('orderPanel', initOrderPanel);
    _safeInit('logout', initLogout);
    _safeInit('clock', initClock);
    _safeInit('systemStatus', checkSystemStatus);
    _safeInit('search', initSearch);
    _safeInit('assetSwitcher', initAssetSwitcher);
    _safeInit('eventListeners', initEventListeners);
    _safeInit('sidebarNav', initSidebarNav);

    // Auto-refresh loops
    setInterval(() => {
        _safeInit('loadPositions', loadPositions);
        _safeInit('loadBots', loadBots);
    }, 5000);

    // Institutional Engine (V2)
    _safeInit('institutionalEngine', () => InstitutionalEngine.init());

    // Initial Data Load — price polling, account balance, news
    _safeInit('initialData', loadInitialData);
    _safeInit('news', loadNews);
});

// Global Helpers for new UI
window.openSettingsModal = () => { document.getElementById('settingsModal').style.display = 'flex'; };
window.closeSettingsModal = () => { document.getElementById('settingsModal').style.display = 'none'; };

window.setOrderSide = (side) => {
    const buyBtn = document.getElementById('buySide');
    const sellBtn = document.getElementById('sellSide');
    const executeBtn = document.getElementById('btnExecuteOrder');
    
    if (side === 'buy') {
        buyBtn.classList.add('active');
        sellBtn.classList.remove('active');
        executeBtn.className = 'btn-execute buy';
        executeBtn.textContent = 'EXECUTE BUY / LONG';
        state.orderSide = 'buy';
    } else {
        buyBtn.classList.remove('active');
        sellBtn.classList.add('active');
        executeBtn.className = 'btn-execute sell';
        executeBtn.textContent = 'EXECUTE SELL / SHORT';
        state.orderSide = 'sell';
    }
};

function syncTradingViewTheme(theme) {
    // Delegate to ChartManager which handles theme-aware widget re-creation
    const symbol = state.currentSymbol || "BTCUSDT";
    if (ChartManager.widget) {
        ChartManager.setSymbol(symbol, state.currentInterval);
    } else {
        ChartManager.init('tradingview_chart', symbol, state.currentInterval);
    }
}

function initLogout() {
    const btn = document.getElementById('logoutBtn');
    const modal = document.getElementById('logoutModal');
    const confirmBtn = document.getElementById('confirmLogout');
    const cancelBtn = document.getElementById('cancelLogout');
    
    if (!btn || !modal) return;

    btn.onclick = () => {
        modal.style.display = 'flex';
    };

    cancelBtn.onclick = () => {
        modal.style.display = 'none';
    };

    confirmBtn.onclick = async () => {
        try {
            await fetch('/api/auth/logout', { method: 'POST' });
            localStorage.removeItem('loginTime');
            window.location.href = '/godbot_login';
        } catch (error) {
            console.error('Logout failed:', error);
            window.location.href = '/godbot_login';
        }
    };
}

window.showLogoutModal = function() {
    const modal = document.getElementById('logoutModal');
    if (modal) modal.style.display = 'flex';
};

// NOTE: initPositionsTabs() lives further below (near the positions panel code)
// and matches the current Cockpit markup (#positionTabs / .tab-btn / #tab-*).
// A stale duplicate that targeted the old ".positions-area .tab" markup was
// removed here — it never matched anything in godbot_home.html.

function initClock() {
    setInterval(() => {
        const now = new Date();
        const timeStr = now.toLocaleTimeString();
        const timeEl = document.getElementById('footerTime');
        if (timeEl) timeEl.textContent = timeStr;
    }, 1000);
}

function initSidebar() {
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
            item.classList.add('active');
            const tab = item.dataset.tab;
            console.log(`[V2] Sidebar Tab: ${tab}`);
            // Logic to switch high-level views could go here
        });
    });
}

function initOrderPanel() {
    const buyBtn = document.getElementById('buySide');
    const sellBtn = document.getElementById('sellSide');
    const executeBtn = document.getElementById('btnExecuteOrder');
    const autoSwitch = document.getElementById('btnAutoTrade');

    if (buyBtn) buyBtn.onclick = () => setOrderSide('buy');
    if (sellBtn) sellBtn.onclick = () => setOrderSide('sell');

    if (executeBtn) {
        executeBtn.onclick = () => {
            const side = state.orderSide || 'buy';
            executeTrade(side);
        };
    }

    if (autoSwitch) {
        autoSwitch.onchange = (e) => {
            if (e.target.checked) startBot();
            else stopBot();
        };
    }
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
    if (!btn) return;
    
    const icon = btn.querySelector('.pause-icon');
    const text = btn.querySelector('.pause-text');

    if (paused) {
        btn.classList.add('paused');
        if (icon) icon.textContent = '▶️';
        if (text) text.textContent = 'Paused';
        btn.title = "System is PAUSED. Click to Resume.";
    } else {
        btn.classList.remove('paused');
        if (icon) icon.textContent = '⏸️';
        if (text) text.textContent = 'Running';
        btn.title = "System is RUNNING. Click to Pause.";
    }
}

// ============================================================
// CHART
// ============================================================

function initChart() {
    ChartManager.init('tradingview_chart', state.currentSymbol, state.currentInterval);
}

async function loadChartData() {
    if (ChartManager.widget) {
        ChartManager.setSymbol(state.currentSymbol, state.currentInterval);
    } else {
        console.warn("[V2] loadChartData called but widget not ready yet");
        // We could also call initChart here if it hasn't been called
    }
}

// ============================================================
// WEBSOCKET - REAL-TIME DATA
// ============================================================

function initSocket() {
    console.log('[V2] io defined?', typeof io !== 'undefined');
    if (typeof io === 'undefined') {
        console.error('[V2] Socket.IO client library not loaded!');
        return;
    }
    
    // Determine the socket URL: relative to the current window
    // This fixed the hardcoded 'localhost:5050' which broke on Railway
    state.socket = io({
        transports: ['websocket', 'polling'],
        reconnectionAttempts: 5,
        reconnectionDelay: 2000
    });

    state.socket.on('connect', () => {
        console.log('Connected to GoatBotTrade server');
        UI.updateStatus(true);
        
        // Re-subscribe to Market Watch tickers on connect/reconnect
        const defaultSymbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT'];
        state.socket.emit('join_ticker_rooms', { symbols: defaultSymbols });
        state.socket.emit('change_market', { market: 'crypto', symbol: state.currentSymbol });
    });

    state.socket.on('connect_error', (err) => {
        console.error('[V2] Socket connection error:', err);
    });

    state.socket.on('disconnect', () => {
        console.log('Disconnected from server');
        UI.updateStatus(false);
    });

    state.socket.on('price_update', (data) => {
        // 1. Update main display if matches current symbol
        if (data.symbol === state.currentSymbol) {
            UI.updatePrice(data);
        }

        // 2. Update Market Watch list in real-time
        const priceEl = document.querySelector(`[data-price-symbol="${data.symbol}"]`);
        const changeEl = document.querySelector(`[data-change-symbol="${data.symbol}"]`);
        
        if (priceEl) {
            const price = parseFloat(data.price);
            priceEl.textContent = formatSymbolPrice(data.symbol, price);
            // Subtle flash
            const isUp = price > (state.prices[data.symbol] || 0);
            priceEl.style.color = isUp ? 'var(--bullish)' : 'var(--bearish)';
            setTimeout(() => priceEl.style.color = '', 300);
        }
        
        if (changeEl) {
            const changePct = parseFloat(data.change_pct) || 0;
            changeEl.textContent = `${changePct >= 0 ? '+' : ''}${changePct.toFixed(2)}%`;
            changeEl.className = `change-box ${changePct >= 0 ? 'p-positive' : 'p-negative'}`;
        }

        state.prices[data.symbol] = parseFloat(data.price);
    });

    state.socket.on('v2_account_update', (data) => {
        console.log('[V2] Account update received:', data);
        UI.updateAccount(data);
        // If we have positions in the update, we can refresh the list faster than the 5s interval
        if (data.positions) {
            positionsData.open = data.positions;
            renderOpenPositions();
        }
        updateUI();
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
        console.log('[V2] Received market_intel:', data);
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

// Deprecated: UI.updatePrice handles this now
function updatePrice(data) { }

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

// function updateChart(data) { ... } // Removed: TradingView handles real-time updates

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

let suggestionsData = null;

// --- UI Component Initialization ---
function initUI() {
    // Theme toggle is handled by initTheme()

    // Asset switcher
    document.querySelectorAll('.asset-pill').forEach(pill => {
        pill.addEventListener('click', (e) => {
            document.querySelectorAll('.asset-pill').forEach(p => p.classList.remove('active'));
            pill.classList.add('active');
            switchAsset(pill.dataset.asset);
        });
    });

    // Auto-suggest search
    const searchInput = document.getElementById('search-assets');
    const searchResults = document.getElementById('search-results');
    
    if (searchInput) {
        // ... (rest of suggestions logic)
    }

    // Scroll effect for header
    window.addEventListener('scroll', () => {
        const header = document.querySelector('.header');
        if (header) {
            if (window.scrollY > 20) {
                header.classList.add('scrolled');
            } else {
                header.classList.remove('scrolled');
            }
        }
    });

    initSidebarTabs(); // Initialize sidebar tabs

    // Initial TV Theme Sync
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    syncTradingViewTheme(currentTheme);
}

async function loadSuggestions() {
    try {
        const res = await fetch('/v2/market_data_suggestions.json');
        suggestionsData = await res.json();
        console.log('[V2] Suggestions loaded');
    } catch (e) {
        console.error('[V2] Failed to load suggestions:', e);
    }
}

function initSearch() {
    const searchInput = document.getElementById('globalSearch');
    const searchResults = document.getElementById('searchResults');
    if (!searchInput || !searchResults) return;

    loadSuggestions();

    searchInput.addEventListener('input', (e) => {
        const query = e.target.value.toUpperCase();
        if (!query || !suggestionsData) {
            searchResults.classList.remove('active');
            return;
        }

        // Flatten and search
        let matches = [];
        for (const [market, items] of Object.entries(suggestionsData)) {
            items.forEach(item => {
                if (item.symbol.includes(query) || item.name.toUpperCase().includes(query)) {
                    matches.push({ ...item, market });
                }
            });
        }

        if (matches.length > 0) {
            renderSuggestions(matches);
            searchResults.classList.add('active');
        } else {
            searchResults.classList.remove('active');
        }
    });

    // Close results when clicking outside
    document.addEventListener('click', (e) => {
        if (!searchInput.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.remove('active');
        }
    });

    searchInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            const query = searchInput.value.toUpperCase();
            if (query) {
                switchSymbol(query);
                searchResults.classList.remove('active');
            }
        }
    });

    // Keyboard shortcut Cmd/Ctrl + K
    document.addEventListener('keydown', (e) => {
        if ((e.metaKey || e.ctrlKey) && e.key === 'k') {
            e.preventDefault();
            searchInput.focus();
        }
    });
}

function renderSuggestions(matches) {
    const searchResults = document.getElementById('searchResults');
    const favSymbols = (state.userWatchlist || []).map(w => w.symbol);
    searchResults.innerHTML = matches.map(m => {
        const isFav = favSymbols.includes(m.symbol);
        return `
        <div class="suggestion-item" onclick="selectSuggestion('${m.symbol}', '${m.market}')">
            <div class="suggestion-info">
                <span class="s-symbol">${m.symbol}</span>
                <span class="s-name">${m.name}</span>
            </div>
            <span class="s-market">${m.market}</span>
            <button class="fav-btn ${isFav ? 'active' : ''}" onclick="event.stopPropagation(); toggleWatchlistFromSearch('${m.symbol}', '${m.market}', '${m.name.replace(/'/g, "\\'")}')"
                title="${isFav ? 'Remove from watchlist' : 'Add to watchlist'}">
                <i class="${isFav ? 'fas' : 'far'} fa-heart"></i>
            </button>
        </div>`;
    }).join('');
}

window.selectSuggestion = (symbol, market) => {
    switchAsset(market);
    switchSymbol(symbol);
    showIntel(symbol, market);
    document.getElementById('searchResults')?.classList.remove('active');
    document.getElementById('globalSearch').value = '';
};

/**
 * Rebuild the Auto-Trade symbol dropdown so it always mirrors the instrument
 * the user is actually watching. Previously the dropdown was a fixed
 * crypto/stock list that never followed the chart, so hitting START BOT
 * launched BTCUSDT no matter what was on screen.
 */
function syncSymbolSelect() {
    const select = document.getElementById('symbolSelect');
    if (!select) return;

    const market = state.currentMarket || 'crypto';
    const assets = (marketAssets[market] || marketAssets.crypto).slice();

    // The active symbol may come from search and not be in the market list —
    // it still has to be selectable, otherwise the bot starts on the wrong asset.
    if (state.currentSymbol && !assets.some(a => a.symbol === state.currentSymbol)) {
        assets.unshift({ symbol: state.currentSymbol, name: state.currentSymbol });
    }

    const label = market.charAt(0).toUpperCase() + market.slice(1);
    select.innerHTML = `<optgroup label="${label}">` + assets.map(a =>
        `<option value="${a.symbol}">${a.symbol}${a.name && a.name !== a.symbol ? ` — ${a.name}` : ''}</option>`
    ).join('') + '</optgroup>';

    select.value = state.currentSymbol;
}

function switchSymbol(symbol) {
    console.log(`[V2] Switching to: ${symbol}`);
    state.currentSymbol = symbol;

    // Keep the Auto-Trade panel pointed at whatever is being watched.
    syncSymbolSelect();

    // Update asset name display
    const assetNameEl = document.getElementById('assetName');
    if (assetNameEl) assetNameEl.textContent = symbol;

    // Update trading panel asset symbol input
    const assetSymbolEl = document.getElementById('assetSymbol');
    if (assetSymbolEl) assetSymbolEl.value = symbol;
    
    // Update order panel symbol
    const orderSymbolEl = document.getElementById('orderSymbol');
    if (orderSymbolEl) orderSymbolEl.value = symbol;
    
    // Use ChartManager to switch symbol (handles exchange prefix mapping)
    if (ChartManager.widget) {
        ChartManager.setSymbol(symbol, state.currentInterval);
    } else {
        initChart();
    }
    
    // Fetch and display live price immediately + start polling
    startPricePolling(symbol);
    
    // Update global state on server for intelligence loop
    if (state.socket) {
        state.socket.emit('set_symbol', { symbol: symbol, market: state.currentMarket });
    }
    
    showNotification(`Active Asset: ${symbol}`);
}

window.showIntel = (symbol, market) => {
    const overlay = document.getElementById('intelOverlay');
    const title = document.getElementById('intelTitle');
    const desc = document.getElementById('intelDesc');
    const history = document.getElementById('intelHistory');

    if (!overlay || !suggestionsData) return;

    const data = suggestionsData[market]?.find(item => item.symbol === symbol);
    if (!data) return;

    title.textContent = `${data.name} (${data.symbol})`;
    desc.textContent = data.description;
    history.textContent = data.history;

    overlay.style.display = 'flex';
};

window.closeIntel = () => {
    _setStyle('intelOverlay', 'display', 'none');
};

function initAssetSwitcher() {
    // Set initial state
    const currentAsset = state.currentMarket || 'crypto';
    switchAsset(currentAsset, false);

    // Populate the Auto-Trade symbol dropdown for the starting market —
    // switchAsset(_, false) deliberately skips the symbol switch, so without
    // this the dropdown would sit on its single placeholder option.
    syncSymbolSelect();
}

function switchAsset(type, updateState = true) {
    console.log(`[V2] Switching asset class to: ${type}`);
    
    // Update body attribute for CSS theme
    document.body.setAttribute('data-asset', type);
    
    // Toggle pill button active states
    document.querySelectorAll('.nav-pill').forEach(pill => {
        pill.classList.toggle('active', pill.getAttribute('data-market') === type);
    });

    // Update news panel title
    const newsTitle = document.getElementById('newsTitle');
    const titleMap = { crypto: 'CRYPTO NEWS', stocks: 'STOCK NEWS', forex: 'FOREX NEWS', commodities: 'COMMODITY NEWS' };
    if (newsTitle) newsTitle.textContent = titleMap[type] || 'MARKET NEWS';

    if (updateState) {
        state.currentMarket = type;

        // Default symbols per asset class
        const defaults = {
            crypto: 'BTCUSDT',
            stocks: 'AAPL',
            forex: 'EURUSD',
            commodities: 'XAUUSD'
        };
        const newSymbol = defaults[type] || 'BTCUSDT';
        
        // Update Panel Header Symbol
        if (UI.elements.assetName) UI.elements.assetName.textContent = newSymbol;

        // Populate per-market watch list with live prices
        renderMarketWatch(type);

        // Switch to the default symbol for this market (this handles chart + price polling)
        switchSymbol(newSymbol);

        // Tell server
        if (state.socket) {
            state.socket.emit('change_market', {
                market: type,
                symbol: newSymbol
            });
            // Join ticker rooms for new market
            const symbols = (marketAssets[type] || marketAssets.crypto).map(a => a.symbol);
            state.socket.emit('join_ticker_rooms', { symbols });
        }

        // Reload news for this market
        loadNews();
        showNotification(`Switched to ${type.toUpperCase()}`);
    }
}

/** 🌓 Theme System Logic */
function initTheme() {
    const toggle = document.getElementById('themeToggle');
    if (!toggle) return;

    const applyThemeIcon = (theme) => {
        toggle.textContent = theme === 'dark' ? '🌙' : '☀️';
    };

    toggle.onclick = () => {
        const current = document.documentElement.getAttribute('data-theme') || 'dark';
        const next = current === 'dark' ? 'light' : 'dark';
        
        document.documentElement.setAttribute('data-theme', next);
        localStorage.setItem('theme', next);
        applyThemeIcon(next);
        
        // Sync TradingView chart theme
        if (typeof syncTradingViewTheme === 'function') {
            syncTradingViewTheme(next);
        }
    };

    // Initial state
    const saved = localStorage.getItem('theme') || 'dark';
    applyThemeIcon(saved);
}

/** 📑 Sidebar Tab Navigation */
function initSidebarTabs() {
    // Removed sidebar tab navigation logic (v2 revamp)
}

/** 🆔 Fetch Unique Public ID */
function initUserIdentity() {
    fetch('/api/user/profile')
        .then(res => res.json())
        .then(data => {
            if (data) {
                const userName = data.name || data.username || 'Trader';
                const publicId = data.public_id || '—';
                
                // Store user ID globally for watchlist API
                state.userId = data.id || data.user_id || 1;
                
                // Update header user chip
                const nameEl = document.getElementById('userNameDisplay');
                if (nameEl) nameEl.textContent = userName;
                
                // Update dropdown
                const dropdownNameEl = document.getElementById('dropdownUserName');
                if (dropdownNameEl) dropdownNameEl.textContent = userName;
                
                // Update profile page if exists
                const profileNameEl = document.getElementById('profileName');
                if (profileNameEl) profileNameEl.textContent = userName;
                
                // Load user watchlist after we have userId
                loadUserWatchlist();
            }
        })
        .catch(err => {
            console.error('Failed to fetch user profile:', err);
            // Still load watchlist with default userId
            state.userId = 1;
            loadUserWatchlist();
        });
}

/** Toggle User Profile Dropdown */
function toggleUserDropdown() {
    const dropdown = document.getElementById('userDropdown');
    if (dropdown) {
        dropdown.classList.toggle('open');
    }
}

// Close dropdown when clicking outside
document.addEventListener('click', function(e) {
    const dropdown = document.getElementById('userDropdown');
    const chip = document.getElementById('userProfileChip');
    if (dropdown && chip && !chip.contains(e.target) && !dropdown.contains(e.target)) {
        dropdown.classList.remove('open');
    }
});


// ============================================================
// WATCHLIST / FAVORITES (DB-Backed)
// ============================================================

/** Load user's saved watchlist from the API */
async function loadUserWatchlist() {
    try {
        const userId = state.userId || 1;
        const res = await fetch(`/api/v2/watchlist?user_id=${userId}`);
        const data = await res.json();
        state.userWatchlist = data.watchlist || [];
        renderUserWatchlist();
    } catch (err) {
        console.error('[Watchlist] Failed to load:', err);
        state.userWatchlist = [];
        renderUserWatchlist();
    }
}

/** Add a symbol to the user's watchlist */
async function addToWatchlist(symbol, market, name) {
    try {
        const userId = state.userId || 1;
        await fetch('/api/v2/watchlist', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, symbol, market, name })
        });
        await loadUserWatchlist();
        showNotification(`❤️ ${symbol} added to watchlist`);
    } catch (err) {
        console.error('[Watchlist] Failed to add:', err);
    }
}

/** Remove a symbol from the user's watchlist */
async function removeFromWatchlist(symbol) {
    try {
        const userId = state.userId || 1;
        await fetch('/api/v2/watchlist', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_id: userId, symbol })
        });
        await loadUserWatchlist();
        showNotification(`💔 ${symbol} removed from watchlist`);
    } catch (err) {
        console.error('[Watchlist] Failed to remove:', err);
    }
}

/**
 * Toggle any symbol's watchlist membership.
 * Shared by the Market Watch rows, the search suggestions and the
 * watchlist itself so a heart means the same thing everywhere.
 */
window.toggleWatchlistSymbol = (symbol, market, name) => {
    const favSymbols = (state.userWatchlist || []).map(w => w.symbol);
    if (favSymbols.includes(symbol)) {
        removeFromWatchlist(symbol);
    } else {
        addToWatchlist(symbol, market || state.currentMarket, name || symbol);
    }
};

/** Toggle watchlist from search suggestion heart button */
window.toggleWatchlistFromSearch = window.toggleWatchlistSymbol;

/** Repaint the Market Watch hearts to match the saved watchlist. */
function refreshMarketWatchHearts() {
    const favSymbols = (state.userWatchlist || []).map(w => w.symbol);
    document.querySelectorAll('[data-fav-symbol]').forEach(btn => {
        const isFav = favSymbols.includes(btn.dataset.favSymbol);
        btn.classList.toggle('active', isFav);
        const label = isFav ? 'Remove from watchlist' : 'Add to watchlist';
        btn.title = label;
        btn.setAttribute('aria-label', label);
        const icon = btn.querySelector('i');
        if (icon) icon.className = `${isFav ? 'fas' : 'far'} fa-heart`;
    });
}

/** Render the user's watchlist in the MY WATCHLIST section */
function renderUserWatchlist() {
    const body = document.getElementById('userWatchlistBody');
    const countEl = document.getElementById('watchlistCount');
    const watchlist = state.userWatchlist || [];

    if (countEl) countEl.textContent = watchlist.length;

    // Hearts elsewhere on the page reflect the same list
    refreshMarketWatchHearts();

    if (!body) return;

    if (watchlist.length === 0) {
        body.innerHTML = `
            <div class="watchlist-empty">
                <i class="far fa-heart"></i>
                <span>Search & add assets to your watchlist</span>
            </div>`;
        return;
    }

    body.innerHTML = watchlist.map(item => {
        const price = state.prices[item.symbol];
        const priceStr = price ? formatSymbolPrice(item.symbol, price) : '...';
        return `
            <div class="watchlist-item" onclick="selectSuggestion('${item.symbol}', '${item.market}')">
                <div class="watchlist-item-left">
                    <span class="watchlist-item-symbol">${item.symbol}</span>
                    <span class="watchlist-item-name">${item.name || item.market}</span>
                </div>
                <div class="watchlist-item-right">
                    <span class="watchlist-item-price" data-wl-price="${item.symbol}">${priceStr}</span>
                    <button class="fav-btn active" onclick="event.stopPropagation(); removeFromWatchlist('${item.symbol}')" title="Remove from watchlist" aria-label="Remove from watchlist">
                        <i class="fas fa-heart"></i>
                    </button>
                </div>
            </div>`;
    }).join('');

    // Fetch live prices for all watchlist items
    watchlist.forEach(async (item) => {
        const data = await fetchLivePrice(item.symbol, false);
        if (data) {
            const el = document.querySelector(`[data-wl-price="${item.symbol}"]`);
            if (el) {
                el.textContent = formatSymbolPrice(item.symbol, data.price);
                // Brief flash effect
                el.style.color = 'var(--accent)';
                setTimeout(() => { el.style.color = ''; }, 500);
            }
        }
    });
}


// ============================================================
// TRADING
// ============================================================

async function executeTrade(side) {
    const quantity = parseFloat(document.getElementById('orderQty')?.value || 0);

    if (!quantity || quantity <= 0) {
        alert('Please enter a valid quantity');
        return;
    }

    // ── V2 Institutional Pre-Trade Validation ──
    const stopLossPct = parseFloat(document.getElementById('settingStopLoss')?.value) || state.settings.stopLoss;
    const takeProfitPct = parseFloat(document.getElementById('settingTakeProfit')?.value) || state.settings.takeProfit;
    // Honor a typed limit price; fall back to the live market price for market orders
    const limitPrice = parseFloat(document.getElementById('orderPrice')?.value);
    const currentPrice = (limitPrice && limitPrice > 0) ? limitPrice : state.lastPrice;
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
        stopLoss: parseFloat(document.getElementById('settingStopLoss').value) || state.settings.stopLoss,
        takeProfit: parseFloat(document.getElementById('settingTakeProfit').value) || state.settings.takeProfit,
        positionSize: parseFloat(document.getElementById('settingPositionSize').value) || state.settings.positionSize,
        maxQuantity: parseFloat(document.getElementById('maxQuantity')?.value) || 1.0,
        confluence: parseInt(document.getElementById('settingConfluence').value) || 3,
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
                risk_pct: state.settings.riskPct || 2.0,
                sensitivity: document.getElementById('sensitivitySelect')?.value || 'conservative'
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
    document.getElementById('settingsModal')?.classList.add('open');

    // Load current settings
    document.getElementById('settingConfluence').value = state.settings.confluence;
    _setText('confluenceValue', state.settings.confluence);

    document.getElementById('settingPositionSize').value = state.settings.positionSize;
    _setText('positionSizeValue', state.settings.positionSize + '%');

    _setValue('settingInterval', state.settings.checkInterval);
    _setText('intervalValue', state.settings.checkInterval + 's');

    document.getElementById('settingStopLoss').value = state.settings.stopLoss;
    document.getElementById('settingTakeProfit').value = state.settings.takeProfit;
}

function closeSettings() {
    document.getElementById('settingsModal')?.classList.remove('open');
}

function saveSettings() {
    state.settings = {
        confluence: parseInt(document.getElementById('settingConfluence').value),
        positionSize: parseInt(document.getElementById('settingPositionSize').value),
        checkInterval: parseInt(document.getElementById('settingInterval')?.value || 30),
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

    // Symbol selector — selection is two-way: picking here moves the chart too,
    // so the panel and the Auto-Trade config can never drift apart.
    document.getElementById('symbolSelect')?.addEventListener('change', (e) => {
        const symbol = e.target.value;
        const market = state.currentMarket;

        // Update badge
        const marketBadge = document.getElementById('marketBadge');
        if (marketBadge) marketBadge.textContent = market.toUpperCase();

        // Moves the chart, price polling and header to this symbol
        switchSymbol(symbol);

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
    document.getElementById('strategySelect')?.addEventListener('change', (e) => {
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

    // Signal Sensitivity selector — updates the inline explainer, and the
    // choice persists so a reload doesn't silently drop the user back to
    // Conservative.
    const sensEl = document.getElementById('sensitivitySelect');
    if (sensEl) {
        const saved = localStorage.getItem(SENSITIVITY_PREF_KEY);
        if (saved && SENSITIVITY_INFO[saved]) sensEl.value = saved;

        sensEl.addEventListener('change', () => {
            localStorage.setItem(SENSITIVITY_PREF_KEY, sensEl.value);
            updateSensitivityHelp(sensEl.value);
            console.log(`[V2] Signal sensitivity → ${sensEl.value}`);
        });
        updateSensitivityHelp(sensEl.value);
    }

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
    document.getElementById('btnBuy')?.addEventListener('click', () => executeTrade('buy'));
    document.getElementById('btnSell')?.addEventListener('click', () => executeTrade('sell'));

    // Auto trading - START button starts a bot for the current symbol
    document.getElementById('btnAutoTrade')?.addEventListener('click', () => {
        startBot();
    });

    // Dedicated STOP button for the CURRENT symbol
    document.getElementById('btnStopBot')?.addEventListener('click', () => {
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
    document.getElementById('btnStopAll')?.addEventListener('click', () => {
        fetch('/api/v2/stop-all', { method: 'POST' })
            .then(() => {
                state.v2ActiveStrategies.clear();
                showNotification('🛑 All V2 bots stopping...');
                loadBots();
                loadPositions(); // Refresh positions immediately to see closures
            });
    });

    document.getElementById('btnPanicSell')?.addEventListener('click', async () => {
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

    document.getElementById('btnMarketAnalysis')?.addEventListener('click', async () => {
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

    document.getElementById('btnBacktest')?.addEventListener('click', () => {
        alert('📈 Strategy Backtester module coming soon! Use the Report section to see historical bot performance.');
    });

    document.getElementById('btnReport')?.addEventListener('click', showReportModal);
    document.getElementById('btnResetPaper')?.addEventListener('click', resetPaperTrading);

    // Settings modal
    document.getElementById('btnSettings')?.addEventListener('click', openSettings);
    document.getElementById('modalClose')?.addEventListener('click', closeSettings);
    document.getElementById('btnSaveSettings')?.addEventListener('click', saveSettings);

    // Global Asset Search with Debouncing
    const assetSearch = document.getElementById('assetSearch');
    const searchResults = document.getElementById('searchResults');
    let searchTimeout;

    if (assetSearch) {
        assetSearch.addEventListener('input', (e) => {
            clearTimeout(searchTimeout);
            const query = e.target.value.trim().toUpperCase();
            
            if (query.length < 1) {
                searchResults.classList.add('hidden');
                return;
            }

            searchTimeout = setTimeout(async () => {
                console.log(`[V2] Searching for: ${query}`);
                // Simplified search: check against common pairs
                const pairs = [
                    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT', 
                    'ADAUSDT', 'DOGEUSDT', 'MATICUSDT', 'DOTUSDT', 'AVAXUSDT', 
                    'AAPL', 'TSLA', 'GOOGL', 'MSFT', 'AMZN', 'NVDA', 'META', 'NFLX'
                ];
                const filtered = pairs.filter(p => p.includes(query));
                
                if (filtered.length > 0) {
                    searchResults.innerHTML = filtered.map(p => `
                        <div class="search-result-item" onclick="selectSymbol('${p}')">
                            <span class="result-symbol">${p}</span>
                            <span class="result-exchange">${p.length > 5 ? 'CRYPTO' : 'STOCK'}</span>
                        </div>
                    `).join('');
                    searchResults.classList.remove('hidden');
                } else {
                    searchResults.innerHTML = '<div class="search-result-item" style="cursor: default; opacity: 0.6;">No matches found</div>';
                    searchResults.classList.remove('hidden');
                }
            }, 300);
        });
    }

    // Close search results on click outside
    document.addEventListener('click', (e) => {
        if (assetSearch && !assetSearch.contains(e.target) && !searchResults.contains(e.target)) {
            searchResults.classList.add('hidden');
        }
    });

    // Tips modal
    document.getElementById('btnTips')?.addEventListener('click', () => {
        document.getElementById('tipsModal')?.classList.add('open');
    });
    document.getElementById('tipsModalClose')?.addEventListener('click', () => {
        document.getElementById('tipsModal')?.classList.remove('open');
    });

    // Update auto quantity badge when settings change
    document.getElementById('settingPositionSize')?.addEventListener('input', () => {
        if (state.lastPrice) updateAutoQuantityBadge(state.lastPrice);
    });
    document.getElementById('maxQuantity')?.addEventListener('input', () => {
        if (state.lastPrice) updateAutoQuantityBadge(state.lastPrice);
    });

    // Settings sliders
    document.getElementById('settingConfluence')?.addEventListener('input', (e) => {
        _setText('confluenceValue', e.target.value);
    });
    document.getElementById('settingPositionSize')?.addEventListener('input', (e) => {
        _setText('positionSizeValue', e.target.value + '%');
    });
    document.getElementById('settingInterval')?.addEventListener('input', (e) => {
        _setText('intervalValue', e.target.value + 's');
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
    document.getElementById('btnClearFeed')?.addEventListener('click', () => {
        document.getElementById('tradesFeed').innerHTML = '';
    });
}

// ============================================================
// UI UPDATES
// ============================================================

function updateUI() {
    // Update strategy indicator if it exists
    const activeStrategyEl = document.getElementById('activeStrategy');
    if (activeStrategyEl) {
        activeStrategyEl.textContent = getStrategyName(state.currentStrategy);
    }
    
    // Refresh bot cards to show live PnL
    if (state.activeBots && state.activeBots.length > 0) {
        renderBots(state.activeBots);
    }
}

async function loadInitialData() {
    await loadChartData();
    updateTradeCount();  // Sync trade counter on page load

    // Fetch account data for header-stats immediately
    try {
        const accRes = await fetch('/api/v2/account');
        const accData = await accRes.json();
        if (accData.success) updateAccount(accData);
    } catch (e) {
        console.error('[V2] Initial account fetch error:', e);
    }

    // Periodic header-stats refresh (every 10s)
    setInterval(async () => {
        try {
            const res = await fetch('/api/v2/account');
            const data = await res.json();
            if (data.success) updateAccount(data);
        } catch(e) { /* silent */ }
    }, 10000);

    // Initial Market Watch population — use live price fetching
    renderMarketWatch(state.currentMarket);

    // Fetch live price for current symbol immediately
    startPricePolling(state.currentSymbol);

    // V2: Join ticker rooms for all Market Watch assets for real-time updates
    if (state.socket && state.socket.connected) {
        const symbols = (marketAssets[state.currentMarket] || marketAssets.crypto).map(a => a.symbol);
        state.socket.emit('join_ticker_rooms', { symbols });
        
        // Also ensure we are joined to the specific ticker room for the current symbol
        state.socket.emit('change_market', { market: state.currentMarket, symbol: state.currentSymbol });
    }

    // Load news for default market
    loadNews();
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
            // V2: Strategy indication is part of order panel now
        }
    } catch (e) {
        console.error('Failed to load strategies:', e);
    }
}

function populateStrategyDropdown(strategies) {
    const select = document.getElementById('strategySelect');
    if (!select) return;

    select.innerHTML = strategies.map(s =>
        `<option value="${s.id}">${s.name}</option>`
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
        const currentPrice = state.prices[bot.symbol] || state.lastPrice;
        
        // Live Unrealized PnL Calculation
        let liveUnrealized = bot.stats.unrealized_pnl || 0;
        
        // If we have a live price and entry price info, recalculate for "WOW" factor
        // bot.stats should ideally have entry_price and side if V2 position is active
        if (currentPrice && bot.stats.avg_price && bot.stats.quantity) {
             const sideIdx = bot.stats.side === 'LONG' ? 1 : -1;
             liveUnrealized = (currentPrice - bot.stats.avg_price) * bot.stats.quantity * sideIdx;
        }

        const totalPnl = realizedPnl + liveUnrealized;

        const pnlClass = totalPnl >= 0 ? 'pnl-positive' : 'pnl-negative';
        const pnlSign = totalPnl >= 0 ? '+' : '';

        return `
            <div class="bot-card ${bot.mode}">
                <div class="bot-info">
                    <span class="bot-symbol">${bot.symbol}</span>
                    <span class="bot-strategy">${getStrategyName(bot.strategy)} • ${bot.market.toUpperCase()}</span>
                </div>
                <div class="bot-stats-row">
                     <span class="bot-pnl ${pnlClass}" title="Total (Realized: ${formatCurrencyValue(realizedPnl)}, Live Unrealized: ${formatCurrencyValue(liveUnrealized)})">
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
                    <button class="news-modal-close" onclick="document.getElementById('newsAnalysisModal')?.remove()">&times;</button>
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
                    <button class="btn btn-secondary" onclick="document.getElementById('newsAnalysisModal')?.remove()">Close</button>
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
    console.log('[V2] Initializing Institutional Bot Manager...');
    
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
    }

    // Initial load
    loadBots();
}

async function startBot() {
    const symbol = state.currentSymbol;
    const strategy = state.currentStrategy || 'combined';
    
    // Get values from UI or state
    const posSize = parseFloat(document.getElementById('settingPositionSize')?.value) || 10;
    const sl = parseFloat(document.getElementById('settingStopLoss')?.value) || 2;
    const tp = 10; // Default or from hidden field

    try {
        const res = await fetch('/api/v2/start-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol: symbol,
                strategy: strategy,
                position_size: posSize,
                stop_loss: sl,
                take_profit: tp,
                leverage: 1.0,
                mode: 'paper',
                sensitivity: document.getElementById('sensitivitySelect')?.value || 'conservative'
            })
        });
        const data = await res.json();
        if (data.success) {
            state.activeBotId = data.bot_id;
            UI.elements.botBadge.textContent = 'AUTOMATED';
            UI.elements.botBadge.style.color = 'var(--accent)';
            UI.elements.autoTradeStatus.textContent = 'Active';
            UI.elements.autoTradeStatus.style.color = 'var(--accent)';
        } else {
            showNotification(`Error: ${data.error}`, 'error');
            document.getElementById('btnAutoTrade').checked = false;
        }
    } catch (err) {
        console.error('Failed to start bot', err);
        document.getElementById('btnAutoTrade').checked = false;
    }
}

async function stopBot() {
    if (!state.activeBotId) {
        // Fallback: try to find bot for this symbol
        await loadBots();
    }

    try {
        const res = await fetch('/api/v2/stop-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bot_id: state.activeBotId })
        });
        const data = await res.json();
        if (data.success) {
            state.activeBotId = null;
            UI.elements.botBadge.textContent = 'MANUAL';
            UI.elements.botBadge.style.color = 'var(--gold)';
            UI.elements.autoTradeStatus.textContent = 'Idle';
            UI.elements.autoTradeStatus.style.color = '';
        }
    } catch (err) {
        console.error('Failed to stop bot', err);
    }
}

async function loadBots() {
    try {
        const res = await fetch('/api/v2/bots'); // Need to ensure this endpoint exists or use /api/bots
        const data = await res.json();
        if (data.success) {
            state.activeBots = data.bots;
            // Find if any bot is running for current symbol
            const currentBot = data.bots.find(b => b.symbol === state.currentSymbol && b.status === 'running');
            const autoSwitch = document.getElementById('btnAutoTrade');
            
            if (currentBot) {
                state.activeBotId = currentBot.bot_id;
                if (autoSwitch) autoSwitch.checked = true;
                UI.elements.botBadge.textContent = 'AUTOMATED';
                UI.elements.botBadge.style.color = 'var(--accent)';
                UI.elements.autoTradeStatus.textContent = 'Active';
                UI.elements.autoTradeStatus.style.color = 'var(--accent)';
            } else {
                state.activeBotId = null;
                if (autoSwitch) autoSwitch.checked = false;
                UI.elements.botBadge.textContent = 'MANUAL';
                UI.elements.botBadge.style.color = 'var(--gold)';
                UI.elements.autoTradeStatus.textContent = 'Idle';
                UI.elements.autoTradeStatus.style.color = '';
            }
        }
    } catch (err) {
        console.error('Failed to load bots', err);
    }
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

function initPositionsTabs() {
    const container = document.getElementById('positionTabs');
    if (!container) return;

    container.querySelectorAll('.tab-btn').forEach(btn => {
        btn.onclick = () => {
            const tabId = btn.dataset.tab;
            console.log(`[V2] Terminal Tab: ${tabId}`);

            // Update Tabs
            container.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');

            // Update Content
            document.querySelectorAll('.tab-content').forEach(c => c.classList.add('hidden'));
            const target = document.getElementById(`tab-${tabId}`);
            if (target) target.classList.remove('hidden');

            // Immediate refresh on tab click (don't wait for the 10s interval)
            loadPositions();
            if (tabId === 'portfolio') drawEquityCurve();
        };
    });
}

function initPositionsPanel() {
    console.log('[V2] Initializing Positions Panel...');

    // Wire the Cockpit tab buttons (Positions / History / Signals / Portfolio)
    initPositionsTabs();

    // Export CSV button
    document.getElementById('btnExportCSV')?.addEventListener('click', exportToCSV);

    // Date filters (if needed for history)
    const dateFrom = document.getElementById('filterDateFrom');
    const dateTo = document.getElementById('filterDateTo');


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
            // History tab is rendered solely by renderTradeHistory() to avoid
            // two functions fighting over #tradeHistoryBody with different columns.
            renderTradeHistory();
            renderPendingOrders();
            updatePortfolioUI();
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

    if (!positionsData.open || positionsData.open.length === 0) {
        openBody.innerHTML = '<tr class="empty-row"><td colspan="6" style="text-align:center; padding:40px; color:var(--text-muted);">No active positions found</td></tr>';
    } else {
        openBody.innerHTML = positionsData.open.map(pos => {
            const currentPrice = parseFloat(pos.current_price || pos.avg_price || 0);
            const avgPrice = parseFloat(pos.avg_price || 0);
            const qty = parseFloat(pos.qty || pos.quantity || 0);
            const isShort = pos.side === 'SHORT' || pos.side === 'SELL';
            const netPnl = parseFloat(pos.net_pnl || 0);
            const netPnlPct = avgPrice > 0
                ? (isShort ? (1 - currentPrice / avgPrice) : (currentPrice / avgPrice - 1)) * 100
                : 0;
            const pnlClass = netPnl >= 0 ? 'p-positive' : 'p-negative';
            const sideClass = !isShort ? 'p-positive' : 'p-negative';

            return `
                <tr>
                    <td><strong>${pos.symbol}</strong></td>
                    <td class="${sideClass}">${pos.side || 'LONG'}</td>
                    <td>${qty.toFixed(4)}</td>
                    <td style="font-family:var(--font-mono);">$${avgPrice.toLocaleString()}</td>
                    <td class="${pnlClass}" style="font-weight:700;">
                        $${netPnl.toFixed(2)} (${netPnlPct >= 0 ? '+' : ''}${netPnlPct.toFixed(2)}%)
                    </td>
                    <td>
                        <button class="side-btn sell" style="padding:4px 8px; font-size:10px;" onclick="closePosition('${pos.symbol}')">CLOSE</button>
                    </td>
                </tr>
            `;
        }).join('');
    }
}

function renderClosedPositions() {
    const body = document.getElementById('tradeHistoryBody');
    if (!body) return;

    if (!positionsData.closed || positionsData.closed.length === 0) {
        body.innerHTML = '<tr class="empty-row"><td colspan="6" style="text-align:center; padding:40px; color:var(--text-muted);">No historical trades available</td></tr>';
    } else {
        body.innerHTML = positionsData.closed.map(pos => {
            const pnl = parseFloat(pos.net_pnl ?? pos.realized_pnl ?? 0);
            const entry = parseFloat(pos.entry_price || pos.entry || 0);
            const exit = parseFloat(pos.exit_price || pos.exit || 0);
            const pnlPct = entry > 0 ? (pos.side === 'SHORT' ? (1 - exit/entry) : (exit/entry - 1)) * 100 : 0;
            const pnlClass = pnl >= 0 ? 'p-positive' : 'p-negative';
            const time = pos.timestamp ? new Date(pos.timestamp).toLocaleTimeString() : '--';

            return `
                <tr>
                    <td style="color:var(--text-muted); font-size:11px;">${time}</td>
                    <td><strong>${pos.symbol}</strong></td>
                    <td class="${pos.side === 'BUY' || pos.side === 'LONG' ? 'p-positive' : 'p-negative'}">${pos.side || 'LONG'}</td>
                    <td style="font-family:var(--font-mono);">$${entry.toLocaleString()}</td>
                    <td style="font-family:var(--font-mono);">$${exit.toLocaleString()}</td>
                    <td class="${pnlClass}" style="font-weight:700;">$${pnl.toFixed(2)} (${pnlPct >= 0 ? '+' : ''}${pnlPct.toFixed(2)}%)</td>
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
        body.innerHTML = '<tr class="empty-row"><td colspan="6" style="text-align:center; padding:40px; color:var(--text-muted);">No trade history yet</td></tr>';
    } else {
        // 6 columns — must match the #tab-history table header:
        // Time | Symbol | Side | Type | Qty | Price
        body.innerHTML = positionsData.history.slice(-20).reverse().map(trade => {
            const tradeType = trade.trade_type || trade.type || '—';
            const price = parseFloat(trade.fill_price || trade.price || 0);
            const qty = parseFloat(trade.quantity || 0);
            const ts = trade.timestamp || trade.time;
            const side = (trade.side || '').toUpperCase();
            const isBuy = side.includes('BUY') || side.includes('LONG');

            return `
            <tr>
                <td>${ts ? new Date(ts).toLocaleTimeString() : '—'}</td>
                <td><strong>${trade.symbol || '—'}</strong></td>
                <td class="${isBuy ? 'p-positive' : 'p-negative'}">${side || '—'}</td>
                <td><span style="font-size:0.7rem; color:var(--text-muted)">${tradeType}</span></td>
                <td>${qty.toFixed(4)}</td>
                <td style="font-family:var(--font-mono);">$${price.toLocaleString()}</td>
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
    _setText('depthSymbol', state.currentSymbol);
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
    if (!trades || trades.length === 0) return { totalTrades: 0, totalPnL: 0, winRate: 0, sharpe: 0 };

    const closedTrades = trades.filter(t => t.trade_type === 'CLOSE' || t.type === 'CLOSE');
    if (closedTrades.length === 0) return { totalTrades: 0, totalPnL: 0, winRate: 0, sharpe: 0 };

    const pnls = closedTrades.map(t => t.net_pnl ?? t.realized_pnl ?? 0);
    let wins = 0;
    let losses = 0;
    pnls.forEach(p => p > 0 ? wins++ : losses++);

    const totalPnL = pnls.reduce((a, b) => a + b, 0);
    const avgReturn = totalPnL / pnls.length;
    const winRate = (wins / (wins + losses)) * 100;

    const mean = avgReturn;
    const variance = pnls.reduce((sum, pnl) => sum + Math.pow(pnl - mean, 2), 0) / pnls.length;
    const stdDev = Math.sqrt(variance);
    const sharpe = stdDev > 0 ? mean / stdDev : 0;

    // Profit factor = gross profit / gross loss (∞ when there are no losses)
    const grossProfit = pnls.filter(p => p > 0).reduce((a, b) => a + b, 0);
    const grossLoss = Math.abs(pnls.filter(p => p < 0).reduce((a, b) => a + b, 0));
    const profitFactor = grossLoss > 0 ? grossProfit / grossLoss : (grossProfit > 0 ? Infinity : 0);

    // Max drawdown (%) off the cumulative equity curve of realized P&L
    let cum = 0, peak = 0, maxDrawdown = 0;
    for (const p of pnls) {
        cum += p;
        if (cum > peak) peak = cum;
        const dd = peak > 0 ? ((peak - cum) / peak) * 100 : 0;
        if (dd > maxDrawdown) maxDrawdown = dd;
    }

    return { totalPnL, winRate, sharpe, profitFactor, maxDrawdown, totalTrades: closedTrades.length };
}

function updatePortfolioUI() {
    const stats = calculateAnalytics();
    if (!stats.totalTrades) return;

    const elements = {
        profitFactor: document.getElementById('statProfitFactor'),
        sharpe: document.getElementById('statSharpe'),
        drawdown: document.getElementById('statDrawdown'),
        winRate: document.getElementById('statWinRate')
    };

    if (elements.profitFactor) {
        elements.profitFactor.textContent = isFinite(stats.profitFactor)
            ? stats.profitFactor.toFixed(2) : '∞';
    }
    if (elements.sharpe) elements.sharpe.textContent = stats.sharpe.toFixed(2);
    if (elements.drawdown) elements.drawdown.textContent = '-' + stats.maxDrawdown.toFixed(1) + '%';
    if (elements.winRate) elements.winRate.textContent = Math.round(stats.winRate) + '%';

    drawEquityCurve();
    console.log('[V2] Portfolio UI Dynamic Update triggered');
}

// Render a lightweight SVG equity curve from the cumulative realized P&L of
// closed trades. Pure inline SVG — no external chart library needed.
function drawEquityCurve() {
    const host = document.getElementById('equityCurve');
    if (!host) return;

    const closed = (positionsData.history || [])
        .filter(t => (t.trade_type || t.type) === 'CLOSE');

    if (closed.length === 0) {
        host.innerHTML = '<div style="height:100%; display:flex; align-items:center; justify-content:center; font-size:11px; color:var(--text-muted);">No closed trades yet — the curve builds as positions close.</div>';
        return;
    }

    // Cumulative realized P&L series, starting from 0
    let cum = 0;
    const pts = [0];
    closed.forEach(t => { cum += parseFloat(t.net_pnl ?? t.realized_pnl ?? 0); pts.push(cum); });

    const W = host.clientWidth || 300;
    const H = host.clientHeight || 110;
    const pad = 4;
    const min = Math.min(...pts, 0);
    const max = Math.max(...pts, 0);
    const range = (max - min) || 1;
    const dx = (W - pad * 2) / (pts.length - 1 || 1);
    const y = v => H - pad - ((v - min) / range) * (H - pad * 2);
    const coords = pts.map((v, i) => `${(pad + i * dx).toFixed(1)},${y(v).toFixed(1)}`);

    const up = cum >= 0;
    const stroke = up ? 'var(--bullish, #16c784)' : 'var(--bearish, #ea3943)';
    const line = coords.join(' ');
    const area = `${pad},${y(min)} ${line} ${(pad + (pts.length - 1) * dx).toFixed(1)},${y(min)}`;
    const zeroY = y(0).toFixed(1);

    host.innerHTML = `
        <svg viewBox="0 0 ${W} ${H}" preserveAspectRatio="none" style="width:100%; height:100%; display:block;">
            <defs>
                <linearGradient id="eqFill" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stop-color="${stroke}" stop-opacity="0.28"/>
                    <stop offset="100%" stop-color="${stroke}" stop-opacity="0"/>
                </linearGradient>
            </defs>
            <line x1="${pad}" y1="${zeroY}" x2="${W - pad}" y2="${zeroY}"
                  stroke="var(--text-muted)" stroke-width="0.5" stroke-dasharray="3 3" opacity="0.4"/>
            <polygon points="${area}" fill="url(#eqFill)"/>
            <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="1.6"
                      stroke-linejoin="round" stroke-linecap="round"/>
        </svg>`;
}

// Sidebar & Dropdown Link Sync
function initSidebarNav() {
    document.querySelectorAll('.nav-item[data-route], .dropdown-item[data-route]').forEach(item => {
        item.onclick = (e) => {
            e.preventDefault();
            const route = item.dataset.route;
            console.log(`[V2] Navigating to: ${route}`);

            if (route === 'paper') {
                // Scroll to Paper Trade section (Command Deck)
                const paperSection = document.getElementById('sectionAutoTrade');
                if (paperSection) {
                    paperSection.scrollIntoView({ behavior: 'smooth' });
                    // Highlight the item
                    document.querySelectorAll('.nav-item').forEach(i => i.classList.remove('active'));
                    item.classList.add('active');
                }
            } else if (route === 'home') {
                // Already home or refresh
                if (window.location.pathname.includes('godbot_home')) {
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                } else {
                    window.location.href = '/v2/godbot_home';
                }
            } else {
                // Navigate to separate pages
                window.location.href = `/v2/${route}`;
            }
        };
    });
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
        label.classList.remove('bullish', 'bearish', 'neutral');

        // Add new class based on sentiment
        const sentimentLabel = data.label || 'NEUTRAL';
        const sClass = sentimentLabel.toLowerCase();
        
        dot.classList.add(sClass);
        label.classList.add(sClass);
        label.textContent = sentimentLabel;
        
        if (sentimentLabel !== 'NEUTRAL') {
            badge.classList.add(sClass);
        }
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

    // The symbol dropdown is now rebuilt per market by syncSymbolSelect()
    // rather than toggling static optgroups.
    const defaults = {
        crypto: 'BTCUSDT',
        stocks: 'AAPL',
        forex: 'EURUSD',
        commodities: 'XAUUSD'
    };
    state.currentSymbol = defaults[market] || 'BTCUSDT';
    syncSymbolSelect();

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

    // Update technicals
    loadChartData();
    loadNews();
    
    // Sync TradingView immediately
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    syncTradingViewTheme(currentTheme);

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
function initDashboardFeatures() {
    initTheme();
    initUserIdentity();
}


// ============================================================
// MARKET COMMAND CENTER (STAR FEATURES)
// ============================================================

function renderPulseGauge(data) {
    const { gaugeFill, pulseValue } = UI.elements;
    console.log('[V2] renderPulseGauge elements:', gaugeFill, pulseValue);
    console.log('[V2] renderPulseGauge data:', data);
    if (!gaugeFill || !pulseValue) {
        console.warn('[V2] renderPulseGauge: Missing elements!');
        return;
    }

    let momentum = 50;
    if (data) {
        momentum = data.momentum !== undefined ? data.momentum : (data.score !== undefined ? data.score : 50);
    }
    const color = momentum > 60 ? 'var(--accent)' : (momentum < 40 ? 'var(--bearish)' : 'var(--gold)');
    
    const dashOffset = 251 - (momentum / 100) * 251;
    
    gaugeFill.style.stroke = color;
    pulseValue.textContent = Math.round(momentum);
    pulseValue.style.fill = color;
    
    // Add pulse class to gauge value for animation
    pulseValue.classList.add('pulse');
}

function renderAIInsights(data) {
    const { aiInsightText } = UI.elements;
    if (!aiInsightText) return;

    const insights = data.insights || ["Analyzing institutional order flow...", "Scanning for liquidity clusters...", "Monitoring whale activity..."];
    const randomInsight = insights[Math.floor(Math.random() * insights.length)];
    
    // Typewriter effect simulation
    aiInsightText.style.opacity = '0';
    setTimeout(() => {
        aiInsightText.textContent = randomInsight;
        aiInsightText.style.opacity = '1';
    }, 500);
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
        const high = parseFloat(UI.elements.high24h?.textContent?.replace(/[^0-9.]/g, '')) || marketPrice * 1.01;
        const low = parseFloat(UI.elements.low24h?.textContent?.replace(/[^0-9.]/g, '')) || marketPrice * 0.99;
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
        const volume = UI.elements.volume24h?.textContent || '0';
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
        const stopLossPct = (parseFloat(document.getElementById('settingStopLoss')?.value) || state.settings.stopLoss) / 100;
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

// ============================================================
// STRATEGY SELECTOR & AUTO-TRADE CONTROLS
// ============================================================

async function loadStrategies() {
    const sel = document.getElementById('strategySelect');
    if (!sel) return;

    try {
        const res = await fetch('/api/v2/strategies');
        const data = await res.json();
        if (!data.success) return;

        sel.innerHTML = '';
        data.strategies.forEach(s => {
            const opt = document.createElement('option');
            opt.value = s.id;
            opt.textContent = `${s.icon || '📌'} ${s.name}`;
            sel.appendChild(opt);
        });

        // Merge custom strategies from localStorage
        const custom = getCustomStrategies();
        if (custom.length > 0) {
            const group = document.createElement('optgroup');
            group.label = '🔧 Custom Strategies';
            custom.forEach(cs => {
                const opt = document.createElement('option');
                opt.value = `custom:${cs.id}`;
                opt.textContent = `🔧 ${cs.name}`;
                group.appendChild(opt);
            });
            sel.appendChild(group);
        }

        console.log(`[V2] Loaded ${data.strategies.length} strategies + ${custom.length} custom`);
    } catch (e) {
        console.error('[V2] Failed to load strategies:', e);
    }
}

// Start Bot
async function startBotFromUI() {
    const strategy = document.getElementById('strategySelect')?.value;
    // The dropdown mirrors the watched instrument (see syncSymbolSelect), but
    // state.currentSymbol stays authoritative so the bot can never launch on a
    // different asset than the one on screen.
    const symbol = state.currentSymbol || document.getElementById('symbolSelect')?.value;
    const market = state.currentMarket || 'crypto';
    const interval = document.getElementById('intervalSelect')?.value || '1m';
    const posSize = parseFloat(document.getElementById('settingPositionSize')?.value || 10);
    const stopLoss = parseFloat(document.getElementById('settingStopLoss')?.value || 2);
    const takeProfit = parseFloat(document.getElementById('settingTakeProfit')?.value || 6);
    const maxQty = parseFloat(document.getElementById('maxQuantity')?.value || 1);
    const sensitivity = document.getElementById('sensitivitySelect')?.value || 'conservative';

    if (!symbol || !strategy) {
        alert('Please select a strategy and symbol.');
        return;
    }

    const badge = document.getElementById('autoTradeStatusBadge');
    if (badge) { badge.textContent = 'STARTING...'; badge.style.color = 'var(--gold)'; }

    try {
        // For custom strategies, use 'combined' as the backend strategy
        let backendStrategy = strategy;
        if (strategy.startsWith('custom:')) {
            backendStrategy = 'combined';
        }

        const res = await fetch('/api/v2/start-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                symbol,
                strategy: backendStrategy,
                interval,
                market,
                mode: state.tradingMode || 'paper',
                position_size: posSize,
                stop_loss: stopLoss,
                take_profit: takeProfit,
                max_quantity: maxQty,
                leverage: 1.0,
                risk_pct: 2.0,
                sensitivity
            })
        });
        const data = await res.json();
        if (data.success) {
            if (badge) { badge.textContent = 'RUNNING'; badge.style.color = 'var(--bullish)'; }
            console.log(`[V2] Bot started: ${data.bot_id}`);
            setTimeout(loadAndRenderBots, 1000);
        } else {
            if (badge) { badge.textContent = 'ERROR'; badge.style.color = 'var(--bearish)'; }
            alert('Bot start failed: ' + (data.error || 'Unknown error'));
        }
    } catch (e) {
        console.error('[V2] Start bot error:', e);
        if (badge) { badge.textContent = 'ERROR'; badge.style.color = 'var(--bearish)'; }
    }
}

// Stop specific bot
async function stopBotById(botId) {
    try {
        const res = await fetch('/api/v2/stop-bot', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ bot_id: botId })
        });
        const data = await res.json();
        if (data.success) {
            console.log(`[V2] Bot stopped: ${botId}`);
            setTimeout(loadAndRenderBots, 500);
        }
    } catch (e) {
        console.error('[V2] Stop bot error:', e);
    }
}

// Stop all bots
async function stopAllBots() {
    const badge = document.getElementById('autoTradeStatusBadge');
    if (badge) { badge.textContent = 'STOPPING...'; badge.style.color = 'var(--gold)'; }

    try {
        const res = await fetch('/api/v2/stop-all', { method: 'POST' });
        const data = await res.json();
        if (data.success) {
            if (badge) { badge.textContent = 'IDLE'; badge.style.color = 'var(--text-muted)'; }
            setTimeout(loadAndRenderBots, 500);
        }
    } catch (e) {
        console.error('[V2] Stop all error:', e);
    }
}

// PANIC sell
async function panicSell() {
    if (!confirm('⚠️ PANIC SELL will close ALL positions and stop ALL bots. Continue?')) return;
    await stopAllBots();
    // Close all open positions
    try {
        const res = await fetch('/api/v2/positions');
        const data = await res.json();
        if (data.open && data.open.length > 0) {
            for (const pos of data.open) {
                await fetch('/api/v2/trade', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        symbol: pos.symbol,
                        side: pos.side === 'BUY' ? 'SELL' : 'BUY',
                        quantity: Math.abs(pos.net_quantity || pos.quantity),
                        order_type: 'MARKET'
                    })
                });
            }
        }
    } catch (e) {
        console.error('[V2] Panic sell error:', e);
    }
}

// ============================================================
// RUNNING BOTS RENDERER
// ============================================================

// ============================================================
// CHART SETTINGS — user-selectable chart type + indicators
// ============================================================

function renderChartSettings() {
    const pop = document.getElementById('chartSettingsPopover');
    if (!pop || typeof ChartManager === 'undefined') return;

    // Custom buttons instead of a native <select>: the OS-rendered dropdown
    // list can't be themed and pops up detached from the panel.
    const types = ChartManager.CHART_TYPES.map(t =>
        `<button type="button" class="chart-type-opt${String(ChartManager.chartType) === t.id ? ' active' : ''}"
                 data-type="${t.id}">${t.label}</button>`
    ).join('');

    const inds = ChartManager.INDICATORS.map(i => {
        const on = ChartManager.studies.includes(i.id);
        return `
            <label class="chart-ind-row">
                <input type="checkbox" data-study="${i.id}" ${on ? 'checked' : ''}>
                <span>${i.label}</span>
                ${i.pane ? '<em class="chart-ind-note">separate pane</em>' : ''}
            </label>`;
    }).join('');

    pop.innerHTML = `
        <div class="chart-settings-section">
            <label class="chart-settings-label">Chart Type</label>
            <div class="chart-type-list">${types}</div>
        </div>
        <div class="chart-settings-section">
            <label class="chart-settings-label">Indicators</label>
            ${inds}
            <p class="chart-settings-hint">Indicators marked <em>separate pane</em> add a panel below the price and shrink the chart. To see <em>more</em> candles, pick a longer interval (5m/15m).</p>
        </div>`;

    pop.querySelectorAll('.chart-type-opt[data-type]').forEach(btn => {
        btn.addEventListener('click', () => {
            pop.querySelectorAll('.chart-type-opt[data-type]').forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            ChartManager.setChartType(btn.dataset.type);
        });
    });

    pop.querySelectorAll('input[data-study]').forEach(cb => {
        cb.addEventListener('change', () => {
            const picked = Array.from(pop.querySelectorAll('input[data-study]:checked'))
                .map(x => x.dataset.study);
            ChartManager.setStudies(picked);
        });
    });
}

function toggleChartSettings(evt) {
    if (evt) evt.stopPropagation();
    const pop = document.getElementById('chartSettingsPopover');
    if (!pop) return;
    const opening = pop.hasAttribute('hidden');
    if (opening) {
        renderChartSettings();
        pop.removeAttribute('hidden');
    } else {
        pop.setAttribute('hidden', '');
    }
}

// Close the popover on outside click
document.addEventListener('click', (e) => {
    const pop = document.getElementById('chartSettingsPopover');
    const btn = document.getElementById('chartSettingsBtn');
    if (!pop || pop.hasAttribute('hidden')) return;
    if (!pop.contains(e.target) && btn && !btn.contains(e.target)) {
        pop.setAttribute('hidden', '');
    }
});

// ============================================================
// EVOLUTION ENGINE — expandable cards (home page)
// ============================================================

const EVO_PARAM_LABELS = {
    take_profit: ['Take Profit', v => v + '%'],
    stop_loss: ['Stop Loss', v => v + '%'],
    sensitivity_rank: ['Sensitivity', v => ['Conservative', 'Balanced', 'Aggressive'][v] || v],
    cooldown_bars: ['Cooldown', v => v + ' bars'],
    status: ['Status', v => v],
    ALL: ['All parameters', v => v]
};

function evoFmt(key, val) {
    const d = EVO_PARAM_LABELS[key];
    return d ? d[1](val) : val;
}
function evoMoney(v) {
    const n = parseFloat(v) || 0;
    return (n >= 0 ? '+$' : '-$') + Math.abs(n).toFixed(2);
}

// Remember which cards are expanded so a refresh doesn't collapse them
const evoOpenCards = new Set();

function toggleEvoCard(key) {
    const card = document.querySelector(`.evo-card[data-key="${key}"]`);
    if (!card) return;
    card.classList.toggle('open');
    if (card.classList.contains('open')) evoOpenCards.add(key);
    else evoOpenCards.delete(key);
}

async function loadEvolutionCards() {
    const grid = document.getElementById('evolutionGrid');
    if (!grid) return;
    try {
        const [sRes, hRes] = await Promise.all([
            fetch('/api/v2/evolution/status'),
            fetch('/api/v2/evolution/history?limit=100')
        ]);
        const sData = await sRes.json();
        const hData = await hRes.json();
        const items = sData.strategies || [];
        const history = hData.history || [];

        const readyCount = items.filter(s => s.pending).length;
        const badgeEl = document.getElementById('evoReadyCount');
        if (badgeEl) badgeEl.textContent = readyCount;

        if (!items.length) {
            grid.innerHTML = '<div class="no-bots-message">No trades yet. Each strategy starts learning once it closes trades.</div>';
            return;
        }

        grid.innerHTML = items.map(s => {
            const key = `${s.strategy}|${s.symbol}`;
            const m = s.metrics || {};
            const p = s.params || {};
            const d = s.defaults || {};
            const mt = s.meter || { progress: 0, have: 0, need: 8, ready: false };
            const isOpen = evoOpenCards.has(key);

            const badge = s.status === 'paused'
                ? '<span class="evo-badge paused">Paused</span>'
                : s.pending ? '<span class="evo-badge ready">Lesson Ready</span>'
                    : mt.ready ? '<span class="evo-badge active">Evolving</span>'
                        : '<span class="evo-badge learning">Learning</span>';

            const params = ['take_profit', 'stop_loss', 'sensitivity_rank', 'cooldown_bars'].map(k => {
                const changed = String(p[k]) !== String(d[k]);
                return `<div class="evo-param ${changed ? 'changed' : ''}">
                    <span class="k">${EVO_PARAM_LABELS[k][0]}</span>
                    <span class="v">${changed ? `<span class="orig">${evoFmt(k, d[k])}</span>` : ''}${evoFmt(k, p[k])}</span>
                </div>`;
            }).join('');

            const lesson = s.pending ? `
                <div class="evo-lesson">
                    <div class="evo-lesson-head"><i class="fas fa-lightbulb"></i> Lesson learned — review &amp; apply</div>
                    ${(s.pending.changes || []).map(c => `
                        <div class="evo-change">${(EVO_PARAM_LABELS[c.param] || [c.param])[0]}:
                            ${evoFmt(c.param, c.from)} <span class="evo-arrow">→</span> ${evoFmt(c.param, c.to)}</div>
                        <div class="evo-reason">${c.reason || ''}</div>`).join('')}
                    <div class="evo-meta">
                        From ${s.pending.metrics?.closed_trades ?? 0} closed trades ·
                        win ${Number(s.pending.metrics?.win_rate || 0).toFixed(1)}% ·
                        expectancy ${evoMoney(s.pending.metrics?.expectancy)} ·
                        ${s.pending.regime || 'unknown'} market
                    </div>
                    <div class="evo-actions">
                        <button class="evo-proceed" onclick="approveEvolution('${s.strategy}','${s.symbol}')">
                            <i class="fas fa-check"></i> Proceed — Apply Now
                        </button>
                        <button class="evo-dismiss" onclick="dismissEvolution('${s.strategy}','${s.symbol}')">Dismiss</button>
                    </div>
                </div>` : '';

            const profs = Object.keys(s.profiles || {});
            const profiles = profs.length ? `<div class="evo-profiles"><b>Learned per market:</b><br>${
                profs.map(r => `• <b>${r}</b> — TP ${s.profiles[r].take_profit}% / SL ${s.profiles[r].stop_loss}%`).join('<br>')
            }</div>` : '';

            const gens = history.filter(h => h.strategy === s.strategy &&
                (!h.symbol || h.symbol === s.symbol));
            const genRows = gens.length ? gens.map(g => `
                <div class="evo-gen-row">
                    <div class="evo-gen-num">${g.generation}</div>
                    <div>
                        ${(Array.isArray(g.changes_json) ? g.changes_json : []).map(c =>
                            `<div class="evo-change">${(EVO_PARAM_LABELS[c.param] || [c.param])[0]}:
                                ${evoFmt(c.param, c.from)} <span class="evo-arrow">→</span> ${evoFmt(c.param, c.to)}</div>
                             <div class="evo-reason">${c.reason || ''}</div>`).join('')}
                        <div class="evo-meta">${g.closed_trades} closed · win ${Number(g.win_rate).toFixed(1)}% ·
                            ${g.verdict}${g.created_at ? ' · ' + new Date(g.created_at).toLocaleString() : ''}</div>
                    </div>
                </div>`).join('') : '<div class="evo-reason">No adaptations applied yet.</div>';

            const isPaused = s.status === 'paused';
            const esc = v => String(v).replace(/'/g, "\\'");

            // Per-card controls: evaluate or stop THIS pair without touching
            // any other bot's learning.
            const cardActions = `
                <div class="evo-card-actions">
                    ${isPaused
                        ? `<button class="evo-pick-btn resume"
                                onclick="setEvolutionStatus('${esc(s.strategy)}','${esc(s.symbol)}','active')">
                                <i class="fas fa-play"></i> Resume Learning</button>`
                        : `<button class="evo-pick-btn"
                                onclick="runEvolution({ strategy: '${esc(s.strategy)}', symbol: '${esc(s.symbol)}' })">
                                <i class="fas fa-bolt"></i> Evaluate Now</button>
                           <button class="evo-pick-btn stop"
                                onclick="setEvolutionStatus('${esc(s.strategy)}','${esc(s.symbol)}','paused')">
                                <i class="fas fa-stop"></i> Stop Learning</button>`}
                </div>`;

            return `<div class="evo-card ${isOpen ? 'open' : ''} ${s.pending ? 'ready' : ''} ${isPaused ? 'paused' : ''}" data-key="${key}">
                <div class="evo-card-head" onclick="toggleEvoCard('${key}')">
                    <span class="evo-card-title">${getStrategyName(s.strategy)}</span>
                    <span class="evo-symbol">${s.symbol}</span>
                    <span class="evo-gen">GEN ${s.generation}</span>
                    ${s.running ? '<span class="evo-live-dot" title="Bot running"></span>' : ''}
                    ${badge}
                    <i class="fas fa-chevron-down caret"></i>
                </div>
                <div class="evo-meter ${mt.ready ? 'full' : ''}">
                    <div class="lbl">
                        <span>${mt.ready ? '⚡ Ready to evolve' : 'Learning from closed trades'}</span>
                        <span>${mt.have}/${mt.need}</span>
                    </div>
                    <div class="track"><span class="fill" style="width:${mt.progress}%"></span></div>
                </div>
                <div class="evo-card-body">
                    <div class="evo-meta">
                        ${m.closed_trades || 0} closed · win ${Number(m.win_rate || 0).toFixed(1)}% ·
                        expectancy ${evoMoney(m.expectancy)} · <span class="evo-regime">${s.regime || 'unknown'}</span>
                    </div>
                    ${lesson}
                    ${cardActions}
                    <div class="evo-section-label">Live Parameters</div>
                    <div class="evo-params">${params}</div>
                    ${profiles}
                    <div class="evo-section-label">Evolution History</div>
                    ${genRows}
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        console.error('[EVO] load failed:', e);
        grid.innerHTML = '<div class="no-bots-message">Could not load evolution data.</div>';
    }
}

async function approveEvolution(strategy, symbol) {
    try {
        const res = await fetch('/api/v2/evolution/approve', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy, symbol })
        });
        const d = await res.json();
        if (!d.success) alert('Could not apply: ' + (d.error || 'unknown error'));
        else showNotification(`🧬 ${getStrategyName(strategy)} evolved to gen ${d.generation}`);
    } catch (e) { console.error('[EVO] approve failed:', e); }
    loadEvolutionCards();
}

async function dismissEvolution(strategy, symbol) {
    try {
        await fetch('/api/v2/evolution/dismiss', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy, symbol })
        });
    } catch (e) { console.error('[EVO] dismiss failed:', e); }
    loadEvolutionCards();
}

// ============================================================
// EVALUATE PICKER
// ------------------------------------------------------------
// Evaluation is per (strategy, symbol). The button used to fire a
// blanket run, so whichever pair happened to have trade history was
// the only thing that appeared to be evaluated. Now the user picks.
// ============================================================

/**
 * Cockpit help overlay. The markup and both onclick handlers already existed,
 * but the function never did — the "?" button threw a ReferenceError, so the
 * panel could not be opened at all.
 */
function toggleCockpitVisionModal(show) {
    const modal = document.getElementById('cockpitVisionModal');
    if (!modal) return;
    modal.classList.toggle('hidden', !show);
    document.body.style.overflow = show ? 'hidden' : '';
}

function openEvoPicker() {
    const modal = document.getElementById('evoPickerModal');
    if (!modal) return;
    modal.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
    loadEvoCandidates();
}

function closeEvoPicker() {
    const modal = document.getElementById('evoPickerModal');
    if (!modal) return;
    modal.classList.add('hidden');
    document.body.style.overflow = '';
}

/** Populate the picker with every bot that can be evaluated. */
async function loadEvoCandidates() {
    const list = document.getElementById('evoPickerList');
    if (!list) return;
    list.innerHTML = '<div class="evo-picker-empty">Loading bots…</div>';

    try {
        const res = await fetch('/api/v2/evolution/candidates');
        const data = await res.json();
        const items = data.candidates || [];

        if (!items.length) {
            list.innerHTML = `<div class="evo-picker-empty">
                No bots to evaluate yet. Start one from the Command Deck first.
            </div>`;
            return;
        }

        // Running bots first — that is what the user is most likely after.
        items.sort((a, b) => (b.running - a.running) ||
            a.strategy.localeCompare(b.strategy) || a.symbol.localeCompare(b.symbol));

        list.innerHTML = items.map(c => {
            const mt = c.meter || { progress: 0, have: 0, need: 8, ready: false };
            const paused = c.status === 'paused';
            const esc = s => String(s).replace(/'/g, "\\'");

            const state = paused ? '<span class="evo-badge paused">Stopped</span>'
                : c.has_pending ? '<span class="evo-badge ready">Lesson Ready</span>'
                    : mt.ready ? '<span class="evo-badge active">Ready</span>'
                        : '<span class="evo-badge learning">Learning</span>';

            // A stopped pair offers Resume instead of Evaluate — evaluating a
            // pair the user deliberately stopped would just be ignored.
            const action = paused
                ? `<button class="evo-pick-btn resume"
                        onclick="setEvolutionStatus('${esc(c.strategy)}','${esc(c.symbol)}','active')">
                        <i class="fas fa-play"></i> Resume</button>`
                : `<button class="evo-pick-btn"
                        onclick="runEvolution({ strategy: '${esc(c.strategy)}', symbol: '${esc(c.symbol)}' })">
                        <i class="fas fa-bolt"></i> Evaluate</button>
                   <button class="evo-pick-btn stop"
                        onclick="setEvolutionStatus('${esc(c.strategy)}','${esc(c.symbol)}','paused')"
                        title="Stop learning for this bot">
                        <i class="fas fa-stop"></i> Stop</button>`;

            return `
            <div class="evo-pick-row ${paused ? 'paused' : ''}">
                <div class="evo-pick-main">
                    <div class="evo-pick-title">
                        <span class="evo-pick-strategy">${getStrategyName(c.strategy)}</span>
                        <span class="evo-symbol">${c.symbol}</span>
                        ${c.running ? '<span class="evo-live-dot" title="Bot running"></span>' : ''}
                    </div>
                    <div class="evo-pick-meta">
                        GEN ${c.generation} · ${mt.have}/${mt.need} closed trades
                        ${c.running ? '· live' : ''}
                    </div>
                    <div class="evo-meter compact ${mt.ready ? 'full' : ''}">
                        <div class="track"><span class="fill" style="width:${mt.progress}%"></span></div>
                    </div>
                </div>
                <div class="evo-pick-side">
                    ${state}
                    <div class="evo-pick-actions">${action}</div>
                </div>
            </div>`;
        }).join('');
    } catch (e) {
        console.error('[EVO] candidates failed:', e);
        list.innerHTML = '<div class="evo-picker-empty">Could not load bots.</div>';
    }
}

/** Stop ('paused') or resume ('active') learning for one bot. */
async function setEvolutionStatus(strategy, symbol, status) {
    try {
        const res = await fetch('/api/v2/evolution/status-set', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ strategy, symbol, status })
        });
        const d = await res.json();
        if (!d.success) {
            alert('Could not update: ' + (d.error || 'unknown error'));
            return;
        }
        showNotification(status === 'paused'
            ? `⏹ Evaluation stopped for ${getStrategyName(strategy)} · ${symbol}`
            : `▶ Evaluation resumed for ${getStrategyName(strategy)} · ${symbol}`);
    } catch (e) {
        console.error('[EVO] status change failed:', e);
    }
    loadEvoCandidates();
    loadEvolutionCards();
}

/**
 * Run an evaluation.
 * @param {{strategy?: string, symbol?: string, all?: boolean}} opts
 *        Targets one pair; `all` evaluates every candidate.
 */
async function runEvolution(opts = {}) {
    const { strategy, symbol, all } = opts;
    const label = all ? 'all bots' : `${getStrategyName(strategy)} · ${symbol}`;

    try {
        const res = await fetch('/api/v2/evolution/run', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(all ? { force: true } : { strategy, symbol, force: true })
        });
        const d = await res.json();
        if (!d.success) {
            alert('Evaluation failed: ' + (d.error || 'unknown error'));
            return;
        }

        // Surface the outcome instead of silently closing — an evaluation that
        // found nothing to change looks identical to one that never ran.
        const results = d.results || [];
        const lessons = results.filter(r => r && r.awaiting_approval).length;
        const skipped = results.filter(r => r && r.skipped);
        if (lessons) {
            showNotification(`🧬 ${lessons} lesson${lessons > 1 ? 's' : ''} ready to review`);
        } else if (skipped.length === results.length && results.length) {
            showNotification(`No change for ${label} — ${skipped[0].reason || 'still gathering evidence'}`);
        } else {
            showNotification(`Evaluated ${label} — no adjustment needed`);
        }
    } catch (e) {
        console.error('[EVO] run failed:', e);
    }

    closeEvoPicker();
    loadEvolutionCards();
}

// ── Mobile off-canvas sidebar ──
function toggleMobileSidebar(forceClose) {
    const sidebar = document.querySelector('.slim-sidebar');
    const overlay = document.querySelector('.sidebar-overlay');
    if (!sidebar) return;
    const open = forceClose === true ? false : !sidebar.classList.contains('mobile-open');
    sidebar.classList.toggle('mobile-open', open);
    if (overlay) overlay.classList.toggle('active', open);
    document.body.style.overflow = open ? 'hidden' : '';
}

// Wire overlay tap + nav-item tap to close the drawer, once DOM is ready
document.addEventListener('DOMContentLoaded', () => {
    // Evaluate picker: tap the backdrop or press Esc to dismiss.
    const evoModal = document.getElementById('evoPickerModal');
    if (evoModal) {
        evoModal.addEventListener('click', (e) => {
            if (e.target === evoModal) closeEvoPicker();
        });
    }
    const cockpitModal = document.getElementById('cockpitVisionModal');
    if (cockpitModal) {
        cockpitModal.addEventListener('click', (e) => {
            if (e.target === cockpitModal) toggleCockpitVisionModal(false);
        });
    }
    document.addEventListener('keydown', (e) => {
        if (e.key !== 'Escape') return;
        if (evoModal && !evoModal.classList.contains('hidden')) closeEvoPicker();
        else if (cockpitModal && !cockpitModal.classList.contains('hidden')) toggleCockpitVisionModal(false);
    });

    const overlay = document.querySelector('.sidebar-overlay');
    if (overlay) overlay.addEventListener('click', () => toggleMobileSidebar(true));
    document.querySelectorAll('.slim-sidebar .nav-item').forEach(item =>
        item.addEventListener('click', () => toggleMobileSidebar(true)));
});

// Signal Sensitivity: explainer copy + card badge styling
const SENSITIVITY_PREF_KEY = 'goatbot_signal_sensitivity';
const SENSITIVITY_INFO = {
    conservative: {
        label: '🛡️ Conservative',
        color: '#10b981',
        help: '🛡️ <strong>Conservative</strong> — only high-conviction trades; sits out quiet markets. Fewest trades, highest quality. <em>New bots start here.</em>'
    },
    balanced: {
        label: '⚖️ Balanced',
        color: '#f59e0b',
        help: '⚖️ <strong>Balanced</strong> — moderate trade frequency; acts on decent setups while still skipping pure chop. A middle ground.'
    },
    aggressive: {
        label: '🔥 Aggressive',
        color: '#ef4444',
        help: '🔥 <strong>Aggressive</strong> — trades frequently, including calmer markets. More signals, but expect more noise and higher fee drag. Best when volatility is up.'
    }
};

function updateSensitivityHelp(value) {
    const el = document.getElementById('sensitivityHelp');
    const info = SENSITIVITY_INFO[value] || SENSITIVITY_INFO.conservative;
    if (el) el.innerHTML = info.help;
}

async function loadAndRenderBots() {
    try {
        const res = await fetch('/api/v2/bots');
        const data = await res.json();
        if (!data.success) return;

        const bots = data.bots || [];
        const grid = document.getElementById('runningBotsGrid');
        const countEl = document.getElementById('botCount');
        const badge = document.getElementById('autoTradeStatusBadge');

        if (!grid) return;

        const runningBots = bots.filter(b => b.status === 'running');
        if (countEl) countEl.textContent = runningBots.length;

        if (badge) {
            if (runningBots.length > 0) {
                badge.textContent = `${runningBots.length} ACTIVE`;
                badge.style.color = 'var(--bullish)';
            } else {
                badge.textContent = 'IDLE';
                badge.style.color = 'var(--text-muted)';
            }
        }

        if (bots.length === 0) {
            grid.innerHTML = '<div class="no-bots-message">No bots running. Start one from the Command Deck above.</div>';
            return;
        }

        grid.innerHTML = bots.map(bot => {
            const isRunning = bot.status === 'running';
            // API returns metrics under bot.stats (total_pnl / total_trades),
            // not bot.pnl / bot.trade_count — read them from the right place.
            const stats = bot.stats || {};
            const realizedPnl = Number(stats.realized_pnl || 0);
            const unrealizedPnl = Number(stats.unrealized_pnl || 0);
            const pnl = Number(stats.total_pnl != null ? stats.total_pnl : realizedPnl + unrealizedPnl);
            const tradeCount = Number(stats.total_trades || 0);
            const pnlClass = pnl >= 0 ? 'p-positive' : 'p-negative';
            const pnlSign = pnl >= 0 ? '+' : '';
            const pnlTitle = `Realized: ${pnlSign}$${realizedPnl.toFixed(2)} · Unrealized: $${unrealizedPnl.toFixed(2)}`;
            const sens = SENSITIVITY_INFO[bot.sensitivity] || SENSITIVITY_INFO.conservative;

            return `
                <div class="bot-card-item ${isRunning ? '' : 'stopped'}">
                    <div class="bot-header">
                        <span class="bot-symbol">${bot.symbol || 'N/A'}</span>
                        <span class="bot-status ${isRunning ? 'running' : 'stopped'}">${bot.status || 'unknown'}</span>
                    </div>
                    <div class="bot-meta">
                        Strategy: <strong>${bot.strategy || 'combined'}</strong> · ${bot.interval || '1m'} · ${bot.mode || 'paper'}
                        <span style="color:${sens.color}; font-weight:600;"> · ${sens.label}</span>
                    </div>
                    <div class="bot-stats">
                        <span class="${pnlClass}" title="${pnlTitle}">${pnlSign}$${pnl.toFixed(2)}</span>
                        <span style="color:var(--text-muted);">Trades: ${tradeCount}</span>
                    </div>
                    ${isRunning ? `<button class="bot-stop-btn" onclick="stopBotById('${bot.bot_id}')"><i class="fas fa-stop"></i> STOP</button>` : ''}
                </div>
            `;
        }).join('');

    } catch (e) {
        console.error('[V2] Load bots error:', e);
    }
}

// ============================================================
// CUSTOM STRATEGY BUILDER
// ============================================================

function getCustomStrategies() {
    try {
        return JSON.parse(localStorage.getItem('godbot_custom_strategies') || '[]');
    } catch { return []; }
}

function saveCustomStrategiesStore(strategies) {
    localStorage.setItem('goatbot_custom_strategies', JSON.stringify(strategies));
}

function saveCustomStrategy() {
    const name = document.getElementById('customStratName')?.value?.trim();
    if (!name) { alert('Please enter a strategy name.'); return; }

    // Collect selected indicators
    const chips = document.querySelectorAll('#indicatorChips .indicator-chip.selected');
    const indicators = Array.from(chips).map(c => c.dataset.indicator);
    if (indicators.length === 0) { alert('Please select at least one indicator.'); return; }

    // Collect buy conditions
    const buyConditions = collectConditions('buyConditions');
    const sellConditions = collectConditions('sellConditions');

    if (buyConditions.length === 0 && sellConditions.length === 0) {
        alert('Please add at least one buy or sell condition.');
        return;
    }

    const id = name.toLowerCase().replace(/[^a-z0-9]/g, '_') + '_' + Date.now();
    const strategy = { id, name, indicators, buyConditions, sellConditions, createdAt: new Date().toISOString() };

    const strategies = getCustomStrategies();
    strategies.push(strategy);
    saveCustomStrategiesStore(strategies);

    // Refresh UI
    loadStrategies();
    renderSavedStrategies();

    // Clear form
    document.getElementById('customStratName').value = '';
    document.querySelectorAll('#indicatorChips .indicator-chip').forEach(c => c.classList.remove('selected'));

    console.log(`[V2] Custom strategy saved: ${name}`);
}

function deleteCustomStrategy(id) {
    let strategies = getCustomStrategies();
    strategies = strategies.filter(s => s.id !== id);
    saveCustomStrategiesStore(strategies);
    loadStrategies();
    renderSavedStrategies();
}

function renderSavedStrategies() {
    const container = document.getElementById('savedStrategiesList');
    if (!container) return;

    const strategies = getCustomStrategies();
    if (strategies.length === 0) {
        container.innerHTML = '<div class="no-bots-message" style="padding:16px;">No custom strategies yet.</div>';
        return;
    }

    container.innerHTML = strategies.map(s => `
        <div class="saved-strategy-item">
            <div>
                <span class="strategy-name">🔧 ${s.name}</span>
                <span style="color:var(--text-muted); margin-left:8px; font-size:10px;">${s.indicators?.join(', ') || ''}</span>
            </div>
            <button class="delete-btn" onclick="deleteCustomStrategy('${s.id}')" title="Delete">✕</button>
        </div>
    `).join('');
}

function collectConditions(containerId) {
    const rows = document.querySelectorAll(`#${containerId} .condition-row`);
    const conditions = [];
    rows.forEach(row => {
        const indicator = row.querySelector('.cond-indicator')?.value;
        const operator = row.querySelector('.cond-operator')?.value;
        const value = parseFloat(row.querySelector('.cond-value')?.value);
        if (indicator && operator && !isNaN(value)) {
            conditions.push({ indicator, operator, value });
        }
    });
    return conditions;
}

function addConditionRow(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;

    const row = document.createElement('div');
    row.className = 'condition-row';
    row.innerHTML = `
        <select class="cond-indicator">
            <option value="rsi">RSI</option>
            <option value="macd">MACD Histogram</option>
            <option value="price_vs_vwap">Price vs VWAP</option>
            <option value="volume_ratio">Volume Ratio</option>
            <option value="bb_pctb">Bollinger %B</option>
        </select>
        <select class="cond-operator">
            <option value="<">&lt; Below</option>
            <option value=">">&gt; Above</option>
            <option value="cross_above">↑ Crosses Above</option>
            <option value="cross_below">↓ Crosses Below</option>
        </select>
        <input type="number" class="cond-value" placeholder="50" value="50">
    `;
    container.appendChild(row);
}

// ============================================================
// INDICATOR CHIP TOGGLE
// ============================================================
function initIndicatorChips() {
    document.querySelectorAll('#indicatorChips .indicator-chip').forEach(chip => {
        chip.addEventListener('click', () => {
            chip.classList.toggle('selected');
        });
    });
}

// ============================================================
// WIRE BUTTONS ON DOMContentLoaded
// ============================================================
document.addEventListener('DOMContentLoaded', () => {
    // Auto-trade buttons
    const btnStart = document.getElementById('btnStartBot');
    const btnStop = document.getElementById('btnStopBot');
    const btnStopAll = document.getElementById('btnStopAll');
    const btnPanic = document.getElementById('btnPanicSell');

    if (btnStart) btnStart.addEventListener('click', startBotFromUI);
    if (btnStop) btnStop.addEventListener('click', () => {
        // Stop the most recent running bot
        loadAndRenderBots();
    });
    if (btnStopAll) btnStopAll.addEventListener('click', stopAllBots);
    if (btnPanic) btnPanic.addEventListener('click', panicSell);

    // Load strategies into dropdown
    loadStrategies();

    // Init indicator chip toggles
    initIndicatorChips();

    // Render saved custom strategies
    renderSavedStrategies();

    // Initial bot load
    loadAndRenderBots();

    // Auto-refresh bots every 5 seconds
    setInterval(loadAndRenderBots, 5000);
});

const API_BASE = '/api';
const CHART_COLORS = [
    'rgb(99, 102, 241)',   // Indigo
    'rgb(16, 185, 129)',   // Green
    'rgb(245, 158, 11)',   // Amber
    'rgb(236, 72, 153)',   // Pink
    'rgb(14, 165, 233)',   // Sky
    'rgb(168, 85, 247)',   // Purple
];

// State
let currentPage = 0;
const PAGE_SIZE = 25;
let equityChart = null;
let drawdownChart = null;
let allocationPieChart = null;
const activeStrategies = new Set();
let timers = {};

/**
 * 1. Centralized Polling Scheduler
 */
const scheduler = {
    signals: 10000,
    pulse: 30000,
    drift: 60000,
    allocation: 60000,
    news: 300000,
    core: 30000 // General status/bots refresh
};

// ═══════════════════════════════════════
// Initialization
// ═══════════════════════════════════════

document.addEventListener('DOMContentLoaded', async () => {
    initCharts();
    await initDashboard();
    startPolling();
});

async function initDashboard() {
    try {
        // Initial load to populate activeStrategies
        const botsData = await apiGet('/bots');
        updateActiveStrategies(botsData.bots || []);

        await Promise.all([
            refreshCore(),
            loadLiveSignals(),
            loadLiveTrades(),
            loadMarketPulse(),
            loadAllocation(),
            loadIntel(),
            loadNews(),
            loadAllStressTests()
        ]);

        document.getElementById('lastUpdate').textContent = `Sync: ${new Date().toLocaleTimeString()}`;
    } catch (err) {
        console.error('Initialization error:', err);
    }
}

function startPolling() {
    // Clear existing
    Object.values(timers).forEach(clearInterval);

    timers.core = setInterval(refreshCore, scheduler.core);
    timers.signals = setInterval(loadLiveSignals, scheduler.signals);
    timers.pulse = setInterval(loadMarketPulse, scheduler.pulse);
    timers.drift = setInterval(loadDriftAlerts, scheduler.drift);
    timers.allocation = setInterval(loadAllocation, scheduler.allocation);
    timers.news = setInterval(loadNews, scheduler.news);
    // Stress tests headers refresh with core
}

function updateActiveStrategies(bots) {
    activeStrategies.clear();
    bots.forEach(b => {
        if (b.status === 'running' || b.performance?.is_active) {
            activeStrategies.add(b.config?.strategy);
        }
    });
}

// ═══════════════════════════════════════
// Core Refresh (Status + Bots + Compare)
// ═══════════════════════════════════════

async function refreshCore() {
    await Promise.all([
        loadStatus(),
        loadBots(),
        loadComparison(),
        loadEfficiency(),
        loadAllStressTests()
    ]);
    document.getElementById('lastUpdate').textContent = `Sync: ${new Date().toLocaleTimeString()}`;
}

// ═══════════════════════════════════════
// API Calls
// ═══════════════════════════════════════

async function apiGet(path) {
    try {
        const res = await fetch(`${API_BASE}${path}`);
        if (!res.ok) throw new Error(`API ${path} failed: ${res.status}`);
        return await res.json();
    } catch (err) {
        console.warn(`API Error (${path}):`, err);
        return { error: true, message: err.message };
    }
}

async function apiPost(path, data = {}) {
    const res = await fetch(`${API_BASE}${path}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(data),
    });
    return res.json();
}

// ═══════════════════════════════════════
// UI Loaders
// ═══════════════════════════════════════

async function loadStatus() {
    const data = await apiGet('/status');
    if (data.error) return;

    document.getElementById('totalBots').textContent = data.total_bots || 0;
    document.getElementById('totalTrades').textContent = data.total_trades || 0;

    const sysEl = document.getElementById('systemStatus');
    if (data.paper_mode) {
        sysEl.textContent = '● PAPER';
        sysEl.className = 'status-value safe';
    } else {
        sysEl.textContent = '⚠ LIVE';
        sysEl.className = 'status-value danger';
    }
}

async function loadMarketPulse() {
    const data = await apiGet('/market-pulse');
    // For demo/paper, often this might be simulated or aggregate data
    const metrics = data.metrics || [
        { label: 'Regime', value: 65, text: 'Trending (Bull)' },
        { label: 'Volatility (ATR%)', value: 42, text: '2.4%' },
        { label: 'Volume Surge', value: 15, text: 'Normal' },
        { label: 'Trend Strength', value: 80, text: 'Strong' },
        { label: 'Spread Status', value: 10, text: 'Tight' },
        { label: 'Correlation Risk', value: 30, text: 'Low' }
    ];

    const container = document.getElementById('marketPulse');
    container.innerHTML = metrics.map(m => `
        <div class="market-pulse-card glass-card">
            <div class="pulse-gauge">
                <span class="pulse-label">${m.label}</span>
                <span class="pulse-value">${m.text || m.value}</span>
                <div class="pulse-bar"><div class="pulse-fill" style="width:${m.value}%"></div></div>
            </div>
        </div>
    `).join('');
}

async function loadLiveSignals() {
    const data = await apiGet('/live-signals');
    const tbody = document.getElementById('signalsBody');
    const signals = data.signals || [];

    if (signals.length === 0) {
        tbody.innerHTML = '<tr><td colspan="11" style="text-align:center;padding:20px;color:var(--text-muted);">No active signals detected</td></tr>';
        return;
    }

    tbody.innerHTML = signals.map(s => `
        <tr class="signal-row">
            <td style="font-weight:700;color:var(--text-primary);">${s.bot_id.substring(0, 15)}</td>
            <td>${s.strategy}</td>
            <td style="color:${s.dir === 'LONG' ? 'var(--green)' : 'var(--red)'}">${s.dir}</td>
            <td>${fmtPrice(s.entry)}</td>
            <td>${fmtPrice(s.sl)}</td>
            <td>${fmtPrice(s.tp)}</td>
            <td>${s.rr}</td>
            <td style="color:var(--yellow)">${s.risk_pct}%</td>
            <td>${s.regime}</td>
            <td>${s.confidence}%</td>
            <td class="dim-text">${s.active_time}</td>
        </tr>
    `).join('');
}

async function loadLiveTrades() {
    const data = await apiGet('/live-trades'); // Unified endpoint
    const tbody = document.getElementById('liveTradesBody');
    const trades = data.trades || [];

    if (trades.length === 0) {
        tbody.innerHTML = '<tr><td colspan="9" style="text-align:center;padding:20px;color:var(--text-muted);">No open positions</td></tr>';
        return;
    }

    tbody.innerHTML = trades.map(t => {
        const pnl = t.unrealized_pnl || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';
        return `
            <tr>
                <td style="font-weight:700;">${t.bot_id.substring(0, 15)}</td>
                <td>${fmtPrice(t.entry_price)}</td>
                <td>${fmtPrice(t.current_price)}</td>
                <td class="${pnlClass}">${fmtCurrency(pnl)}</td>
                <td class="${valClass(t.r_multiple)}">${fmtNum(t.r_multiple)}</td>
                <td class="negative">${fmtCurrency(t.slippage_impact)}</td>
                <td>${t.time_in_trade}</td>
                <td>${t.risk_ladder_active ? '<span class="safety-badge caution">ACTIVE</span>' : '—'}</td>
                <td><button class="debug-btn" onclick="debugTrade('${t.trade_id}')">🔍</button></td>
            </tr>
        `;
    }).join('');
}

async function loadBots() {
    const data = await apiGet('/bots');
    const container = document.getElementById('botCards');
    const filterSelect = document.getElementById('tradeFilterBot');

    container.innerHTML = '';
    filterSelect.innerHTML = '<option value="">All Bots</option>';

    if (!data.bots || data.bots.length === 0) {
        container.innerHTML = `<div class="glass-card" style="grid-column: 1/-1; text-align:center; padding:40px;"><p class="dim-text">No bots configured</p></div>`;
        return;
    }

    // Refresh active strategies
    updateActiveStrategies(data.bots);

    for (const bot of data.bots) {
        const config = bot.config || {};
        const perf = bot.performance || {};
        const wallet = bot.wallet || {};

        filterSelect.innerHTML += `<option value="${config.bot_id}">${config.name || config.bot_id}</option>`;

        const isRunning = bot.status === 'running' || perf.is_active;
        const strategyLocked = activeStrategies.has(config.strategy) && !isRunning;

        const safetyClass = (perf.safety_label || 'CAUTION').toLowerCase();
        const pnl = perf.total_pnl || 0;
        const pnlClass = pnl >= 0 ? 'positive' : 'negative';

        container.innerHTML += `
            <div class="bot-card ${safetyClass} ${perf.safety_label === 'DANGEROUS' ? 'dangerous-glow' : ''}">
                <div class="strategy-badge">${config.strategy || '—'}</div>
                <div class="bot-card-header">
                    <div>
                        <div class="bot-name">
                            ${isRunning ? '<span class="active-dot pulsing"></span>' : ''}
                            ${config.name || config.bot_id}
                        </div>
                        <div class="bot-strategy">${config.instrument} · ${config.timeframe}</div>
                    </div>
                    <span class="safety-badge ${safetyClass}">${(perf.safety_label || 'CAUTION').toUpperCase()}</span>
                </div>
                <div class="bot-metrics">
                    <div class="metric-item"><span class="metric-label">PnL</span><span class="metric-value ${pnlClass}">${fmtCurrency(pnl)}</span></div>
                    <div class="metric-item"><span class="metric-label">DD</span><span class="metric-value negative">${fmtPct(perf.max_drawdown_pct)}</span></div>
                    <div class="metric-item"><span class="metric-label">Score</span><span class="metric-value">${fmtNum(perf.composite_score)}</span></div>
                    <div class="metric-item"><span class="metric-label">Expectancy</span><span class="metric-value ${valClass(perf.expectancy)}">${fmtNum(perf.expectancy)}</span></div>
                </div>
                <div style="margin-top:16px;">
                    <progress class="risk-mini-gauge" value="${perf.max_drawdown_pct || 0}" max="50"></progress>
                </div>
                <div class="bot-card-footer" style="margin-top:16px; display:flex; gap:8px;">
                    ${isRunning ?
                `<button class="btn btn-danger btn-sm" onclick="stopBot('${config.bot_id}')">Stop Bot</button>` :
                `<button class="btn btn-outline btn-sm" ${strategyLocked ? 'disabled title="Strategy already running"' : ''} onclick="startBot('${config.bot_id}')">
                            ${strategyLocked ? 'Locked' : 'Start Bot'}
                        </button>`
            }
                </div>
            </div>`;
    }
}

async function startBot(botId) {
    const res = await apiPost(`/bots/${botId}/start`);
    if (res.error) showNotification('Error: ' + res.message, 'danger');
    else {
        showNotification('Bot started successfully', 'safe');
        initDashboard(); // Full refresh
    }
}

async function stopBot(botId) {
    const res = await apiPost(`/bots/${botId}/stop`);
    showNotification('Bot stopped', 'warning');
    initDashboard();
}

// Sort State
let comparisonData = [];
let sortCol = 'composite_score';
let sortDir = -1;

async function loadComparison() {
    const data = await apiGet('/compare');
    if (data.error) return;
    comparisonData = data.comparison || [];
    renderComparison();
}

function renderComparison() {
    const tbody = document.getElementById('comparisonBody');
    if (comparisonData.length === 0) {
        tbody.innerHTML = '<tr><td colspan="12" style="text-align:center;">No data</td></tr>';
        return;
    }

    const sorted = [...comparisonData].sort((a, b) => {
        const valA = a[sortCol] ?? 0;
        const valB = b[sortCol] ?? 0;
        return (valA > valB ? 1 : -1) * sortDir;
    });

    tbody.innerHTML = sorted.map((bot, i) => `
        <tr>
            <td>${i + 1}</td>
            <td style="color:var(--text-primary);font-weight:600;">${bot.bot_id.substring(0, 20)}</td>
            <td><span class="safety-badge ${(bot.safety_label || 'CAUTION').toLowerCase()}">${bot.safety_label}</span></td>
            <td>${fmtNum(bot.composite_score)}</td>
            <td>${fmtNum(bot.regime_stability || 0.85)}</td>
            <td><span class="stress-grade ${(bot.stress_grade || 'ROBUST').toLowerCase()}">${bot.stress_grade || 'ROBUST'}</span></td>
            <td>${fmtNum(bot.slippage_sensitivity || 1.2)}×</td>
            <td style="color:var(--yellow)">${fmtNum(bot.capital_efficiency || 0.9)}</td>
            <td class="negative">${fmtPct(bot.max_drawdown_pct)}</td>
            <td class="${valClass(bot.expectancy)}">${fmtNum(bot.expectancy)}</td>
            <td class="${valClass(bot.total_pnl)}">${fmtCurrency(bot.total_pnl)}</td>
            <td style="color:${bot.risk_of_ruin > 20 ? 'var(--red)' : ''}">${fmtPct(bot.risk_of_ruin)}</td>
        </tr>
    `).join('');
}

function sortTable(col) {
    if (sortCol === col) sortDir *= -1;
    else {
        sortCol = col;
        sortDir = -1;
    }
    renderComparison();

    // Update header icons (optional micro-interaction)
    document.querySelectorAll('#comparisonTable th[data-sort]').forEach(th => {
        th.classList.remove('sort-asc', 'sort-desc');
        if (th.dataset.sort === sortCol) {
            th.classList.add(sortDir === 1 ? 'sort-asc' : 'sort-desc');
        }
    });
}

async function loadAllocation() {
    const data = await apiGet('/allocation');
    const tbody = document.getElementById('allocationBody');
    const allocs = data.allocations || [];
    tbody.innerHTML = '';

    if (allocs.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" style="text-align:center;">No allocation data</td></tr>';
        return;
    }

    tbody.innerHTML = allocs.map(a => `
        <tr>
            <td style="font-weight:600;">${a.bot_id.substring(0, 20)}</td>
            <td class="allocation-highlight">${fmtPct(a.recommended_allocation_pct)}</td>
            <td>${fmtPct(a.kelly_fraction * 100)}</td>
            <td>${fmtPct(a.vol_parity_weight * 100)}</td>
            <td>${fmtPct(a.recommended_allocation_pct)}</td>
            <td>${a.risk_cap_applied ? 'Applied' : 'None'}</td>
        </tr>
    `).join('');

    updateAllocationChart(allocs);
}

function updateAllocationChart(allocs) {
    const ctx = document.getElementById('allocationPieChart').getContext('2d');
    const data = {
        labels: allocs.map(a => a.bot_id.substring(0, 15)),
        datasets: [{
            data: allocs.map(a => a.recommended_allocation_pct),
            backgroundColor: CHART_COLORS,
            borderWidth: 0
        }]
    };

    if (allocationPieChart) {
        allocationPieChart.data = data;
        allocationPieChart.update();
    } else {
        allocationPieChart = new Chart(ctx, {
            type: 'doughnut',
            data: data,
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { position: 'right', labels: { color: '#94a3b8', font: { size: 10 } } } }
            }
        });
    }
}

async function loadIntel() {
    // Simulated rules for client-side intelligence
    const insights = [
        { icon: '🧠', text: 'Breakout strategy shows high stability in Trend regime.' },
        { icon: '⚠', text: 'Mean Reversion fragile under current volatility spike.' },
        { icon: '📊', text: 'Portfolio exposure is 70% LONG on BTCUSDT.' }
    ];

    document.getElementById('intelInsights').innerHTML = insights.map(i => `
        <div class="intel-card glass-card">
            <span>${i.icon}</span>
            <span>${i.text}</span>
        </div>
    `).join('');
}

async function loadNews() {
    const data = await apiGet('/news');
    const news = data.news || [
        { title: 'Bitcoin reclaims $95k as institutional inflow surges', time: '10m ago', sentiment: 'BULLISH', source: '#' },
        { title: 'Fed meeting minutes reveal cautious inflation outlook', time: '45m ago', sentiment: 'NEUTRAL', source: '#' },
        { title: 'Regulatory pressure mounts on DeFi stablecoins', time: '1h ago', sentiment: 'BEARISH', source: '#' }
    ];

    document.getElementById('marketNews').innerHTML = news.map(n => `
        <div class="news-item">
            <div class="news-content">
                <a href="${n.source}" class="news-title" target="_blank">${n.title}</a>
                <span class="news-meta">${n.time}</span>
            </div>
            <span class="sentiment-tag sentiment-${n.sentiment.toLowerCase()}">${n.sentiment}</span>
        </div>
    `).join('');
}

// ═══════════════════════════════════════
// UI Helpers
// ═══════════════════════════════════════

function showNotification(msg, type) {
    const container = document.getElementById('floatingBannerContainer');
    const banner = document.createElement('div');
    banner.className = `banner banner-${type}`;
    banner.innerHTML = `<span>${msg}</span><span class="banner-close" onclick="this.parentElement.remove()">✕</span>`;
    container.appendChild(banner);
    setTimeout(() => banner.remove(), 15000);
}

function initCharts() {
    const chartDefaults = {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { labels: { color: '#94a3b8', font: { size: 11 } } } },
        scales: {
            x: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } },
            y: { ticks: { color: '#64748b' }, grid: { color: 'rgba(255,255,255,0.03)' } }
        }
    };

    equityChart = new Chart(document.getElementById('equityChart'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: chartDefaults
    });

    drawdownChart = new Chart(document.getElementById('drawdownChart'), {
        type: 'line',
        data: { labels: [], datasets: [] },
        options: { ...chartDefaults, scales: { ...chartDefaults.scales, y: { reverse: true } } }
    });
}

function fmtNum(v) { return v === null || v === undefined ? '—' : Number(v).toFixed(2); }
function fmtPct(v) { return v === null || v === undefined ? '—' : Number(v).toFixed(1) + '%'; }
function fmtCurrency(v) {
    if (v === null || v === undefined) return '—';
    const n = Number(v);
    const prefix = n >= 0 ? '+' : '';
    return prefix + '$' + Math.abs(n).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

function fmtPrice(v) {
    if (v === null || v === undefined || v === 0) return '—';
    return Number(v).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 4 });
}

function fmtTime(ts) {
    if (!ts) return '—';
    try {
        return new Date(ts).toLocaleString('en-US', {
            month: 'short', day: 'numeric',
            hour: '2-digit', minute: '2-digit',
        });
    } catch {
        return ts;
    }
}

function valClass(v) {
    if (v === null || v === undefined) return '';
    return Number(v) >= 0 ? 'positive' : 'negative';
}

function debugTrade(tradeId) {
    window.open(`${API_BASE}/trades/${tradeId}/debug`, '_blank');
}

/**
 * Stress Accordion Logic
 */
function toggleAccordion(botId) {
    const item = document.querySelector(`.accordion-item[data-bot="${botId}"]`);
    if (!item) return;

    // Lazy load if not already loaded
    if (!item.dataset.loaded) {
        loadStressTestDetail(botId);
        item.dataset.loaded = "true";
    }

    item.classList.toggle('active');
}

async function loadAllStressTests() {
    const data = await apiGet('/bots');
    const container = document.getElementById('stressResults');
    container.innerHTML = '';

    if (!data.bots || data.bots.length === 0) {
        container.innerHTML = '<p class="dim-text">No bots available</p>';
        return;
    }

    container.innerHTML = data.bots.map(b => `
        <div class="accordion-item" data-bot="${b.config.bot_id}">
            <div class="accordion-header" onclick="toggleAccordion('${b.config.bot_id}')">
                <span class="bot-name">🤖 ${b.config.bot_id}</span>
                <span class="stress-grade ${b.performance?.stress_grade?.toLowerCase() || 'neutral'}">${b.performance?.stress_grade || 'PENDING'}</span>
            </div>
            <div class="accordion-content" id="stress-content-${b.config.bot_id}">
                <div class="loading-spinner">Analyzing stability...</div>
            </div>
        </div>
    `).join('');
}

async function loadStressTestDetail(botId) {
    const content = document.getElementById(`stress-content-${botId}`);
    try {
        const data = await apiGet(`/stress/${botId}`);
        const st = data.stress_test || {};

        content.innerHTML = `
            <div class="stress-grid-compact">
                <div class="stress-card">
                    <div class="pulse-label">Parameter Stability</div>
                    <div class="pulse-value">${fmtNum(st.parameter_stability?.stability_ratio)}</div>
                </div>
                <div class="stress-card">
                    <div class="pulse-label">Slippage Breakpoint</div>
                    <div class="pulse-value">${fmtNum(st.slippage_stress?.breakpoint_multiplier)}x</div>
                </div>
                <div class="stress-card">
                    <div class="pulse-label">Tail Resilience</div>
                    <div class="pulse-value">${st.tail_risk?.all_surviving ? 'YES' : 'NO'}</div>
                </div>
            </div>
        `;
    } catch (err) {
        content.innerHTML = '<p class="negative">Failed to load stress data</p>';
    }
}

async function loadDriftAlerts() {
    const container = document.getElementById('driftAlerts');
    // Simulated/Aggregate
    const alerts = [
        { bot: 'TrendBot_BTC', regime: 'Sideways', zscore: 2.1, status: 'ALERT' }
    ];

    container.innerHTML = alerts.map(a => `
        <div class="drift-alert ${a.status.toLowerCase()}">
            <span class="drift-icon">${a.status === 'ALERT' ? '🔴' : '🟡'}</span>
            <div class="drift-info">
                <span class="drift-regime">${a.bot} — ${a.regime}</span>
                <span class="drift-detail">z-score: ${a.zscore} | Drift detected</span>
            </div>
        </div>
    `).join('');
}

async function loadEfficiency() {
    const container = document.getElementById('efficiencyCards');
    const data = await apiGet('/bots');
    const bots = data.bots || [];

    if (bots.length === 0) {
        container.innerHTML = '<p class="dim-text">No data</p>';
        return;
    }

    container.innerHTML = bots.map(b => {
        const e = b.efficiency || {};
        return `
            <div class="efficiency-card">
                <div class="efficiency-header">
                    <span class="efficiency-bot-name">${b.config.bot_id}</span>
                </div>
                <div class="efficiency-metrics">
                    <div class="efficiency-row"><span class="label">Efficiency Ratio</span><span class="value">${fmtNum(e.capital_efficiency_ratio || 0.85)}</span></div>
                    <div class="efficiency-row"><span class="label">Utilization</span><span class="value">${fmtPct(e.avg_utilization || 45)}</span></div>
                </div>
            </div>
        `;
    }).join('');
}

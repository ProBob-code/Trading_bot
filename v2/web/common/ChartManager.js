/**
 * ChartManager.js
 * Unified Chart Manager for GodBotTrade
 *
 * Chooses a renderer per symbol and never ends in an error state:
 *   - TradingView, when the backend confirms TradingView actually carries it
 *   - GoatBot's own canvas chart (NativeChart), drawn from /api/v2/candles,
 *     for everything else
 *
 * The old version guessed an exchange prefix from a hardcoded list and handed it
 * straight to the widget. When a pair was renamed on-exchange (MATIC -> POL) the
 * guess went stale, TradingView answered "This symbol doesn't exist" inside a
 * cross-origin iframe we cannot read or override, and the user was stuck looking
 * at a dead panel for a coin we could still price and chart perfectly well.
 */

const ChartManager = {
    widget: null,
    containerId: null,
    currentSymbol: null,     // TradingView-prefixed, e.g. BINANCE:POLUSDT
    rawSymbol: null,         // as the app knows it, e.g. MATICUSDT
    currentInterval: null,
    mode: 'tradingview',     // 'tradingview' | 'native'
    resolution: null,        // last /api/v2/symbol payload
    _retryTimer: null,
    _retryAttempt: 0,

    // ── User-customisable chart preferences (persisted in localStorage) ──
    // TradingView "style" codes for the main series.
    CHART_TYPES: [
        { id: '1',  label: 'Candles' },
        { id: '9',  label: 'Hollow Candles' },
        { id: '8',  label: 'Heikin Ashi' },
        { id: '0',  label: 'Bars (OHLC)' },
        { id: '11', label: 'Hi-Lo' },
        { id: '2',  label: 'Line' },
        { id: '14', label: 'Step Line' },
        { id: '13', label: 'Line + Markers' },
        { id: '3',  label: 'Area' },
        { id: '15', label: 'HLC Area' },
        { id: '10', label: 'Baseline' },
        { id: '12', label: 'Columns' },
        { id: '4',  label: 'Renko' },
        { id: '5',  label: 'Kagi' },
        { id: '6',  label: 'Point & Figure' },
        { id: '7',  label: 'Line Break' }
    ],
    // `pane: true` studies get their own sub-pane and eat vertical space —
    // that is what made the small/mobile chart look cramped, so they are OFF
    // by default and the user opts in.
    INDICATORS: [
        { id: 'MASimple@tv-basicstudies', label: 'Moving Average', pane: false },
        { id: 'BB@tv-basicstudies',       label: 'Bollinger Bands', pane: false },
        { id: 'VWAP@tv-basicstudies',     label: 'VWAP',            pane: false },
        { id: 'Volume@tv-basicstudies',   label: 'Volume',          pane: true  },
        { id: 'RSI@tv-basicstudies',      label: 'RSI',             pane: true  },
        { id: 'MACD@tv-basicstudies',     label: 'MACD',            pane: true  }
    ],
    PREFS_KEY: 'goatbot_chart_prefs',
    chartType: '1',
    studies: ['MASimple@tv-basicstudies'],

    /** Load saved chart preferences (chart type + indicators). */
    loadPrefs() {
        try {
            const raw = localStorage.getItem(this.PREFS_KEY);
            if (raw) {
                const p = JSON.parse(raw);
                if (p.chartType) this.chartType = String(p.chartType);
                if (Array.isArray(p.studies)) this.studies = p.studies;
            }
        } catch (e) {
            console.warn('[ChartManager] Could not load chart prefs:', e);
        }
    },

    savePrefs() {
        try {
            localStorage.setItem(this.PREFS_KEY, JSON.stringify({
                chartType: this.chartType,
                studies: this.studies
            }));
        } catch (e) {
            console.warn('[ChartManager] Could not save chart prefs:', e);
        }
    },

    /** Build the full widget config — single source of truth for init + rebuilds. */
    _buildConfig(theme) {
        theme = theme || document.documentElement.getAttribute('data-theme') || 'dark';
        const dark = theme === 'dark';
        return {
            "width": "100%",
            "height": "100%",
            "symbol": this.currentSymbol,
            "interval": this.currentInterval,
            "timezone": "Etc/UTC",
            "theme": theme,
            "style": this.chartType,
            "locale": "en",
            "toolbar_bg": dark ? "#0f172a" : "#f1f3f6",
            "enable_publishing": false,
            "hide_side_toolbar": true,
            "allow_symbol_change": true,
            "container_id": this.containerId,
            "studies": this.studies.slice(),
            "autosize": true,
            "hide_top_toolbar": true,
            "hide_legend": true,
            "disabled_features": ["header_widget", "left_toolbar", "legend_widget", "timeframes_toolbar"],
            "enabled_features": ["study_templates"],
            "overrides": {
                "mainSeriesProperties.candleStyle.upColor": "#00FFB2",
                "mainSeriesProperties.candleStyle.downColor": "#FF3366",
                "mainSeriesProperties.candleStyle.drawWick": true,
                "mainSeriesProperties.candleStyle.drawBorder": true,
                "mainSeriesProperties.candleStyle.borderUpColor": "#00FFB2",
                "mainSeriesProperties.candleStyle.borderDownColor": "#FF3366",
                "mainSeriesProperties.candleStyle.wickUpColor": "#00FFB2",
                "mainSeriesProperties.candleStyle.wickDownColor": "#FF3366",
                "mainSeriesProperties.hollowCandleStyle.upColor": "#00FFB2",
                "mainSeriesProperties.hollowCandleStyle.downColor": "#FF3366",
                "mainSeriesProperties.haStyle.upColor": "#00FFB2",
                "mainSeriesProperties.haStyle.downColor": "#FF3366",
                "mainSeriesProperties.barStyle.upColor": "#00FFB2",
                "mainSeriesProperties.barStyle.downColor": "#FF3366",
                "mainSeriesProperties.lineStyle.color": "#00FFB2",
                "mainSeriesProperties.areaStyle.color1": "rgba(0, 255, 178, 0.3)",
                "mainSeriesProperties.areaStyle.color2": "rgba(0, 255, 178, 0.02)",
                "mainSeriesProperties.areaStyle.linecolor": "#00FFB2",
                "paneProperties.background": dark ? "#0d1420" : "#ffffff",
                "paneProperties.vertGridProperties.color": dark ? "rgba(42, 46, 57, 0.15)" : "rgba(240, 243, 250, 0.15)",
                "paneProperties.horzGridProperties.color": dark ? "rgba(42, 46, 57, 0.15)" : "rgba(240, 243, 250, 0.15)",
                "scalesProperties.textColor": dark ? "#94A3B8" : "#475569",
                "scalesProperties.lineColor": dark ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)",
                "paneProperties.crossHairProperties.color": dark ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.2)"
            }
        };
    },

    /** Rebuild in place (TradingView's embed API has no live setters). */
    _recreate() {
        if (!this.containerId) return;
        return this._draw();
    },

    /** Change the main series style (candles / line / Heikin Ashi / …). */
    setChartType(styleId) {
        this.chartType = String(styleId);
        this.savePrefs();
        console.log(`[ChartManager] Chart type → ${this.chartType}`);
        this._recreate();
    },

    /** Replace the active indicator set. */
    setStudies(studyIds) {
        this.studies = Array.isArray(studyIds) ? studyIds.slice() : [];
        this.savePrefs();
        console.log(`[ChartManager] Studies → ${this.studies.join(', ') || 'none'}`);
        this._recreate();
    },

    /**
     * Initialize the chart.
     * @param {string} containerId - The ID of the div container
     * @param {string} symbol - Initial symbol (e.g., BTCUSDT)
     * @param {string} interval - UI interval (1m, 5m, 1h, 1d)
     */
    init(containerId, symbol, interval) {
        this.containerId = containerId;
        this.rawSymbol = String(symbol || 'BTCUSDT').toUpperCase();
        this.currentSymbol = this.formatSymbol(symbol);
        this.currentInterval = this.formatInterval(interval);
        this.loadPrefs();
        return this._draw();
    },

    /**
     * Decide who draws this symbol, then draw it.
     *
     * TradingView is only asked when the backend has confirmed it carries the
     * symbol. Otherwise GoatBot renders its own candles. There is no path out of
     * this function that leaves the user looking at an error.
     */
    async _draw() {
        const container = document.getElementById(this.containerId);
        if (!container) return false;

        const resolution = await this.resolve(this.rawSymbol);
        this.resolution = resolution;

        if (resolution && resolution.symbol && resolution.symbol !== this.rawSymbol) {
            // A migrated ticker (MATIC -> POL) charts under its live name.
            console.log(`[ChartManager] ${this.rawSymbol} resolved to ${resolution.symbol}`);
        }

        const tvSymbol = (resolution && resolution.tv_symbol) || this.formatSymbol(this.rawSymbol);
        const tvUsable = typeof TradingView !== 'undefined'
            && (!resolution || resolution.chart === 'tradingview');

        if (tvUsable) {
            this.currentSymbol = tvSymbol;
            this.mode = 'tradingview';
            if (window.NativeChart) NativeChart.destroy();
            try {
                container.innerHTML = '';
                this.widget = new TradingView.widget(this._buildConfig());
                console.log(`[ChartManager] TradingView chart for ${tvSymbol}`);
                return true;
            } catch (e) {
                console.error('[ChartManager] TradingView widget failed, falling back:', e);
            }
        }

        // Either TradingView cannot serve this instrument or its library never
        // loaded. Draw from our own candles rather than showing a dead panel.
        this.mode = 'native';
        this.widget = null;
        const drawn = await NativeChart.render(
            this.containerId,
            (resolution && resolution.symbol) || this.rawSymbol,
            this._uiInterval(this.currentInterval),
            { note: resolution && resolution.note, source: null }
        );
        if (!drawn) this._scheduleRetry();
        return drawn;
    },

    /**
     * Ask the backend what this ticker really is and who can chart it.
     * A failed lookup returns null, which keeps the TradingView-first behaviour —
     * an unreachable backend must not downgrade a chart that would have worked.
     */
    async resolve(symbol) {
        try {
            const tvGuess = this.formatSymbol(symbol);
            const res = await fetch(
                `/api/v2/symbol/${encodeURIComponent(symbol)}?tv=${encodeURIComponent(tvGuess)}`);
            if (!res.ok) return null;
            const data = await res.json();
            return data.success ? data : null;
        } catch (e) {
            console.warn('[ChartManager] symbol resolve failed:', e);
            return null;
        }
    },

    /**
     * A feed that is briefly down is not a permanent "no". Keep retrying in the
     * background so the panel fills itself the moment a source responds.
     */
    _scheduleRetry() {
        if (this._retryTimer) clearTimeout(this._retryTimer);
        this._retryAttempt = Math.min((this._retryAttempt || 0) + 1, 5);
        const delay = Math.min(5000 * this._retryAttempt, 30000);
        console.warn(`[ChartManager] no data yet for ${this.rawSymbol}, retrying in ${delay}ms`);
        this._retryTimer = setTimeout(() => this._draw(), delay);
    },

    /** TradingView interval code → the UI/backend interval string. */
    _uiInterval(tv) {
        const map = { '1': '1m', '5': '5m', '15': '15m', '30': '30m',
                      '60': '1h', '240': '4h', 'D': '1d' };
        return map[String(tv)] || '1m';
    },

    /** Feed a live tick to whichever renderer is active. */
    onTick(symbol, price) {
        if (this.mode !== 'native' || !window.NativeChart) return;
        const active = (this.resolution && this.resolution.symbol) || this.rawSymbol;
        if (String(symbol).toUpperCase() === String(active).toUpperCase()) {
            NativeChart.applyTick(price);
        }
    },

    /**
     * Update the symbol and/or interval.
     * @param {string} symbol - New symbol
     * @param {string} interval - New interval (optional)
     */
    setSymbol(symbol, interval = null) {
        if (!this.containerId) return;

        // Guard on containerId, not on `this.widget`: in native mode there is no
        // TradingView widget, and the old check made the chart unswitchable once
        // a symbol had fallen back to our own renderer.
        if (this._retryTimer) { clearTimeout(this._retryTimer); this._retryTimer = null; }
        this._retryAttempt = 0;

        this.rawSymbol = String(symbol || '').toUpperCase();
        this.currentSymbol = this.formatSymbol(symbol);
        if (interval) this.currentInterval = this.formatInterval(interval);

        console.log(`[ChartManager] Switching to ${this.rawSymbol} (${this.currentInterval})`);
        return this._draw();
    },

    /**
     * Update only the interval.
     * @param {string} interval - New interval (1, 5, 15, 60, D)
     */
    setInterval(interval) {
        if (!this.containerId) return;
        this.currentInterval = this.formatInterval(interval);
        console.log(`[ChartManager] Changing interval to ${this.currentInterval}`);
        return this._draw();
    },

    // Tickers renamed on-exchange. The backend registry is the authority; this
    // copy just stops the very first guess being wrong.
    RENAMES: {
        'MATICUSDT': 'POLUSDT', 'MATIC': 'POL',
        'LUNAUSDT': 'LUNCUSDT', 'LUNA': 'LUNC',
        'FTMUSDT': 'SUSDT', 'FTM': 'S'
    },

    /**
     * Format symbol to TradingView standard matching our backend logic.
     * This is a *guess*; /api/v2/symbol validates it before it reaches the widget.
     * @param {string} symbol - Raw symbol from UI (e.g., BTCUSDT or BTC/USDT)
     * @returns {string} - Formatted symbol (e.g., BINANCE:BTCUSDT)
     */
    formatSymbol(symbol) {
        if (!symbol) return "BINANCE:BTCUSDT";

        let cleanSymbol = symbol.toUpperCase().replace('/', '');
        cleanSymbol = this.RENAMES[cleanSymbol] || cleanSymbol;

        // Already has exchange prefix?
        if (cleanSymbol.includes(':')) return cleanSymbol;

        // ── Indian NSE Stocks (Nifty 50 + popular) ──
        const nseStocks = [
            'RELIANCE', 'TCS', 'INFY', 'HDFCBANK', 'ICICIBANK', 'SBIN',
            'BHARTIARTL', 'ITC', 'HINDUNILVR', 'WIPRO', 'TATAMOTORS',
            'ADANIENT', 'LT', 'BAJFINANCE', 'MARUTI', 'AXISBANK', 'TITAN',
            'SUNPHARMA', 'ASIANPAINT', 'NESTLEIND', 'TATASTEEL', 'HCLTECH',
            'POWERGRID', 'COALINDIA', 'ONGC', 'NTPC', 'BAJAJFINSV',
            'ADANIPORTS', 'ULTRACEMCO', 'JSWSTEEL', 'TECHM', 'INDUSINDBK',
            'HINDALCO', 'DRREDDY', 'CIPLA', 'EICHERMOT', 'DIVISLAB',
            'BPCL', 'GRASIM', 'APOLLOHOSP', 'HEROMOTOCO', 'TATACONSUM',
            'SBILIFE', 'HDFCLIFE', 'BRITANNIA', 'KOTAKBANK', 'BAJAJ_AUTO'
        ];
        if (nseStocks.includes(cleanSymbol)) {
            return `NSE:${cleanSymbol}`;
        }

        // NSE indices are not equities — they live under different tickers.
        const indianIndices = {
            'NIFTY': 'NSE:NIFTY',
            'NIFTY50': 'NSE:NIFTY',
            'BANKNIFTY': 'NSE:BANKNIFTY',
            'FINNIFTY': 'NSE:CNXFINANCE',
            'SENSEX': 'BSE:SENSEX'
        };
        if (indianIndices[cleanSymbol]) {
            return indianIndices[cleanSymbol];
        }

        // ── NASDAQ Stocks ──
        const nasdaq = [
            'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'META', 'NVDA',
            'TSLA', 'AMD', 'NFLX', 'INTC', 'UBER', 'PYPL', 'QCOM',
            'ADBE', 'CSCO', 'AVGO', 'TXN', 'COST', 'PEP', 'SBUX',
            'ABNB', 'MRVL', 'MU', 'LRCX', 'AMAT', 'KLAC', 'SNPS',
            'CDNS', 'PANW', 'CRWD', 'ZS', 'DDOG', 'NET', 'SNOW',
            'COIN', 'MSTR', 'PLTR', 'ARM', 'SMCI'
        ];
        if (nasdaq.includes(cleanSymbol)) {
            return `NASDAQ:${cleanSymbol}`;
        }

        // ── NYSE Stocks ──
        const nyse = [
            'BRK.B', 'KO', 'DIS', 'V', 'JPM', 'JNJ', 'WMT', 'PG',
            'MA', 'UNH', 'HD', 'BAC', 'XOM', 'CVX', 'LLY', 'ABBV',
            'MRK', 'PFE', 'TMO', 'ABT', 'ORCL', 'CRM', 'ACN', 'IBM',
            'GE', 'CAT', 'BA', 'RTX', 'GS', 'MS', 'BLK', 'SCHW', 'C'
        ];
        if (nyse.includes(cleanSymbol)) {
            return `NYSE:${cleanSymbol}`;
        }

        // ── Forex Pairs ──
        const forexPairs = [
            'EURUSD', 'GBPUSD', 'USDJPY', 'USDCHF', 'AUDUSD', 'USDCAD',
            'NZDUSD', 'EURGBP', 'EURJPY', 'GBPJPY', 'USDINR', 'GBPINR',
            'EURINR', 'JPYINR', 'EURCHF', 'AUDNZD', 'AUDCAD', 'CADJPY'
        ];
        if (forexPairs.includes(cleanSymbol)) {
            return `FX:${cleanSymbol}`;
        }

        // ── Commodities ──
        const commodityMap = {
            'XAUUSD': 'TVC:GOLD',
            'XAGUSD': 'TVC:SILVER',
            'XCUUSD': 'COMEX:HG1!',
            'XBRUSD': 'TVC:UKOIL',
            'XTIUSD': 'TVC:USOIL',
            'XNGUSD': 'NYMEX:NG1!',
            'GOLD': 'TVC:GOLD',
            'SILVER': 'TVC:SILVER',
            'COPPER': 'COMEX:HG1!',
            'PLATINUM': 'TVC:PLATINUM',
            'PALLADIUM': 'TVC:PALLADIUM'
        };
        if (commodityMap[cleanSymbol]) {
            return commodityMap[cleanSymbol];
        }

        // ── Crypto (ends with USDT/BUSD/USDC) → Binance ──
        if (cleanSymbol.endsWith('USDT') || cleanSymbol.endsWith('BUSD') || cleanSymbol.endsWith('USDC')) {
            return `BINANCE:${cleanSymbol}`;
        }

        // ── Indices ──
        const indexMap = {
            'SPX': 'SP:SPX',
            'DJI': 'DJ:DJI',
            'IXIC': 'NASDAQ:IXIC',
            'VIX': 'CBOE:VIX'
        };
        if (indexMap[cleanSymbol]) {
            return indexMap[cleanSymbol];
        }

        // ── Fallback: short alpha-only symbols are most likely US equities ──
        if (cleanSymbol.length <= 6 && /^[A-Z]+$/.test(cleanSymbol)) {
            return `NASDAQ:${cleanSymbol}`;
        }

        return `BINANCE:${cleanSymbol}`;
    },

    /**
     * Convert UI intervals (1m, 5m, 1h, 1d) to TradingView intervals (1, 5, 60, D)
     * @param {string|number} interval - Interval from UI
     * @returns {string} - TV compatible interval
     */
    formatInterval(interval) {
        const i = interval.toString().toLowerCase();
        if (i === '1m') return '1';
        if (i === '5m') return '5';
        if (i === '15m') return '15';
        if (i === '30m') return '30';
        if (i === '1h') return '60';
        if (i === '4h') return '240';
        if (i === '1d') return 'D';
        return i;
    }
};

window.ChartManager = ChartManager;

/**
 * ChartManager.js
 * Unified TradingView Chart Manager for GodBotTrade
 */

const ChartManager = {
    widget: null,
    containerId: null,
    currentSymbol: null,
    currentInterval: null,

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

    /** Rebuild the widget in place (TradingView's embed API has no live setters). */
    _recreate() {
        if (!this.containerId) return;
        const container = document.getElementById(this.containerId);
        if (container) container.innerHTML = '';
        try {
            this.widget = new TradingView.widget(this._buildConfig());
        } catch (e) {
            console.error('[ChartManager] ❌ Exception during widget rebuild:', e);
        }
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
     * Initialize the TradingView Advanced Chart Widget
     * @param {string} containerId - The ID of the div container
     * @param {string} symbol - Initial symbol (e.g., BTCUSDT)
     */
    init(containerId, symbol, interval) {
        this.containerId = containerId;
        this.currentSymbol = this.formatSymbol(symbol);
        this.currentInterval = this.formatInterval(interval);
        this.loadPrefs();

        if (typeof TradingView === 'undefined') {
            console.error('[ChartManager] ❌ ERROR: TradingView library (tv.js) not loaded!');
            const container = document.getElementById(containerId);
            if (container) {
                container.innerHTML = 
                    '<div style="color: #ef4444; padding: 20px; text-align: center;">' +
                    '❌ TradingView Library Error<br><small>Check internet connection or script source.</small></div>';
            }
            return;
        }

        console.log(`[ChartManager] 🚀 Initializing for ${this.currentSymbol} (${this.currentInterval}) on #${containerId}`);

        try {
            this.widget = new TradingView.widget(this._buildConfig());
            console.log("[ChartManager] ✅ Widget constructor called successfully");
        } catch (e) {
            console.error("[ChartManager] ❌ Exception during widget creation:", e);
        }
    },

    /**
     * Update the symbol and/or interval
     * @param {string} symbol - New symbol
     * @param {string} interval - New interval (optional)
     */
    setSymbol(symbol, interval = null) {
        if (!this.widget) return;

        this.currentSymbol = this.formatSymbol(symbol);
        if (interval) {
            this.currentInterval = this.formatInterval(interval);
        }

        console.log(`[ChartManager] Switching to ${this.currentSymbol} (${this.currentInterval})`);
        
        // Re-create widget with new symbol (TradingView widget.setSymbol doesn't exist on the embed API)
        const container = document.getElementById(this.containerId);
        if (container) {
            container.innerHTML = '';
        }
        
        const theme = document.documentElement.getAttribute('data-theme') || 'dark';
        
        try {
            this.widget = new TradingView.widget(this._buildConfig(theme));
            console.log(`[ChartManager] ✅ Widget re-created for ${this.currentSymbol}`);
        } catch (e) {
            console.error("[ChartManager] ❌ Exception during widget re-creation:", e);
        }
    },

    /**
     * Update only the interval
     * @param {string} interval - New interval (1, 5, 15, 60, D)
     */
    setInterval(interval) {
        if (!this.widget) return;

        this.currentInterval = this.formatInterval(interval);
        console.log(`[ChartManager] Changing interval to ${this.currentInterval}`);
        // Re-create with current symbol and new interval
        this.setSymbol(this.currentSymbol.split(':').pop(), this.currentInterval);
    },

    /**
     * Format symbol to TradingView standard matching our backend logic
     * @param {string} symbol - Raw symbol from UI (e.g., BTCUSDT or BTC/USDT)
     * @returns {string} - Formatted symbol (e.g., BINANCE:BTCUSDT)
     */
    formatSymbol(symbol) {
        if (!symbol) return "BINANCE:BTCUSDT";

        let cleanSymbol = symbol.toUpperCase().replace('/', '');

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
            'SBILIFE', 'HDFCLIFE', 'BRITANNIA', 'KOTAKBANK', 'BAJAJ_AUTO',
            'NIFTY', 'BANKNIFTY', 'SENSEX'
        ];
        if (nseStocks.includes(cleanSymbol)) {
            return `NSE:${cleanSymbol}`;
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
            'GOLD': 'TVC:GOLD',
            'SILVER': 'TVC:SILVER',
            'XCUUSD': 'COMEX:HG1!',
            'XBRUSD': 'TVC:UKOIL',
            'XTIUSD': 'TVC:USOIL',
            'XNGUSD': 'NYMEX:NG1!',
            'COPPER': 'COMEX:HG1!',
            'PLATINUM': 'TVC:PLATINUM',
            'PALLADIUM': 'TVC:PALLADIUM'
        };
        if (commodityMap[cleanSymbol]) {
            return commodityMap[cleanSymbol];
        }

        // ── Crypto (ends with USDT/USD/BTC/ETH) → Binance ──
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

        // ── Fallback: try BINANCE for crypto-like symbols, otherwise plain ──
        if (cleanSymbol.length <= 6 && /^[A-Z]+$/.test(cleanSymbol)) {
            // Short alpha-only symbols → likely stocks, try NASDAQ
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

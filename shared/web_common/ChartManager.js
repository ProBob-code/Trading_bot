/**
 * ChartManager.js
 * Unified TradingView Chart Manager for GodBotTrade
 */

const ChartManager = {
    widget: null,
    containerId: null,
    currentSymbol: null,
    currentInterval: null,

    /**
     * Initialize the TradingView Advanced Chart Widget
     * @param {string} containerId - The ID of the div container
     * @param {string} symbol - Initial symbol (e.g., BTCUSDT)
     */
    init(containerId, symbol, interval) {
        this.containerId = containerId;
        this.currentSymbol = this.formatSymbol(symbol);
        this.currentInterval = this.formatInterval(interval);
        
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
            this.widget = new TradingView.widget({
            "width": "100%",
            "height": "100%",
            "symbol": this.currentSymbol,
            "interval": this.currentInterval,
            "timezone": "Etc/UTC",
            "theme": "dark",
            "style": "1",
            "locale": "en",
            "toolbar_bg": "#0f172a",
            "enable_publishing": false,
            "hide_side_toolbar": true,
            "allow_symbol_change": true,
            "container_id": containerId,
            "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
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
                "paneProperties.background": "#0d1420",
                "paneProperties.vertGridProperties.color": "rgba(42, 46, 57, 0.15)",
                "paneProperties.horzGridProperties.color": "rgba(42, 46, 57, 0.15)",
                "scalesProperties.textColor": "#94A3B8",
                "scalesProperties.lineColor": "rgba(255, 255, 255, 0.05)",
                "paneProperties.crossHairProperties.color": "rgba(255, 255, 255, 0.2)"
            }
        });
        
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
            this.widget = new TradingView.widget({
                "width": "100%",
                "height": "100%",
                "symbol": this.currentSymbol,
                "interval": this.currentInterval,
                "timezone": "Etc/UTC",
                "theme": theme,
                "style": "1",
                "locale": "en",
                "toolbar_bg": theme === 'dark' ? "#0f172a" : "#f1f3f6",
                "enable_publishing": false,
                "hide_side_toolbar": true,
                "allow_symbol_change": true,
                "container_id": this.containerId,
                "studies": ["MASimple@tv-basicstudies", "RSI@tv-basicstudies"],
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
                    "paneProperties.background": theme === 'dark' ? "#0d1420" : "#ffffff",
                    "paneProperties.vertGridProperties.color": theme === 'dark' ? "rgba(42, 46, 57, 0.15)" : "rgba(240, 243, 250, 0.15)",
                    "paneProperties.horzGridProperties.color": theme === 'dark' ? "rgba(42, 46, 57, 0.15)" : "rgba(240, 243, 250, 0.15)",
                    "scalesProperties.textColor": theme === 'dark' ? "#94A3B8" : "#475569",
                    "scalesProperties.lineColor": theme === 'dark' ? "rgba(255, 255, 255, 0.05)" : "rgba(0, 0, 0, 0.05)",
                    "paneProperties.crossHairProperties.color": theme === 'dark' ? "rgba(255, 255, 255, 0.2)" : "rgba(0, 0, 0, 0.2)"
                }
            });
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
        if (i === '1h') return '60';
        if (i === '1d') return 'D';
        return i;
    }
};

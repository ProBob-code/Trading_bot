/**
 * NativeChart.js
 * GoatBot's own candlestick renderer — the reason the chart panel is never empty.
 * ==============================================================================
 *
 * TradingView refuses symbols it does not carry, and its answer ("This symbol
 * doesn't exist") is rendered inside a cross-origin iframe we cannot read or
 * override. So when the symbol registry tells us TradingView will not serve an
 * instrument, we stop asking it and draw the candles ourselves from
 * /api/v2/candles — which walks Binance, then Yahoo.
 *
 * Deliberately zero-dependency: a plain <canvas>. A charting library loaded from
 * a CDN would be one more thing that can fail, and this code path exists
 * precisely for when other things have failed.
 */

const NativeChart = {
    containerId: null,
    symbol: null,
    interval: '1m',
    candles: [],
    _canvas: null,
    _ctx: null,
    _resizeObs: null,
    _hover: null,
    _refreshTimer: null,
    _meta: {},

    // Fraction of the plot given to the volume histogram at the bottom.
    VOLUME_RATIO: 0.18,
    PAD: { top: 16, right: 68, bottom: 26, left: 10 },

    /** Read the live theme so the chart matches the rest of the terminal. */
    _theme() {
        const css = getComputedStyle(document.documentElement);
        const pick = (name, fallback) => (css.getPropertyValue(name) || '').trim() || fallback;
        const dark = (document.documentElement.getAttribute('data-theme') || 'dark') === 'dark';
        return {
            dark,
            up: pick('--bullish', '#00FFB2'),
            down: pick('--bearish', '#FF3366'),
            text: pick('--text-dim', dark ? '#94A3B8' : '#475569'),
            grid: dark ? 'rgba(255,255,255,0.05)' : 'rgba(15,23,42,0.06)',
            crosshair: dark ? 'rgba(255,255,255,0.28)' : 'rgba(15,23,42,0.28)',
            bg: pick('--panel-bg', dark ? '#0d1420' : '#ffffff'),
            accent: pick('--accent', '#00FFB2')
        };
    },

    /**
     * Draw `symbol` into `containerId`.
     * Returns true once candles are on screen, false if every source came back
     * empty — the caller uses that to decide what to tell the user.
     */
    async render(containerId, symbol, interval, meta = {}) {
        this.containerId = containerId;
        this.symbol = symbol;
        this.interval = interval || '1m';
        this._meta = meta || {};

        const container = document.getElementById(containerId);
        if (!container) return false;

        this._mount(container);
        this._drawMessage('Loading ' + symbol + '…');

        const ok = await this.loadCandles();
        if (!ok) {
            this._drawMessage('Reconnecting to the data feed…');
            return false;
        }

        this.draw();
        this._startRefresh();
        return true;
    },

    /** Pull candles from the backend cascade (Binance → Yahoo). */
    async loadCandles() {
        try {
            const url = `/api/v2/candles/${encodeURIComponent(this.symbol)}`
                + `?interval=${encodeURIComponent(this.interval)}&limit=300`;
            const res = await fetch(url);
            const data = await res.json();
            if (data.success && Array.isArray(data.candles) && data.candles.length) {
                // A null OHLC value from a provider gap would break the scale maths.
                this.candles = data.candles.filter(c =>
                    [c.open, c.high, c.low, c.close].every(v => typeof v === 'number' && isFinite(v)));
                this._meta.source = data.source;
                this._meta.note = data.note || this._meta.note;
                return this.candles.length > 0;
            }
        } catch (err) {
            console.warn('[NativeChart] candle fetch failed:', err);
        }
        return false;
    },

    /**
     * Fold a live tick into the newest candle so the native chart keeps moving
     * between refreshes, the way the TradingView one does.
     */
    applyTick(price) {
        if (!this.candles.length || !isFinite(price)) return;
        const last = this.candles[this.candles.length - 1];
        last.close = price;
        if (price > last.high) last.high = price;
        if (price < last.low) last.low = price;
        this.draw();
    },

    _startRefresh() {
        if (this._refreshTimer) clearInterval(this._refreshTimer);
        // Intraday candles roll over quickly; daily ones do not need the churn.
        const ms = this.interval === '1d' ? 300000 : 60000;
        this._refreshTimer = setInterval(async () => {
            if (await this.loadCandles()) this.draw();
        }, ms);
    },

    destroy() {
        if (this._refreshTimer) clearInterval(this._refreshTimer);
        this._refreshTimer = null;
        if (this._resizeObs) this._resizeObs.disconnect();
        this._resizeObs = null;
        this._canvas = null;
        this._ctx = null;
    },

    // ------------------------------------------------------------------
    // Canvas plumbing
    // ------------------------------------------------------------------
    _mount(container) {
        if (this._resizeObs) this._resizeObs.disconnect();
        container.innerHTML = '';

        const wrap = document.createElement('div');
        wrap.style.cssText = 'position:relative;width:100%;height:100%;';

        const canvas = document.createElement('canvas');
        canvas.style.cssText = 'width:100%;height:100%;display:block;cursor:crosshair;';
        wrap.appendChild(canvas);

        const badge = document.createElement('div');
        badge.className = 'native-chart-badge';
        wrap.appendChild(badge);

        const readout = document.createElement('div');
        readout.className = 'native-chart-readout';
        wrap.appendChild(readout);

        container.appendChild(wrap);

        this._canvas = canvas;
        this._ctx = canvas.getContext('2d');
        this._badge = badge;
        this._readout = readout;

        canvas.addEventListener('mousemove', (e) => {
            const rect = canvas.getBoundingClientRect();
            this._hover = { x: e.clientX - rect.left, y: e.clientY - rect.top };
            this.draw();
        });
        canvas.addEventListener('mouseleave', () => { this._hover = null; this.draw(); });

        this._resizeObs = new ResizeObserver(() => this.draw());
        this._resizeObs.observe(container);
    },

    /** Size the backing store to the device pixel ratio so text stays sharp. */
    _sizeCanvas() {
        const canvas = this._canvas;
        if (!canvas) return null;
        const dpr = window.devicePixelRatio || 1;
        const w = canvas.clientWidth, h = canvas.clientHeight;
        if (!w || !h) return null;
        if (canvas.width !== w * dpr || canvas.height !== h * dpr) {
            canvas.width = w * dpr;
            canvas.height = h * dpr;
        }
        this._ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
        return { w, h };
    },

    _drawMessage(text) {
        const size = this._sizeCanvas();
        if (!size) return;
        const t = this._theme();
        const ctx = this._ctx;
        ctx.clearRect(0, 0, size.w, size.h);
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, size.w, size.h);
        ctx.fillStyle = t.text;
        ctx.font = '13px ui-monospace, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(text, size.w / 2, size.h / 2);
    },

    // ------------------------------------------------------------------
    // Drawing
    // ------------------------------------------------------------------
    draw() {
        const size = this._sizeCanvas();
        if (!size || !this.candles.length) return;

        const { w, h } = size;
        const t = this._theme();
        const ctx = this._ctx;
        const P = this.PAD;

        const plotW = w - P.left - P.right;
        const plotH = h - P.top - P.bottom;
        const volH = plotH * this.VOLUME_RATIO;
        const priceH = plotH - volH - 8;
        if (plotW <= 0 || priceH <= 0) return;

        const candles = this.candles;
        let hi = -Infinity, lo = Infinity, maxVol = 0;
        for (const c of candles) {
            if (c.high > hi) hi = c.high;
            if (c.low < lo) lo = c.low;
            if ((c.volume || 0) > maxVol) maxVol = c.volume || 0;
        }
        // A flat series would divide by zero; give it a nominal band instead.
        if (hi === lo) { hi += hi * 0.001 || 1; lo -= lo * 0.001 || 1; }
        const span = hi - lo;
        hi += span * 0.06;
        lo -= span * 0.06;

        const yOf = p => P.top + (hi - p) / (hi - lo) * priceH;
        const step = plotW / candles.length;
        const xOf = i => P.left + i * step + step / 2;

        ctx.clearRect(0, 0, w, h);
        ctx.fillStyle = t.bg;
        ctx.fillRect(0, 0, w, h);

        // ── Grid + price axis ──
        ctx.strokeStyle = t.grid;
        ctx.fillStyle = t.text;
        ctx.lineWidth = 1;
        ctx.font = '10px ui-monospace, monospace';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        const ROWS = 5;
        for (let i = 0; i <= ROWS; i++) {
            const price = hi - (hi - lo) * (i / ROWS);
            const y = Math.round(yOf(price)) + 0.5;
            ctx.beginPath();
            ctx.moveTo(P.left, y);
            ctx.lineTo(P.left + plotW, y);
            ctx.stroke();
            ctx.fillStyle = t.text;
            ctx.fillText(this._fmtPrice(price), P.left + plotW + 8, y);
        }

        // ── Volume ──
        const volTop = P.top + priceH + 8;
        for (let i = 0; i < candles.length; i++) {
            const c = candles[i];
            const vh = maxVol ? ((c.volume || 0) / maxVol) * volH : 0;
            ctx.fillStyle = c.close >= c.open ? t.up : t.down;
            ctx.globalAlpha = 0.28;
            ctx.fillRect(xOf(i) - Math.max(step * 0.3, 0.5), volTop + volH - vh,
                Math.max(step * 0.6, 1), vh);
        }
        ctx.globalAlpha = 1;

        // ── Candles ──
        const bodyW = Math.max(step * 0.62, 1);
        for (let i = 0; i < candles.length; i++) {
            const c = candles[i];
            const x = xOf(i);
            const up = c.close >= c.open;
            ctx.strokeStyle = ctx.fillStyle = up ? t.up : t.down;

            ctx.beginPath();
            ctx.moveTo(Math.round(x) + 0.5, yOf(c.high));
            ctx.lineTo(Math.round(x) + 0.5, yOf(c.low));
            ctx.stroke();

            const yO = yOf(c.open), yC = yOf(c.close);
            // A doji has zero body height and would otherwise vanish.
            const top = Math.min(yO, yC);
            const bh = Math.max(Math.abs(yC - yO), 1);
            ctx.fillRect(x - bodyW / 2, top, bodyW, bh);
        }

        // ── Time axis ──
        ctx.fillStyle = t.text;
        ctx.textAlign = 'center';
        ctx.textBaseline = 'top';
        const labels = Math.min(6, candles.length);
        for (let i = 0; i < labels; i++) {
            const idx = Math.floor(i * (candles.length - 1) / Math.max(labels - 1, 1));
            ctx.fillText(this._fmtTime(candles[idx].time), xOf(idx), P.top + plotH + 8);
        }

        // ── Last price marker ──
        const last = candles[candles.length - 1];
        const lastY = yOf(last.close);
        const lastUp = last.close >= last.open;
        ctx.strokeStyle = lastUp ? t.up : t.down;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(P.left, Math.round(lastY) + 0.5);
        ctx.lineTo(P.left + plotW, Math.round(lastY) + 0.5);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = lastUp ? t.up : t.down;
        ctx.fillRect(P.left + plotW + 2, lastY - 8, P.right - 4, 16);
        ctx.fillStyle = t.dark ? '#08111d' : '#ffffff';
        ctx.textAlign = 'left';
        ctx.textBaseline = 'middle';
        ctx.font = 'bold 10px ui-monospace, monospace';
        ctx.fillText(this._fmtPrice(last.close), P.left + plotW + 8, lastY);

        this._drawCrosshair(t, { w, h, plotW, priceH, step, xOf, yOf, hi, lo });
        this._updateChrome(last);
    },

    _drawCrosshair(t, geo) {
        const ctx = this._ctx;
        const hover = this._hover;
        if (!hover || hover.x < this.PAD.left || hover.x > this.PAD.left + geo.plotW) {
            if (this._readout) this._readout.style.display = 'none';
            return;
        }

        const i = Math.max(0, Math.min(this.candles.length - 1,
            Math.floor((hover.x - this.PAD.left) / geo.step)));
        const c = this.candles[i];

        ctx.strokeStyle = t.crosshair;
        ctx.setLineDash([2, 3]);
        ctx.beginPath();
        ctx.moveTo(Math.round(geo.xOf(i)) + 0.5, this.PAD.top);
        ctx.lineTo(Math.round(geo.xOf(i)) + 0.5, this.PAD.top + geo.priceH);
        ctx.moveTo(this.PAD.left, Math.round(hover.y) + 0.5);
        ctx.lineTo(this.PAD.left + geo.plotW, Math.round(hover.y) + 0.5);
        ctx.stroke();
        ctx.setLineDash([]);

        if (this._readout) {
            const up = c.close >= c.open;
            this._readout.style.display = 'flex';
            this._readout.innerHTML =
                `<span>${this._fmtTime(c.time, true)}</span>` +
                `<span>O <b>${this._fmtPrice(c.open)}</b></span>` +
                `<span>H <b>${this._fmtPrice(c.high)}</b></span>` +
                `<span>L <b>${this._fmtPrice(c.low)}</b></span>` +
                `<span class="${up ? 'p-positive' : 'p-negative'}">C <b>${this._fmtPrice(c.close)}</b></span>`;
        }
    },

    /** Name the data source, so a fallback chart is never mistaken for TradingView. */
    _updateChrome(last) {
        if (!this._badge) return;
        const src = (this._meta.source || 'goatbot').toUpperCase();
        const note = this._meta.note ? ` · ${this._meta.note}` : '';
        this._badge.textContent = `GOATBOT CHART · ${this.symbol} · ${src}${note}`;
    },

    _fmtPrice(p) {
        const abs = Math.abs(p);
        // Sub-dollar instruments lose all their movement at two decimals.
        const dp = abs >= 1000 ? 2 : abs >= 1 ? 2 : abs >= 0.01 ? 4 : 6;
        return p.toLocaleString(undefined, { minimumFractionDigits: dp, maximumFractionDigits: dp });
    },

    _fmtTime(unixSeconds, withDate = false) {
        const d = new Date(unixSeconds * 1000);
        const intraday = ['1m', '5m', '15m', '30m', '1h', '4h'].includes(this.interval);
        if (intraday && !withDate) {
            return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        }
        if (intraday) {
            return d.toLocaleString([], { month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit' });
        }
        return d.toLocaleDateString([], { month: 'short', day: 'numeric' });
    }
};

window.NativeChart = NativeChart;

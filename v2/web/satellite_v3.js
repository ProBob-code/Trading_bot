/* ============================================================
   GoatBot V3 Satellite Core — shared by portfolio / live / profile
   Auth gate, navbar, fetch helpers, formatting, toasts.
   ============================================================ */

const V3 = {
    user: null,

    /** Redirect to login if not authenticated; returns profile when authed. */
    async authGate() {
        try {
            const res = await fetch('/api/auth/status');
            const data = await res.json();
            if (!data.authenticated) { window.location.href = '/godbot_login'; return null; }
            try {
                const p = await fetch('/api/user/profile');
                if (p.ok) this.user = await p.json();
            } catch (e) { /* profile optional */ }
            return this.user || {};
        } catch (err) {
            window.location.href = '/godbot_login';
            return null;
        }
    },

    /** GET json with sane failure -> null */
    async get(url) {
        try {
            const res = await fetch(url);
            if (!res.ok) return null;
            return await res.json();
        } catch (e) { return null; }
    },

    /** POST json */
    async post(url, body) {
        try {
            const res = await fetch(url, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: body ? JSON.stringify(body) : undefined,
            });
            return await res.json();
        } catch (e) { return { success: false, error: String(e) }; }
    },

    fmtUsd(v, digits = 2) {
        const n = parseFloat(v);
        if (!isFinite(n)) return '$—';
        return '$' + n.toLocaleString(undefined, { minimumFractionDigits: digits, maximumFractionDigits: digits });
    },

    fmtNum(v, digits = 4) {
        const n = parseFloat(v);
        return isFinite(n) ? n.toLocaleString(undefined, { maximumFractionDigits: digits }) : '—';
    },

    fmtPct(v, digits = 2) {
        const n = parseFloat(v);
        if (!isFinite(n)) return '—';
        return (n >= 0 ? '+' : '') + n.toFixed(digits) + '%';
    },

    pnlClass(v) { return parseFloat(v) >= 0 ? 'pos' : 'neg'; },

    esc(s) {
        return String(s ?? '').replace(/[&<>"']/g, c =>
            ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
    },

    toast(msg, ok = true) {
        let el = document.getElementById('v3Toast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'v3Toast';
            document.body.appendChild(el);
        }
        el.textContent = msg;
        el.className = ok ? 'ok' : 'err';
        requestAnimationFrame(() => el.classList.add('show'));
        clearTimeout(el._t);
        el._t = setTimeout(() => el.classList.remove('show'), 3500);
    },

    /** Render the shared top navbar. `active`: home|report|live|portfolio|profile */
    navbar(active) {
        const links = [
            ['home', '/godbot_home', 'fa-gauge-high', 'Command Deck'],
            ['report', '/v2/report', 'fa-file-lines', 'Report'],
            ['live', '/v2/live', 'fa-satellite-dish', 'Live Trade'],
            ['portfolio', '/v2/portfolio', 'fa-chart-pie', 'Portfolio'],
            ['profile', '/v2/profile', 'fa-user-gear', 'Profile'],
        ];
        const nav = document.createElement('nav');
        nav.className = 'navbar';
        nav.innerHTML = `
            <a href="/godbot_home" class="brand">
                <img src="/logo.svg" alt="GoatBot" class="brand-logo">
                <span class="brand-name">GOATBOT<span>TRADE</span></span>
            </a>
            <div class="nav-links">
                ${links.map(([id, href, icon, label]) =>
                    `<a href="${href}" class="${id === active ? 'active' : ''}">
                        <i class="fa-solid ${icon}"></i> ${label}</a>`).join('')}
                <button class="logout" id="v3LogoutBtn"><i class="fa-solid fa-right-from-bracket"></i> Logout</button>
            </div>`;
        document.body.prepend(nav);
        document.getElementById('v3LogoutBtn').addEventListener('click', async () => {
            await this.post('/api/auth/logout');
            window.location.href = '/godbot_login';
        });
    },
};

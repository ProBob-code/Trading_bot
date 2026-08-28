/**
 * nav.js — the app's single navigation drawer
 * ===========================================
 *
 * Every page gets the same nav: a floating hamburger, an off-canvas drawer,
 * and a dimming overlay. The home terminal used to be the odd one out with a
 * 72px hover-rail; it now uses this too, so the navigation is identical
 * wherever you are.
 *
 * Self-contained on purpose — it injects its own markup AND styles, because
 * the satellite pages each carry their own stylesheet and share nothing.
 */
(function () {
    'use strict';

    var ROUTES = [
        { route: 'home',      href: '/godbot_home',  icon: 'fa-home',         label: 'Home' },
        { route: 'report',    href: '/v2/report',    icon: 'fa-file-invoice', label: 'Report' },
        { route: 'paper',     href: '/godbot_home',  icon: 'fa-vial',         label: 'Paper Trade',
          scrollTo: 'sectionAutoTrade' },
        { route: 'live',      href: '/v2/live',      icon: 'fa-rocket',       label: 'Live Trade' },
        { route: 'portfolio', href: '/v2/portfolio', icon: 'fa-wallet',       label: 'Portfolio' },
        { route: 'profile',   href: '/v2/profile',   icon: 'fa-user-cog',     label: 'Profile Settings' },
        // Hidden until the server confirms this account is an administrator.
        // The gate that matters is server-side; this only avoids showing a link
        // that would 403.
        { route: 'admin',     href: '/v2/admin',     icon: 'fa-shield-halved', label: 'Admin Console',
          adminOnly: true }
    ];

    var CSS = [
        /* ── Hamburger ──────────────────────────────────────────
           Three drawn bars that morph into an X, rather than an
           icon-font glyph in a plain box. */
        '.gb-nav-toggle{position:fixed;top:14px;left:14px;z-index:6001;',
        'width:44px;height:44px;display:flex;flex-direction:column;align-items:center;',
        'justify-content:center;gap:5px;border-radius:13px;cursor:pointer;padding:0;',
        'background:var(--gb-nav-btn,rgba(15,23,42,.82));',
        'border:1px solid var(--gb-nav-border,rgba(255,255,255,.14));',
        'backdrop-filter:blur(12px);-webkit-backdrop-filter:blur(12px);',
        'box-shadow:0 8px 22px -10px rgba(0,0,0,.65);',
        'transition:border-color .2s,background .2s,transform .2s}',
        '.gb-nav-toggle:hover{border-color:#a855f7;transform:translateY(-1px)}',
        '.gb-nav-toggle:active{transform:translateY(0) scale(.96)}',
        '.gb-nav-toggle:focus-visible{outline:2px solid #a855f7;outline-offset:2px}',

        '.gb-nav-bar{display:block;width:19px;height:2px;border-radius:2px;',
        'background:var(--gb-nav-ink,#fff);',
        'transition:transform .28s cubic-bezier(.16,1,.3,1),opacity .18s ease,width .28s ease}',
        '.gb-nav-toggle.open .gb-nav-bar:nth-child(1){transform:translateY(7px) rotate(45deg)}',
        '.gb-nav-toggle.open .gb-nav-bar:nth-child(2){opacity:0;width:0}',
        '.gb-nav-toggle.open .gb-nav-bar:nth-child(3){transform:translateY(-7px) rotate(-45deg)}',

        /* Nudge each page's own header clear of the floating button. */
        '.navbar,.report-header .header-inner,.admin-header .header-inner{padding-left:70px}',

        /* ── Drawer ─────────────────────────────────────────── */
        '.gb-nav-drawer{position:fixed;top:0;left:0;bottom:0;width:248px;z-index:6002;',
        'display:flex;flex-direction:column;padding:0 0 14px;overflow-y:auto;',
        'background:var(--gb-nav-bg,#080d16);',
        'border-right:1px solid var(--gb-nav-border,rgba(255,255,255,.08));',
        'transform:translateX(-100%);transition:transform .3s cubic-bezier(.16,1,.3,1);',
        'overscroll-behavior:contain}',
        '.gb-nav-drawer.open{transform:none}',

        '.gb-nav-brand{display:flex;align-items:center;gap:11px;padding:18px 18px 16px;',
        'margin-bottom:10px;border-bottom:1px solid var(--gb-nav-border,rgba(255,255,255,.07))}',
        '.gb-nav-brand img{width:34px;height:34px;object-fit:contain;border-radius:9px;flex-shrink:0}',
        '.gb-nav-brand span{font-size:15px;font-weight:800;letter-spacing:.2px;',
        'color:var(--gb-nav-ink,#fff);white-space:nowrap}',

        '.gb-nav-items{display:flex;flex-direction:column;gap:3px;padding:0 10px}',

        '.gb-nav-item{position:relative;display:flex;align-items:center;gap:13px;',
        'padding:12px 14px;border-radius:11px;font-size:13.5px;font-weight:600;',
        'color:var(--gb-nav-muted,#94a3b8);text-decoration:none;cursor:pointer;',
        'background:none;border:none;width:100%;text-align:left;font-family:inherit;',
        'transition:background .18s,color .18s}',
        '.gb-nav-item i{width:19px;text-align:center;font-size:15px;flex-shrink:0}',
        '.gb-nav-item:hover{background:var(--gb-nav-hover,rgba(168,85,247,.12));',
        'color:var(--gb-nav-ink,#fff)}',
        '.gb-nav-item:focus-visible{outline:2px solid #a855f7;outline-offset:-2px}',

        '.gb-nav-item.active{color:#a855f7;background:var(--gb-nav-active,rgba(168,85,247,.15))}',
        '.gb-nav-item.active::before{content:"";position:absolute;left:-10px;top:50%;',
        'transform:translateY(-50%);width:3px;height:22px;border-radius:0 3px 3px 0;',
        'background:#a855f7;box-shadow:0 0 12px rgba(168,85,247,.6)}',

        '.gb-nav-foot{margin-top:auto;padding:14px 10px 0;',
        'border-top:1px solid var(--gb-nav-border,rgba(255,255,255,.07))}',
        '.gb-nav-item.logout{color:#f87171}',
        '.gb-nav-item.logout:hover{background:rgba(248,113,113,.12);color:#fca5a5}',

        '.gb-nav-overlay{position:fixed;inset:0;z-index:6000;',
        'background:rgba(3,7,14,.62);backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);',
        'opacity:0;pointer-events:none;transition:opacity .26s ease}',
        '.gb-nav-overlay.open{opacity:1;pointer-events:auto}',

        /* ── Light mode ─────────────────────────────────────── */
        '[data-theme="light"]{--gb-nav-bg:#ffffff;--gb-nav-ink:#0f172a;',
        '--gb-nav-muted:#475569;--gb-nav-border:rgba(15,23,42,.10);',
        '--gb-nav-hover:rgba(168,85,247,.10);--gb-nav-active:rgba(168,85,247,.13);',
        '--gb-nav-btn:rgba(255,255,255,.94)}',
        '[data-theme="light"] .gb-nav-toggle{box-shadow:0 8px 22px -12px rgba(15,23,42,.35)}',
        '[data-theme="light"] .gb-nav-drawer{box-shadow:0 0 60px -20px rgba(15,23,42,.3)}',
        '[data-theme="light"] .gb-nav-overlay{background:rgba(15,23,42,.34)}',

        /* ── Mobile ─────────────────────────────────────────── */
        '@media (max-width:640px){',
        '.gb-nav-toggle{top:10px;left:10px;width:42px;height:42px}',
        '.gb-nav-drawer{width:min(80vw,272px);padding-bottom:calc(14px + env(safe-area-inset-bottom))}',
        '.gb-nav-item{padding:14px;font-size:14px;min-height:48px}',
        '.navbar,.report-header .header-inner,.admin-header .header-inner{padding-left:62px}',
        '}',

        '@media (prefers-reduced-motion:reduce){',
        '.gb-nav-drawer,.gb-nav-overlay,.gb-nav-bar{transition:none}}'
    ].join('');

    function injectStyles() {
        if (document.getElementById('gb-nav-styles')) return;
        var st = document.createElement('style');
        st.id = 'gb-nav-styles';
        st.textContent = CSS;
        document.head.appendChild(st);
    }

    /** Which route this page represents, for the active marker. */
    function currentRoute() {
        var path = (window.location.pathname || '').toLowerCase();
        if (path.indexOf('report') > -1) return 'report';
        if (path.indexOf('portfolio') > -1) return 'portfolio';
        if (path.indexOf('profile') > -1) return 'profile';
        if (path.indexOf('live') > -1) return 'live';
        if (path.indexOf('admin') > -1) return 'admin';
        return 'home';
    }

    function build() {
        var active = currentRoute();
        var here = (window.location.pathname || '').toLowerCase();

        var overlay = document.createElement('div');
        overlay.className = 'gb-nav-overlay';

        var toggle = document.createElement('button');
        toggle.className = 'gb-nav-toggle';
        toggle.id = 'gbNavToggle';
        toggle.setAttribute('aria-label', 'Menu');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = '<span class="gb-nav-bar"></span>' +
                           '<span class="gb-nav-bar"></span>' +
                           '<span class="gb-nav-bar"></span>';

        var drawer = document.createElement('nav');
        drawer.className = 'gb-nav-drawer';
        drawer.setAttribute('aria-label', 'Main navigation');

        var html = '<div class="gb-nav-brand">' +
            '<img src="/logo.svg" alt="">' +
            '<span>GoatBot Trade</span></div><div class="gb-nav-items">';

        ROUTES.forEach(function (r) {
            html += '<a class="gb-nav-item' + (r.route === active && !r.scrollTo ? ' active' : '') +
                '" href="' + r.href + '" data-route="' + r.route + '"' +
                (r.adminOnly ? ' data-admin-only="1" style="display:none"' : '') +
                (r.scrollTo ? ' data-scroll="' + r.scrollTo + '"' : '') +
                '><i class="fas ' + r.icon + '"></i>' + r.label + '</a>';
        });

        html += '</div><div class="gb-nav-foot">' +
            '<button class="gb-nav-item logout" id="gbNavLogout">' +
            '<i class="fas fa-sign-out-alt"></i>Logout</button></div>';

        drawer.innerHTML = html;

        document.body.appendChild(overlay);
        document.body.appendChild(drawer);
        document.body.appendChild(toggle);

        function setOpen(open) {
            drawer.classList.toggle('open', open);
            overlay.classList.toggle('open', open);
            toggle.classList.toggle('open', open);
            toggle.setAttribute('aria-expanded', String(open));
            document.body.style.overflow = open ? 'hidden' : '';
        }

        toggle.addEventListener('click', function () {
            setOpen(!drawer.classList.contains('open'));
        });
        overlay.addEventListener('click', function () { setOpen(false); });
        document.addEventListener('keydown', function (e) {
            if (e.key === 'Escape') setOpen(false);
        });

        // A link that points at the page you are already on should scroll,
        // not reload it.
        drawer.querySelectorAll('.gb-nav-item[href]').forEach(function (a) {
            a.addEventListener('click', function (e) {
                var target = (a.getAttribute('href') || '').toLowerCase();
                var samePage = here.indexOf(target.replace(/^\//, '')) > -1 ||
                               (target.indexOf('godbot_home') > -1 && here.indexOf('godbot_home') > -1);
                if (!samePage) return;

                e.preventDefault();
                var id = a.getAttribute('data-scroll');
                var el = id ? document.getElementById(id) : null;
                if (el) el.scrollIntoView({ behavior: 'smooth', block: 'start' });
                else window.scrollTo({ top: 0, behavior: 'smooth' });
                setOpen(false);
            });
        });

        var logout = document.getElementById('gbNavLogout');
        if (logout) {
            logout.addEventListener('click', function () {
                // A failed call must still get the user out of the session.
                fetch('/api/auth/logout', { method: 'POST' })
                    .catch(function () {})
                    .then(function () { window.location.href = '/godbot_login'; });
            });
        }
    }

    /** Reveal admin-only entries once the server says this account is one. */
    function revealAdminEntries() {
        var hidden = document.querySelectorAll('.gb-nav-item[data-admin-only]');
        if (!hidden.length) return;
        fetch('/api/v2/admin/whoami', { credentials: 'same-origin' })
            .then(function (r) { return r.ok ? r.json() : null; })
            .then(function (d) {
                if (!d || !d.is_admin) return;
                hidden.forEach(function (el) { el.style.display = ''; });
            })
            .catch(function () { /* not an admin, or not signed in — stay hidden */ });
    }

    function init() {
        if (document.getElementById('gbNavToggle')) return;   // already built
        injectStyles();
        build();
        revealAdminEntries();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

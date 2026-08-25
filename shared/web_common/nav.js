/**
 * nav.js — shared navigation drawer
 * =================================
 *
 * The Report / Portfolio / Profile / Live pages each shipped their own
 * standalone navbar with only a "Return to Command Deck" link, so the
 * three-line menu vanished the moment you left the home terminal.
 *
 * This module injects the same drawer — hamburger, off-canvas panel and
 * overlay — into any page that includes it, and carries its own styles so it
 * does not depend on the host page's stylesheet.
 *
 * The home terminal already has the drawer in its markup, so this bails out
 * there rather than rendering a second one.
 */
(function () {
    'use strict';

    var ROUTES = [
        { route: 'home',      href: '/godbot_home',  icon: 'fa-home',         label: 'Home' },
        { route: 'report',    href: '/v2/report',    icon: 'fa-file-invoice', label: 'Report' },
        { route: 'paper',     href: '/godbot_home',  icon: 'fa-vial',         label: 'Paper Trade' },
        { route: 'live',      href: '/v2/live',      icon: 'fa-rocket',       label: 'Live Trade' },
        { route: 'portfolio', href: '/v2/portfolio', icon: 'fa-wallet',       label: 'Portfolio' },
        { route: 'profile',   href: '/v2/profile',   icon: 'fa-user-cog',     label: 'Profile Settings' }
    ];

    var CSS = [
        '.gb-nav-toggle{position:fixed;top:14px;left:14px;z-index:6001;width:42px;height:42px;',
        'display:flex;align-items:center;justify-content:center;border-radius:11px;cursor:pointer;',
        'background:rgba(15,23,42,.72);border:1px solid rgba(255,255,255,.14);color:#fff;',
        'font-size:16px;backdrop-filter:blur(10px);-webkit-backdrop-filter:blur(10px);',
        'box-shadow:0 6px 18px -8px rgba(0,0,0,.6);transition:border-color .2s,color .2s}',
        '.gb-nav-toggle:hover{border-color:#a855f7;color:#a855f7}',
        '[data-theme="light"] .gb-nav-toggle{background:rgba(255,255,255,.94);color:#0f172a;',
        'border-color:rgba(15,23,42,.14);box-shadow:0 6px 18px -10px rgba(15,23,42,.3)}',

        '.gb-nav-drawer{position:fixed;top:0;left:0;bottom:0;width:248px;z-index:6002;',
        'display:flex;flex-direction:column;padding:18px 0;overflow-y:auto;',
        'background:#080d16;border-right:1px solid rgba(255,255,255,.08);',
        'transform:translateX(-100%);transition:transform .26s cubic-bezier(.16,1,.3,1)}',
        '.gb-nav-drawer.open{transform:none}',
        '[data-theme="light"] .gb-nav-drawer{background:#fff;border-right-color:rgba(15,23,42,.1)}',

        '.gb-nav-brand{display:flex;align-items:center;gap:10px;padding:0 20px 18px;',
        'margin-bottom:8px;border-bottom:1px solid rgba(255,255,255,.07)}',
        '[data-theme="light"] .gb-nav-brand{border-bottom-color:rgba(15,23,42,.09)}',
        '.gb-nav-brand img{width:32px;height:32px;object-fit:contain}',
        '.gb-nav-brand span{font-size:15px;font-weight:800;letter-spacing:.3px;color:#fff}',
        '[data-theme="light"] .gb-nav-brand span{color:#0f172a}',

        '.gb-nav-item{display:flex;align-items:center;gap:13px;padding:12px 20px;',
        'font-size:13px;font-weight:600;color:#94a3b8;text-decoration:none;cursor:pointer;',
        'background:none;border:none;width:100%;text-align:left;font-family:inherit;',
        'transition:background .18s,color .18s}',
        '.gb-nav-item i{width:18px;text-align:center;font-size:14px}',
        '.gb-nav-item:hover{background:rgba(168,85,247,.12);color:#fff}',
        '[data-theme="light"] .gb-nav-item{color:#475569}',
        '[data-theme="light"] .gb-nav-item:hover{background:rgba(168,85,247,.1);color:#0f172a}',
        '.gb-nav-item.active{color:#a855f7;background:rgba(168,85,247,.14);',
        'box-shadow:inset 3px 0 0 #a855f7}',
        '.gb-nav-item.logout{margin-top:auto;color:#f87171}',
        '.gb-nav-item.logout:hover{background:rgba(248,113,113,.12);color:#fca5a5}',

        '.gb-nav-overlay{position:fixed;inset:0;z-index:6000;background:rgba(3,7,14,.6);',
        'backdrop-filter:blur(3px);-webkit-backdrop-filter:blur(3px);',
        'opacity:0;pointer-events:none;transition:opacity .24s ease}',
        '.gb-nav-overlay.open{opacity:1;pointer-events:auto}',

        /* The host pages put their own header at the top-left; nudge it clear
           of the floating toggle rather than letting the two overlap. */
        '@media (max-width:1024px){.navbar,.report-header .header-inner{padding-left:66px}}',
        '@media (min-width:1025px){.navbar,.report-header .header-inner{padding-left:66px}}',

        '@media (prefers-reduced-motion:reduce){',
        '.gb-nav-drawer,.gb-nav-overlay{transition:none}}'
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
        return 'home';
    }

    function build() {
        var active = currentRoute();

        var overlay = document.createElement('div');
        overlay.className = 'gb-nav-overlay';

        var toggle = document.createElement('button');
        toggle.className = 'gb-nav-toggle';
        toggle.setAttribute('aria-label', 'Menu');
        toggle.setAttribute('aria-expanded', 'false');
        toggle.innerHTML = '<i class="fas fa-bars"></i>';

        var drawer = document.createElement('nav');
        drawer.className = 'gb-nav-drawer';
        drawer.setAttribute('aria-label', 'Main navigation');

        var html = '<div class="gb-nav-brand">' +
            '<img src="/logo.svg" alt="">' +
            '<span>GoatBot Trade</span></div>';

        ROUTES.forEach(function (r) {
            html += '<a class="gb-nav-item' + (r.route === active ? ' active' : '') + '" ' +
                'href="' + r.href + '"><i class="fas ' + r.icon + '"></i>' + r.label + '</a>';
        });

        html += '<button class="gb-nav-item logout" id="gbNavLogout">' +
            '<i class="fas fa-sign-out-alt"></i>Logout</button>';

        drawer.innerHTML = html;

        document.body.appendChild(overlay);
        document.body.appendChild(drawer);
        document.body.appendChild(toggle);

        function setOpen(open) {
            drawer.classList.toggle('open', open);
            overlay.classList.toggle('open', open);
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

        var logout = document.getElementById('gbNavLogout');
        if (logout) {
            logout.addEventListener('click', function () {
                fetch('/api/auth/logout', { method: 'POST' })
                    .catch(function () { /* log out locally regardless */ })
                    .then(function () { window.location.href = '/godbot_login'; });
            });
        }
    }

    function init() {
        // The home terminal ships its own drawer — do not duplicate it.
        if (document.querySelector('.slim-sidebar')) return;
        injectStyles();
        build();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();

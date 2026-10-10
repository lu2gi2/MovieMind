/* MovieMind — Sky Light. Shared behaviour:
   API helpers, one state store (Like / Favorite / Watchlist), reusable card, toasts,
   empty/error/skeleton states, header search, and the scroll-driven botanical background. */

/* ---------- helpers ---------- */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, cls = '') => `<svg class="i ${cls}" aria-hidden="true" focusable="false"><use href="#i-${name}"/></svg>`;

async function api(url, opts = {}) {
    const init = { credentials: 'same-origin', ...opts };
    if (opts.body && typeof opts.body !== 'string') {
        init.headers = { 'Content-Type': 'application/json', ...(opts.headers || {}) };
        init.body = JSON.stringify(opts.body);
    }
    let res;
    try { res = await fetch(url, init); } catch (e) { return { ok: false, status: 0, data: { error: 'Network error' } }; }
    const data = await res.json().catch(() => ({}));
    if (res.status === 401 && !location.pathname.match(/^\/(login|signup)?$/)) { location.href = '/login'; }
    return { ok: res.ok, status: res.status, data };
}
const get = (url) => api(url);

/* ---------- toasts ---------- */
function toast(msg, kind = '') {
    const box = $('toasts'); if (!box) return;
    const t = document.createElement('div');
    t.className = 'toast ' + (kind === 'error' ? 'is-error' : '');
    t.setAttribute('role', kind === 'error' ? 'alert' : 'status');
    t.textContent = msg;
    box.appendChild(t);
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 260); }, kind === 'error' ? 5000 : 3200);
}

/* ---------- one source of truth for interaction state ---------- */
const KINDS = {
    like: { set: 'likes', url: '/api/user/likes', on: 'Remove like', off: 'Like', label: 'Like' },
    fav: { set: 'favorites', url: '/api/user/favorites', on: 'Remove from Favorites', off: 'Add to Favorites', label: 'Favorite' },
    watch: { set: 'watchlist', url: '/api/user/watchlist', on: 'Remove from Watchlist', off: 'Add to Watchlist', label: 'Watchlist' },
};
const ICON = { like: 'heart', fav: 'star', watch: 'bookmark' };

const Store = {
    likes: new Set(), favorites: new Set(), watchlist: new Set(),
    loaded: false, pending: new Set(), listeners: [],
    on(fn) { this.listeners.push(fn); },
    emit(detail = {}) { this.listeners.forEach((fn) => fn(detail)); syncControls(); },
    load() {                                         // idempotent: every page can await it
        return this._p || (this._p = (async () => {
            const { ok, data } = await get('/api/user/state');
            if (ok) { this.likes = new Set(data.likes); this.favorites = new Set(data.favorites); this.watchlist = new Set(data.watchlist); }
            this.loaded = true; this.emit({ loaded: true });
        })());
    },
    has(kind, id) { return this[KINDS[kind].set].has(id); },
    async toggle(kind, id) {
        const k = KINDS[kind], key = kind + id;
        if (this.pending.has(key)) return;           // no duplicate requests / duplicate preference updates
        this.pending.add(key);
        const set = this[k.set], was = set.has(id);
        was ? set.delete(id) : set.add(id);           // optimistic
        this.emit({ kind, id, optimistic: true });
        const res = was ? await api(`${k.url}/${id}`, { method: 'DELETE' })
                        : await api(k.url, { method: 'POST', body: { movie_id: id } });
        const fine = res.ok || (!was && res.status === 409) || (was && res.status === 404);
        if (!fine) {                                  // restore previous state and tell the user
            was ? set.add(id) : set.delete(id);
            toast(`Couldn’t update ${k.label.toLowerCase()}. Please try again.`, 'error');
        } else if (kind === 'like') {
            toast(was ? 'Like removed. Your picks will adjust.' : (res.data.taste_message || 'Liked. Your picks will adjust.'));
        } else {
            toast(was ? `Removed from ${k.label === 'Favorite' ? 'Favorites' : 'Watchlist'}.` : `Added to ${k.label === 'Favorite' ? 'Favorites' : 'Watchlist'}.`);
        }
        this.pending.delete(key);
        this.emit({ kind, id, done: true, changedTaste: kind === 'like' && fine });
        return fine;
    },
};

function actButton(kind, id, { large = false } = {}) {
    const k = KINDS[kind], on = Store.has(kind, id);
    const lbl = on ? k.on : k.off;
    if (large) {
        return `<button class="btn-act act--${kind}" data-act="${kind}" data-id="${id}" aria-pressed="${on}"><span class="ico">${icon(ICON[kind], on ? 'i--fill' : '')}</span><span class="txt">${esc(on ? k.on : k.off)}</span></button>`;
    }
    return `<button class="act act--${kind}" data-act="${kind}" data-id="${id}" aria-pressed="${on}" aria-label="${esc(lbl)}" title="${esc(lbl)}">${icon(ICON[kind], on ? 'i--fill' : '')}</button>`;
}

function syncControls() {
    document.querySelectorAll('[data-act]:not([data-static])').forEach((b) => {
        const kind = b.dataset.act, id = +b.dataset.id, k = KINDS[kind], on = Store.has(kind, id);
        if (b.getAttribute('aria-pressed') === String(on)) return;
        b.setAttribute('aria-pressed', on);
        const lbl = on ? k.on : k.off;
        if (b.classList.contains('btn-act')) {
            b.querySelector('.ico').innerHTML = icon(ICON[kind], on ? 'i--fill' : '');
            b.querySelector('.txt').textContent = lbl;
        } else {
            b.setAttribute('aria-label', lbl); b.title = lbl; b.innerHTML = icon(ICON[kind], on ? 'i--fill' : '');
            b.classList.remove('pop'); void b.offsetWidth; b.classList.add('pop');
        }
    });
    const f = $('count-fav'), w = $('count-watch');
    if (f) f.textContent = Store.favorites.size || '';
    if (w) w.textContent = Store.watchlist.size || '';
}

document.addEventListener('click', (e) => {
    const b = e.target.closest('[data-act]');
    if (!b) return;
    e.preventDefault();
    Store.toggle(b.dataset.act, +b.dataset.id);
});

/* ---------- movie card (one component, used everywhere) ---------- */
function posterHTML(m) {
    const fb = `<div class="poster-fallback">${esc(m.title)}</div>`;
    if (!m.poster_url) return fb;
    return `<img src="${esc(m.poster_url)}" alt="" loading="lazy" decoding="async" data-fb="${esc(fb)}" onerror="this.outerHTML=this.dataset.fb">`;
}
const metaLine = (m) => [m.release_year, (m.genres || []).slice(0, 2).join(' · ')].filter(Boolean).join(' · ');

function rememberWhy(m) {
    if (!m.explanation) return;
    try { sessionStorage.setItem('mm:why:' + m.movie_id, JSON.stringify({ text: m.explanation, detail: m.explanation_detail || null })); } catch (e) { /* storage unavailable */ }
}

function cardHTML(m, { why = false, i = 0 } = {}) {
    if (why) rememberWhy(m);
    const href = `/movie/${m.movie_id}`;
    return `<article class="card" style="--i:${Math.min(i, 12)}">
        <a class="poster" href="${href}" tabindex="-1" aria-hidden="true">${posterHTML(m)}</a>
        <div class="card-body">
            <a class="card-title" href="${href}">${esc(m.title)}</a>
            <span class="card-meta">${esc(metaLine(m))}</span>
            ${why && m.explanation ? `<span class="card-why">${esc(m.explanation)}</span>` : ''}
        </div>
        <div class="actions">${actButton('like', m.movie_id)}${actButton('fav', m.movie_id)}${actButton('watch', m.movie_id)}<a class="more" href="${href}" aria-label="Details for ${esc(m.title)}">Details</a></div>
    </article>`;
}
const gridHTML = (movies, opts = {}) => movies.map((m, i) => cardHTML(m, { ...opts, i })).join('');

const skeleton = (n = 8) => Array.from({ length: n }, () => '<div class="skel-card" aria-hidden="true"><div class="skel"></div><i></i><i style="width:45%"></i></div>').join('');
const emptyHTML = (title, text, link) => `<div class="empty"><h2>${esc(title)}</h2><p>${esc(text)}</p>${link ? `<a class="btn" href="${link.href}">${esc(link.label)}</a>` : ''}</div>`;
const errorHTML = (msg = 'Something went wrong.', retry = 'location.reload()') => `<div class="empty" role="alert"><h2>That didn’t load</h2><p>${esc(msg)}</p><button class="btn btn--ghost" onclick="${retry}">Try again</button></div>`;

/* ---------- header: search, profile menu, active nav ---------- */
document.addEventListener('DOMContentLoaded', () => {
    const here = location.pathname;
    document.querySelectorAll('.nav a, .tabbar a').forEach((a) => {
        const h = a.getAttribute('href');
        if (h === here || (h === '/dashboard' && here === '/discover') || (h === '/browse' && here.startsWith('/movie/') && false)) a.setAttribute('aria-current', 'page');
    });
    const btn = $('user-btn'), menu = $('user-menu');
    if (btn && menu) {
        btn.addEventListener('click', (e) => { e.stopPropagation(); menu.hidden = !menu.hidden; btn.setAttribute('aria-expanded', String(!menu.hidden)); });
        document.addEventListener('click', () => { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); });
        document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { menu.hidden = true; btn.setAttribute('aria-expanded', 'false'); btn.focus(); } });
    }
    const out = $('logout-btn');
    if (out) out.addEventListener('click', async (e) => { e.preventDefault(); await api('/api/auth/logout', { method: 'POST' }); location.href = '/'; });
    if (document.body.dataset.auth === '1') Store.load();
});

/* ---------- scroll-driven botanical background ----------
   Moves briefly while the page scrolls, then eases back to rest and STOPS (no idle animation).
   Only CSS custom properties on <html> change; layers use transform. */
(function botanical() {
    const root = document.documentElement;
    if (!document.querySelector('.bg')) return;
    if (matchMedia('(prefers-reduced-motion: reduce)').matches) return;
    const touch = matchMedia('(pointer: coarse)').matches;
    const SCALE = touch ? 0.5 : 1;
    const MAIN = 16 * SCALE, LEAF = 9 * SCALE;
    let lastY = window.scrollY, target = 0, cur = 0, raf = 0, idle = 0;

    function apply(v) {
        const n = v / MAIN;                                   // -1 … 1
        root.style.setProperty('--by', (-v).toFixed(2) + 'px');
        root.style.setProperty('--bx', (v * 0.35).toFixed(2) + 'px');
        root.style.setProperty('--br', (n * 0.35).toFixed(3) + 'deg');
        root.style.setProperty('--fy', (-n * LEAF).toFixed(2) + 'px');
        root.style.setProperty('--fx', (n * LEAF * 0.7).toFixed(2) + 'px');
        root.style.setProperty('--fr', (n * 1.1).toFixed(3) + 'deg');
        root.style.setProperty('--cy', (n * 5 * SCALE).toFixed(2) + 'px');
        root.style.setProperty('--cx', (-n * 3 * SCALE).toFixed(2) + 'px');
    }
    function tick() {
        cur += (target - cur) * 0.14;                          // ease toward target
        if (target === 0 && Math.abs(cur) < 0.04) { cur = 0; apply(0); raf = 0; return; }   // settled: stop the loop
        apply(cur);
        raf = requestAnimationFrame(tick);
    }
    function onScroll() {
        const dy = window.scrollY - lastY; lastY = window.scrollY;
        target = Math.max(-MAIN, Math.min(MAIN, target + dy * 0.05));
        clearTimeout(idle);
        idle = setTimeout(() => { target = 0; if (!raf) raf = requestAnimationFrame(tick); }, 120);   // scrolling stopped → settle
        if (!raf) raf = requestAnimationFrame(tick);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('pagehide', () => { window.removeEventListener('scroll', onScroll); cancelAnimationFrame(raf); clearTimeout(idle); });
})();

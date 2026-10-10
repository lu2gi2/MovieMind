/* MovieMind — shared behaviour: auth nav, cards, detail dialog, favourites, toasts, vibe map */

const MM = { favs: new Set(), recs: {}, authed: false };

const esc = s => String(s ?? '').replace(/[&<>"']/g, c => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const icon = (name, cls = '') => `<svg class="i ${cls}" aria-hidden="true"><use href="#i-${name}"/></svg>`;
const json = (url, opts) => fetch(url, opts).then(async r => ({ ok: r.ok, status: r.status, data: await r.json().catch(() => ({})) }));
const post = (url, body) => json(url, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body || {}) });

/* ---------- auth + nav ---------- */
async function checkAuth() {
    try {
        const { data } = await json('/api/auth/check');
        MM.authed = !!data.authenticated;
        document.querySelectorAll('[data-auth="in"]').forEach(el => el.hidden = !MM.authed);
        document.querySelectorAll('[data-auth="out"]').forEach(el => el.hidden = MM.authed);
        if (MM.authed && data.user) {
            document.querySelectorAll('[data-username]').forEach(el => el.textContent = data.user.username);
        }
        return data;
    } catch (e) { return { authenticated: false }; }
}

document.addEventListener('DOMContentLoaded', () => {
    const here = location.pathname;
    document.querySelectorAll('.nav a').forEach(a => { if (a.getAttribute('href') === here) a.setAttribute('aria-current', 'page'); });

    const menuBtn = document.getElementById('menu-btn');
    if (menuBtn) menuBtn.addEventListener('click', () => {
        const open = document.querySelector('.nav').classList.toggle('open');
        menuBtn.setAttribute('aria-expanded', open);
    });

    const who = document.getElementById('who-btn'), menu = document.getElementById('who-menu');
    if (who) {
        who.addEventListener('click', e => { e.stopPropagation(); menu.hidden = !menu.hidden; who.setAttribute('aria-expanded', !menu.hidden); });
        document.addEventListener('click', () => { menu.hidden = true; who.setAttribute('aria-expanded', false); });
        document.addEventListener('keydown', e => { if (e.key === 'Escape') menu.hidden = true; });
    }

    const out = document.getElementById('logout-btn');
    if (out) out.addEventListener('click', async e => { e.preventDefault(); await post('/api/auth/logout'); location.href = '/'; });

    checkAuth();
});

/* ---------- toasts ---------- */
function toast(html, ms = 4200) {
    const box = document.getElementById('toasts');
    if (!box) return;
    const t = document.createElement('div');
    t.className = 'toast'; t.setAttribute('role', 'status'); t.innerHTML = html;
    box.appendChild(t);
    setTimeout(() => { t.classList.add('out'); setTimeout(() => t.remove(), 300); }, ms);
}

/* ---------- favourites ---------- */
async function loadFavIds() {
    const { ok, data } = await json('/api/user/favorites');
    if (ok && data.favorites) MM.favs = new Set(data.favorites.map(m => m.movie_id));
    return MM.favs;
}

async function toggleFav(id, btn) {
    const was = MM.favs.has(id);
    if (btn) btn.disabled = true;
    try {
        let res;
        if (was) res = await json(`/api/user/favorites/${id}`, { method: 'DELETE' });
        else res = await post('/api/user/favorites', { movie_id: id });
        if (!res.ok && res.status !== 409) { toast('Couldn’t save that one. Try again in a second.'); return was; }
        if (was) { MM.favs.delete(id); toast('Dropped from your list. Your taste has shifted back.'); }
        else {
            MM.favs.add(id);
            const msg = res.data.taste_message ? esc(res.data.taste_message) : 'Added to your list.';
            toast(`<b>Noted.</b> ${msg}`);
        }
        document.querySelectorAll(`.heart[data-id="${id}"]`).forEach(h => {
            h.setAttribute('aria-pressed', MM.favs.has(id));
            h.setAttribute('aria-label', MM.favs.has(id) ? 'Remove from my list' : 'Add to my list');
            h.innerHTML = icon('heart', MM.favs.has(id) ? 'i--fill' : '');
            h.classList.remove('pop'); void h.offsetWidth; h.classList.add('pop');
        });
        document.dispatchEvent(new CustomEvent('taste:changed'));
        return MM.favs.has(id);
    } finally { if (btn) btn.disabled = false; }
}

/* ---------- cards ---------- */
function posterHTML(m) {
    const fb = `<div class="poster-fallback">${esc(m.title)}</div>`;
    if (!m.poster_url) return fb;
    return `<img src="${esc(m.poster_url)}" alt="" loading="lazy" onerror="this.outerHTML=this.dataset.fb" data-fb="${esc(fb)}">`;
}

function meterHTML(d) {
    if (!d) return '';
    const t = Math.max(0, Math.min(1, d.taste_match)) * 100;
    const mo = Math.max(0, Math.min(0.3, d.mood_effect * 3)) * 100;
    const sign = d.mood_effect >= 0 ? '+' : '−';
    return `<div class="meter" role="img" aria-label="Taste match ${d.taste_match.toFixed(2)}, mood effect ${sign}${Math.abs(d.mood_effect).toFixed(2)}">
              <span class="m-taste" style="width:${t}%"></span><span class="m-mood" style="width:${mo}%"></span></div>
            <div class="meter-label mono"><span class="t">taste <b>${d.taste_match.toFixed(2)}</b></span><span class="m">mood <b>${sign}${Math.abs(d.mood_effect).toFixed(2)}</b></span></div>`;
}

function cardHTML(m, { rec = false, i = 0 } = {}) {
    const fav = MM.favs.has(m.movie_id);
    if (rec) MM.recs[m.movie_id] = m;
    const meta = [m.release_year, (m.genres || []).slice(0, 2).join(' / '), m.vote_average ? m.vote_average.toFixed(1) : null].filter(Boolean).join(' · ');
    return `<article class="card" style="--i:${i}">
        <div class="media">
            <button class="poster" data-open="${m.movie_id}" aria-label="Details: ${esc(m.title)}">${posterHTML(m)}</button>
            <button class="heart" data-id="${m.movie_id}" aria-pressed="${fav}" aria-label="${fav ? 'Remove from my list' : 'Add to my list'}">${icon('heart', fav ? 'i--fill' : '')}</button>
        </div>
        <div class="body" style="min-width:0">
            <h3 class="title">${esc(m.title)}</h3>
            <p class="sub mono">${esc(meta)}</p>
            ${rec ? `<p class="overview">${esc(m.overview || '')}</p><p class="why">${esc(m.explanation || '')}</p>${meterHTML(m.explanation_detail)}` : ''}
        </div>
    </article>`;
}

function bindCards(root) {
    root.addEventListener('click', e => {
        const open = e.target.closest('[data-open]'), heart = e.target.closest('.heart');
        if (heart) { e.stopPropagation(); toggleFav(+heart.dataset.id, heart); }
        else if (open) openDetail(+open.dataset.open);
    });
}

/* ---------- detail dialog ---------- */
async function openDetail(id) {
    const dlg = document.getElementById('movie-dialog');
    if (!dlg) return;
    const { ok, data } = await json(`/api/movies/${id}`);
    if (!ok) { toast('That film wouldn’t open. Try again.'); return; }
    const m = data.movie, r = MM.recs[id];
    const fav = MM.favs.has(id);
    dlg.querySelector('.dlg').innerHTML = `
        <div class="poster" style="cursor:default">${posterHTML(m)}</div>
        <div>
            <p class="eyebrow">${esc([m.release_year, m.runtime ? m.runtime + ' min' : null].filter(Boolean).join(' · '))}</p>
            <h2 id="dlg-title">${esc(m.title)}</h2>
            <p>${esc(m.overview || 'No synopsis on file.')}</p>
            <dl>
                <dt>Genres</dt><dd>${esc((m.genres || []).join(', ') || '—')}</dd>
                <dt>Director</dt><dd>${esc(m.director || '—')}</dd>
                <dt>Cast</dt><dd>${esc((m.cast || []).slice(0, 5).join(', ') || '—')}</dd>
                <dt>Rating</dt><dd class="mono">${m.vote_average ? m.vote_average.toFixed(1) + ' / 10' : '—'}</dd>
            </dl>
            ${r ? `<div class="why-box"><p class="eyebrow">Why this one</p><p class="why" style="font-size:1rem">${esc(r.explanation)}</p>${meterHTML(r.explanation_detail)}</div>` : ''}
            <div class="dlg-actions">
                <button class="btn ${fav ? 'btn--ghost' : ''}" id="dlg-fav" data-id="${id}">${icon('heart', fav ? 'i--fill' : '')} ${fav ? 'On your list' : 'Add to my list'}</button>
                <button class="btn btn--ghost" id="dlg-close">Close</button>
            </div>
        </div>`;
    dlg.querySelector('#dlg-close').onclick = () => dlg.close();
    dlg.querySelector('#dlg-fav').onclick = async function () {
        const now = await toggleFav(id, this);
        this.className = 'btn' + (now ? ' btn--ghost' : '');
        this.innerHTML = `${icon('heart', now ? 'i--fill' : '')} ${now ? 'On your list' : 'Add to my list'}`;
    };
    dlg.showModal();
}

document.addEventListener('DOMContentLoaded', () => {
    const dlg = document.getElementById('movie-dialog');
    if (!dlg) return;
    dlg.addEventListener('click', e => { if (e.target === dlg) dlg.close(); });
    dlg.querySelector('.dlg-x').addEventListener('click', () => dlg.close());
});

/* ---------- vibe map (the idea in one picture) ---------- */
function drawVibe(svg, { lam = 0.3, mood = null, label = true } = {}) {
    const W = 280, H = 150, NS = 'http://www.w3.org/2000/svg';
    svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
    if (!svg.dataset.built) {
        svg.dataset.built = 1;
        let s = 7; const rnd = () => (s = (s * 16807) % 2147483647) / 2147483647;
        const blobs = [[60, 96, 'heist'], [118, 44, 'mystery'], [196, 110, 'romance'], [232, 40, 'horror'], [150, 124, 'drama']];
        let dots = '';
        blobs.forEach(([cx, cy]) => { for (let i = 0; i < 26; i++) { const a = rnd() * 6.28, r = Math.abs((rnd() + rnd() - 1)) * 26; dots += `<circle cx="${(cx + r * Math.cos(a)).toFixed(1)}" cy="${(cy + r * Math.sin(a) * .8).toFixed(1)}" r="1.5"/>`; } });
        svg.innerHTML = `<g class="dotfield">${dots}</g>
            <line class="path" x1="60" y1="96" x2="232" y2="40" stroke="#B8381A" stroke-width="1.2" stroke-dasharray="3 3"/>
            <circle class="target" cx="232" cy="40" r="11" fill="none" stroke="#B8381A" stroke-width="1.6"/>
            <circle cx="60" cy="96" r="5" fill="#0E6660"/>
            <circle class="pos" cx="60" cy="96" r="4" fill="#F3EEE3" stroke="currentColor" stroke-width="2"/>
            <text class="t-taste" x="60" y="116" text-anchor="middle" fill="#0E6660">your taste</text>
            <text class="t-mood" x="232" y="68" text-anchor="middle" fill="#B8381A">mood</text>`;
    }
    const x = 60 + (232 - 60) * lam, y = 96 + (40 - 96) * lam;
    const pos = svg.querySelector('.pos');
    pos.setAttribute('cx', x); pos.setAttribute('cy', y);
    const on = !!mood;
    svg.querySelector('.path').style.opacity = on ? 1 : .15;
    svg.querySelector('.target').style.opacity = on ? 1 : .15;
    svg.querySelector('.t-mood').textContent = on ? mood : 'no mood';
    svg.querySelector('.t-mood').style.opacity = on ? 1 : .4;
    if (!on) { pos.setAttribute('cx', 60); pos.setAttribute('cy', 96); }
}

function lambdaWords(v, mood) {
    const m = mood || 'mood';
    if (!mood) return 'Only what you already love';
    if (v === 0) return 'Pure taste. Mood ignored';
    if (v <= 0.15) return `A whisper of ${m}`;
    if (v <= 0.4) return `A gentle nudge toward ${m}`;
    if (v <= 0.7) return `Pushing out toward ${m}`;
    if (v < 1) return `Mostly ${m}, a little you`;
    return `All ${m}. Your history is ignored`;
}

#!/usr/bin/env python3
"""
Offline evaluation of MovieMind v2 on MovieLens (ml-latest-small), DB-free and deterministic.

Protocol
  * catalogue  = the 4,988 enriched movies (rows of movie_embeddings.npy)
  * per user   = ratings on catalogue movies sorted by time; earliest 80 % = train, latest 20 % = test
  * relevant   = test rating >= 4.0 ;  "liked" in train = rating >= 4.0
  * all train-rated movies are excluded from every system's candidate list
  * users need >= MIN_TRAIN_LIKED liked train movies and >= 1 relevant test movie

Systems (top-K = 10)
  POP  most-liked movies in the training data (reference only)
  B1   TF-IDF + genre engine of v1 (hybrid similarity, genre-EMA taste, genre mood table)
  B2   embedding centroid, no mood                      (lambda = 0)
  B3   embedding centroid, mood as a genre-table boost  (v1-style mood, but on embeddings)
  OURS embedding centroid + lambda * mood_direction     (lambda = 0.3)

Mood proxy (MovieLens has no mood labels, no test leakage): a user's "current mood" is the mood
direction closest (cosine) to (mean of their last RECENT liked train movies - mean of all liked
train movies), i.e. the direction their recent viewing has drifted. Controls: random mood.

Metrics: HR@K, NDCG@K (binary), Diversity@K (1 - mean pairwise cosine of the list in embedding
space), Serendipity@K (relevant hits whose genres miss the user's 3 most-watched train genres, / K).
"""
import json
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from app.ml.embeddings import EmbeddingStore, MOODS, unit
from app.ml.similarity import SimilarityEngine
from app.ml.recommender import RecommendationEngine

K, SPLIT, REL, MIN_TRAIN_LIKED, RECENT, LAMBDA, SEED = 10, 0.8, 4.0, 8, 5, 0.3, 0
SWEEP = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
N_BOOT = 2000
OUT = os.path.join(ROOT, 'docs', 'report', 'results')
FIG = os.path.join(ROOT, 'docs', 'report', 'figures')
MOOD_MAP = RecommendationEngine.MOOD_GENRE_MAP


# ------------------------------------------------------------------ data
def load():
    store = EmbeddingStore.load()
    meta = pd.read_csv(os.path.join(ROOT, 'data/raw/movies_enriched.csv')).set_index('movieId')
    genres = [json.loads(meta.loc[int(i), 'genres']) if int(i) in meta.index and isinstance(meta.loc[int(i), 'genres'], str) else []
              for i in store.ids]
    vote = np.array([float(meta.loc[int(i), 'vote_average']) if int(i) in meta.index and not pd.isna(meta.loc[int(i), 'vote_average']) else 5.0
                     for i in store.ids])
    ratings = pd.read_csv(os.path.join(ROOT, 'data/raw/ratings.csv'))
    ratings = ratings[ratings.movieId.isin(store.id_to_row)].copy()
    ratings['row'] = ratings.movieId.map(store.id_to_row)
    return store, genres, vote, ratings


def make_users(store, ratings):
    users = []
    for uid, g in ratings.sort_values(['timestamp', 'movieId']).groupby('userId'):
        n_train = int(len(g) * SPLIT)
        tr, te = g.iloc[:n_train], g.iloc[n_train:]
        liked = tr[tr.rating >= REL].row.tolist()           # chronological
        rel = set(te[te.rating >= REL].row.tolist())
        if len(liked) < MIN_TRAIN_LIKED or not rel:
            continue
        users.append({'uid': int(uid), 'liked': liked, 'seen': np.array(sorted(set(tr.row.tolist()))), 'rel': rel})
    return users


# ------------------------------------------------------------------ helpers
def diverse(order, sims, clusters, k=K):
    cap, counts, out = max(2, k // 3), {}, []
    for r in order:
        if not np.isfinite(sims[r]):
            break
        c = int(clusters[r])
        if counts.get(c, 0) < cap:
            out.append(int(r)); counts[c] = counts.get(c, 0) + 1
        if len(out) == k:
            break
    return out


def proxy_mood(store, u):
    E = store.E
    dev = E[u['liked'][-RECENT:]].mean(axis=0) - E[u['liked']].mean(axis=0)
    if np.linalg.norm(dev) < 1e-8:
        return None
    sims = [float(unit(dev) @ store.mood_vectors[m]) for m in MOODS]
    return MOODS[int(np.argmax(sims))]


def oracle_mood(store, u):
    """DIAGNOSTIC ONLY (uses test items): mood closest to the drift from taste toward the relevant test movies."""
    dev = store.E[list(u['rel'])].mean(axis=0) - store.E[u['liked']].mean(axis=0)
    return MOODS[int(np.argmax([float(unit(dev) @ store.mood_vectors[m]) for m in MOODS]))]


def taste_of(store, u):
    return unit(store.E[u['liked']].mean(axis=0))


def emb_rank(store, u, mood, lam):
    sims = store.scores(store.search_position(taste_of(store, u), mood, lam))
    sims[u['seen']] = -np.inf
    return diverse(np.argsort(-sims), sims, store.clusters)


def b3_rank(store, genres, u, mood):
    sims = store.scores(taste_of(store, u)).astype('float64')
    if mood:
        boosts = MOOD_MAP[mood]
        mult = np.array([max([boosts.get(g, 1.0) for g in gs] or [1.0]) for gs in genres])
        sims = sims * mult
    sims[u['seen']] = -np.inf
    return diverse(np.argsort(-sims), sims, store.clusters)


class V1:
    """The v1 engine (TF-IDF + genre hybrid similarity, genre EMA taste, genre mood boost), DB-free."""

    def __init__(self, store, genres, vote):
        eng = SimilarityEngine().load_matrices(os.path.join(ROOT, 'data/models'))
        assert [eng.idx_to_movie_id[i] for i in range(len(store.ids))] == [int(x) for x in store.ids], 'row order mismatch'
        from sklearn.metrics.pairwise import cosine_similarity
        S = 0.7 * cosine_similarity(eng.tfidf_matrix) + 0.3 * cosine_similarity(eng.genre_matrix)
        np.fill_diagonal(S, -np.inf)
        self.S32 = S.astype('float32')
        self.top50 = np.argsort(-self.S32, axis=1)[:, :50]          # per-movie neighbours, as get_similar_movies(n=50)
        self.genres, self.vote = genres, vote
        self.prim = [gs[0] if gs else 'Unknown' for gs in genres]

    def rank(self, u, mood, store):
        # genre EMA taste, alpha = 0.2, neutral 0.5 (TasteUpdater.update_on_like), in chronological order
        taste = {}
        for r in u['liked']:
            for g in self.genres[r]:
                taste[g] = 0.8 * taste.get(g, 0.5) + 0.2
        agg = {}
        liked = set(u['liked'])
        for r in u['liked']:
            for c in self.top50[r]:
                if int(c) not in liked:
                    agg.setdefault(int(c), []).append(float(self.S32[r, c]))
        seen = set(int(x) for x in u['seen'])
        cands = sorted(((c, np.mean(v)) for c, v in agg.items() if c not in seen), key=lambda x: -x[1])[:3 * K]
        boosts = MOOD_MAP.get(mood, {}) if mood else {}
        scored = []
        for c, sim in cands:
            gs = self.genres[c]
            tm = np.mean([taste.get(g, 0.5) for g in gs]) if gs else 0.5
            score = 0.4 * sim + 0.3 * tm + 0.15 * min(self.vote[c] / 10.0, 1.0)
            score *= max([boosts.get(g, 1.0) for g in gs] or [1.0])
            scored.append((c, score))
        scored.sort(key=lambda x: -x[1])
        cap, counts, out = max(2, K // 3), {}, []
        for c, _ in scored:                                           # v1 diversity: per primary genre
            if counts.get(self.prim[c], 0) < cap:
                out.append(c); counts[self.prim[c]] = counts.get(self.prim[c], 0) + 1
            if len(out) == K:
                break
        return out


# ------------------------------------------------------------------ metrics
def metrics(rows, u, store, genres, top_genres):
    rel = u['rel']
    hits = [1 if r in rel else 0 for r in rows]
    dcg = sum(h / np.log2(i + 2) for i, h in enumerate(hits))
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(rel), K)))
    if len(rows) > 1:
        M = store.E[rows] @ store.E[rows].T
        div = 1.0 - (M.sum() - len(rows)) / (len(rows) * (len(rows) - 1))
    else:
        div = 0.0
    seren = sum(1 for r, h in zip(rows, hits) if h and not (set(genres[r]) & top_genres)) / K
    return {'hr': float(any(hits)), 'ndcg': dcg / idcg if idcg else 0.0, 'div': float(div), 'seren': float(seren)}


def user_top_genres(u, genres):
    cnt = {}
    for r in u['liked']:
        for g in genres[r]:
            cnt[g] = cnt.get(g, 0) + 1
    return set(sorted(cnt, key=lambda g: -cnt[g])[:3])


def boot_ci(x, rng, n=N_BOOT):
    x = np.asarray(x, dtype=float)
    idx = rng.integers(0, len(x), size=(n, len(x)))
    m = x[idx].mean(axis=1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


# ------------------------------------------------------------------ main
def main():
    rng = np.random.default_rng(SEED)
    t_load = time.time()
    store, genres, vote, ratings = load()
    users = make_users(store, ratings)
    print(f'catalogue {len(store.ids)} movies | ratings on catalogue {len(ratings)} | users evaluated {len(users)} / {ratings.userId.nunique()}')
    v1 = V1(store, genres, vote)
    print(f'setup {time.time() - t_load:.1f}s')

    topg = [user_top_genres(u, genres) for u in users]
    pmood = [proxy_mood(store, u) for u in users]
    rmood = [MOODS[i] for i in rng.integers(0, len(MOODS), size=len(users))]
    pop_count = ratings[ratings.rating >= REL].groupby('row').size()
    train_pop = np.zeros(len(store.ids))   # popularity from the same training split to avoid leakage
    for u in users:
        for r in u['liked']:
            train_pop[r] += 1

    def run(fn):
        res = [metrics(fn(i, u), u, store, genres, topg[i]) for i, u in enumerate(users)]
        return {k: np.array([r[k] for r in res]) for k in ('hr', 'ndcg', 'div', 'seren')}

    def pop_fn(i, u):
        s = train_pop.copy(); s[u['seen']] = -np.inf
        return diverse(np.argsort(-s), s, store.clusters)

    systems = {
        'POP (reference)': run(pop_fn),
        'B1: TF-IDF + genre (v1)': run(lambda i, u: v1.rank(u, pmood[i], store)),
        'B2: semantic centroid, no mood (lambda=0)': run(lambda i, u: emb_rank(store, u, None, 0.0)),
        'B3: semantic centroid + genre-table mood': run(lambda i, u: b3_rank(store, genres, u, pmood[i])),
        f'Proposed: mood-shifted (lambda={LAMBDA}), proxy mood': run(lambda i, u: emb_rank(store, u, pmood[i], LAMBDA)),
        f'Control: mood-shifted (lambda={LAMBDA}), random mood': run(lambda i, u: emb_rank(store, u, rmood[i], LAMBDA)),
    }

    table, brng = [], np.random.default_rng(SEED + 1)
    for name, m in systems.items():
        row = {'system': name, 'n_users': len(users)}
        for k in ('hr', 'ndcg', 'div', 'seren'):
            mean, lo, hi = boot_ci(m[k], brng)
            row[k], row[k + '_lo'], row[k + '_hi'] = mean, lo, hi
        table.append(row)

    # paired comparison: Proposed - B2 (same users)
    prop = systems[f'Proposed: mood-shifted (lambda={LAMBDA}), proxy mood']
    b2 = systems['B2: semantic centroid, no mood (lambda=0)']
    paired = {}
    for k in ('hr', 'ndcg', 'div', 'seren'):
        d = prop[k] - b2[k]
        mean, lo, hi = boot_ci(d, brng)
        paired[k] = {'diff': mean, 'lo': lo, 'hi': hi, 'users_better': int((d > 0).sum()), 'users_worse': int((d < 0).sum()), 'users_equal': int((d == 0).sum())}

    # lambda sweep (proxy and random mood)
    sweep = []
    for lam in SWEEP:
        for label, moods in (('proxy', pmood), ('random', rmood)):
            m = run(lambda i, u: emb_rank(store, u, moods[i], lam))
            r = {'lambda': lam, 'mood_source': label}
            for k in ('hr', 'ndcg', 'div', 'seren'):
                mean, lo, hi = boot_ci(m[k], brng, 500)
                r[k], r[k + '_lo'], r[k + '_hi'] = mean, lo, hi
            sweep.append(r)

    # diagnostics (clearly labelled; NOT proposed methods)
    omood = [oracle_mood(store, u) for u in users]
    for lam in SWEEP:
        m = run(lambda i, u: emb_rank(store, u, omood[i], lam))
        r = {'lambda': lam, 'mood_source': 'oracle'}
        for k in ('hr', 'ndcg', 'div', 'seren'):
            mean, lo, hi = boot_ci(m[k], brng, 500)
            r[k], r[k + '_lo'], r[k + '_hi'] = mean, lo, hi
        sweep.append(r)
    diag = {'quality_prior': {}, 'no_diversity_filter': {}}
    for w in (0.0, 0.05, 0.15, 0.3):
        def qfn(i, u, w=w):
            sc = store.scores(taste_of(store, u)) + w * vote / 10.0
            sc[u['seen']] = -np.inf
            return diverse(np.argsort(-sc), sc, store.clusters)
        m = run(qfn)
        diag['quality_prior'][str(w)] = {'hr': float(m['hr'].mean()), 'ndcg': float(m['ndcg'].mean())}
    def nodiv(i, u):
        sc = store.scores(taste_of(store, u)); sc[u['seen']] = -np.inf
        return [int(x) for x in np.argsort(-sc)[:K]]
    m = run(nodiv)
    diag['no_diversity_filter'] = {'hr': float(m['hr'].mean()), 'ndcg': float(m['ndcg'].mean()), 'div': float(m['div'].mean())}

    # sanity: lambda = 0 must equal B2 exactly
    z = run(lambda i, u: emb_rank(store, u, pmood[i], 0.0))
    sanity = bool(all(np.array_equal(z[k], b2[k]) for k in z))

    # query-time compute latency (score + sort + diversity), per request
    lat = []
    for i, u in enumerate(users[:200]):
        t = time.perf_counter(); emb_rank(store, u, pmood[i], LAMBDA); lat.append((time.perf_counter() - t) * 1000)
    latency = {'median_ms': float(np.median(lat)), 'p95_ms': float(np.percentile(lat, 95)), 'n': len(lat)}

    mood_counts = {m: int(sum(1 for x in pmood if x == m)) for m in MOODS}
    result = {'config': {'K': K, 'split': SPLIT, 'relevant_rating': REL, 'min_train_liked': MIN_TRAIN_LIKED, 'recent_window': RECENT,
                         'lambda': LAMBDA, 'seed': SEED, 'bootstrap': N_BOOT},
              'data': {'catalogue': len(store.ids), 'ratings_on_catalogue': int(len(ratings)), 'users_total': int(ratings.userId.nunique()), 'users_evaluated': len(users)},
              'systems': table, 'paired_vs_B2': paired, 'lambda0_equals_B2': sanity, 'latency_compute': latency,
              'proxy_mood_distribution': mood_counts, 'diagnostics': diag, 'sweep': sweep}
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, 'results.json'), 'w'), indent=1)
    pd.DataFrame(table).to_csv(os.path.join(OUT, 'systems.csv'), index=False)
    pd.DataFrame(sweep).to_csv(os.path.join(OUT, 'sweep.csv'), index=False)

    print('\n%-58s %7s %8s %9s %8s' % ('system', 'HR@10', 'NDCG@10', 'Div@10', 'Seren@10'))
    for r in table:
        print('%-58s %7.3f %8.3f %9.3f %8.3f' % (r['system'], r['hr'], r['ndcg'], r['div'], r['seren']))
    print('\npaired (Proposed - B2):', json.dumps({k: {a: round(b, 4) if isinstance(b, float) else b for a, b in v.items()} for k, v in paired.items()}))
    print('lambda=0 identical to B2:', sanity, '| latency', {k: round(v, 2) for k, v in latency.items()})
    print('proxy mood distribution:', mood_counts)
    print('diagnostics:', json.dumps(diag))
    make_charts(table, sweep)


# ------------------------------------------------------------------ charts
def make_charts(table, sweep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NAVY, TEAL, ORANGE, GREY = '#1F4E79', '#00897B', '#E67E22', '#8A8F98'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    df = pd.DataFrame(sweep)
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (k, title) in zip(ax, [('hr', 'Hit Rate@10'), ('ndcg', 'NDCG@10')]):
        for src, col, ls, lab in (('proxy', NAVY, '-', 'proxy mood (recent-likes drift)'), ('random', GREY, '-', 'random mood'),
                                  ('oracle', ORANGE, '--', 'oracle mood (upper bound; uses test items)')):
            d = df[df.mood_source == src]
            a.plot(d['lambda'], d[k], marker='o', color=col, lw=2, ls=ls, label=lab)
            a.fill_between(d['lambda'], d[k + '_lo'], d[k + '_hi'], color=col, alpha=0.12)
        a.axvline(LAMBDA, color='#999999', ls=':', lw=1.2); a.text(LAMBDA + 0.01, a.get_ylim()[0], ' default λ=0.3', color='#666666', fontsize=9, va='bottom')
        a.set_xlabel('λ (mood strength)'); a.set_title(title, loc='left', fontweight='bold'); a.grid(alpha=0.25)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_7_1_lambda.png'), dpi=160); plt.close(fig)

    names = {'POP (reference)': 'Popularity', 'B1: TF-IDF + genre (v1)': 'B1  TF-IDF\n+ genre (v1)', 'B2: semantic centroid, no mood (lambda=0)': 'B2  semantic\nλ = 0',
             'B3: semantic centroid + genre-table mood': 'B3  semantic +\ngenre mood'}
    rows = [r for r in table if 'random' not in r['system']]
    lab = [names.get(r['system'], 'Proposed\nλ = 0.3') for r in rows]
    cols = [GREY, '#B0B7C3', '#5B8DB8', TEAL, NAVY][:len(rows)]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for a, (k, title) in zip(ax, [('hr', 'Hit Rate@10'), ('ndcg', 'NDCG@10')]):
        v = [r[k] for r in rows]
        err = [[r[k] - r[k + '_lo'] for r in rows], [r[k + '_hi'] - r[k] for r in rows]]
        a.bar(range(len(rows)), v, yerr=err, color=cols, capsize=3, width=0.65)
        a.set_xticks(range(len(rows))); a.set_xticklabels(lab, fontsize=8.5); a.set_title(title + '  (95% bootstrap CI)', loc='left', fontweight='bold'); a.grid(axis='y', alpha=0.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_7_2_methods.png'), dpi=160); plt.close(fig)
    print('charts written')


if __name__ == '__main__':
    main()

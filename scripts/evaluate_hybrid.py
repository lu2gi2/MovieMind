#!/usr/bin/env python3
"""
Evaluation of the hybrid recommender (v2.1) with a leakage-free, user-level protocol.

  * Users are split at random into TRAIN (60%), VALIDATION (20%) and TEST (20%) groups. Repeated for 5 seeds.
  * Item statistics (mean rating, popularity, item-item "liked together" similarity) are built ONLY from the
    TRAIN group's ratings, so nothing the evaluated users did can leak into them. This mirrors the product:
    a new user versus knowledge of the crowd.
  * Every evaluated user (validation / test) is scored the same way as in evaluate.py: the earliest 80% of
    their ratings are their history (liked = rating >= 4), the latest 20% are the targets (relevant = >= 4).
  * Blend weights are tuned on VALIDATION users (maximising NDCG@10). TEST users are only used to report.
  * Mood: proxy (drift of recent likes), random (control) and oracle (diagnostic upper bound, uses test items).

Run: python scripts/evaluate_hybrid.py     (writes docs/report/results/hybrid.json, charts, data/models/*)
"""
import itertools
import json
import os
import sys
import time
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

import evaluate as ev                                   # shared loading, metrics, v1 baseline, mood proxies
from app.ml import hybrid as H
from app.ml.embeddings import MODELS_DIR, MOODS

K = 10
SEEDS = [0, 1, 2, 3, 4]
GRID = [0.0, 0.5, 1.0, 2.0]
GAMMAS = [0.0, 0.3, 0.5, 0.7]       # collaborative popularity discount candidates
MIN_COVERAGE = 0.15                 # product requirement fixed in advance: >= 15% distinct titles per recommendation slot (validation)
FRONTIER = [0.0, 0.10, 0.15, 0.20, 0.25]   # constraint levels reported as an accuracy-vs-variety frontier
LAMBDA = 0.3
SWEEP = [round(x, 1) for x in np.arange(0.0, 1.01, 0.1)]
OUT = ev.OUT
FIG = ev.FIG


def split_groups(user_ids, seed):
    rng = np.random.default_rng(1000 + seed)
    ids = np.array(sorted(user_ids)); rng.shuffle(ids)
    a, b = int(len(ids) * 0.6), int(len(ids) * 0.8)
    return set(ids[:a].tolist()), set(ids[a:b].tolist()), set(ids[b:].tolist())


def ndcg_hits(top, rel):
    hits = np.array([1.0 if r in rel else 0.0 for r in top])
    dcg = (hits / np.log2(np.arange(2, len(hits) + 2))).sum()
    idcg = (1 / np.log2(np.arange(2, min(len(rel), K) + 2))).sum()
    return float(hits.max() > 0), (dcg / idcg if idcg else 0.0)


def tune(users, comps, mask=None, min_cov=0.0):
    """
    Grid search blend weights on validation users (maximise NDCG@10). mask: components allowed to be non-zero.
    min_cov: only weights whose validation lists show at least this many distinct titles per recommendation slot qualify.
    """
    mask = mask or H.COMPONENTS
    best = (-1, -1, None, 0.0)
    stack = np.stack([[c[k] for k in H.COMPONENTS] for c in comps])          # (U, 4, N)
    seen = [u['seen'] for u in users]
    for w in itertools.product(GRID, repeat=len(H.COMPONENTS)):
        if not any(w) or any(wi and k not in mask for wi, k in zip(w, H.COMPONENTS)):
            continue
        sc = np.einsum('k,ukn->un', np.array(w), stack)
        hr, nd, flat = [], [], []
        for i, u in enumerate(users):
            s = sc[i].copy(); s[seen[i]] = -np.inf
            top = np.argpartition(-s, K)[:K]; top = top[np.argsort(-s[top])]
            h, n = ndcg_hits(top.tolist(), u['rel']); hr.append(h); nd.append(n); flat += top.tolist()
        cov = len(set(flat)) / len(flat)
        if cov < min_cov:
            continue
        cand = (float(np.mean(nd)), float(np.mean(hr)), dict(zip(H.COMPONENTS, w)), cov)
        if cand[:2] > best[:2]:
            best = cand
    if best[2] is None:                     # nothing met the constraint: fall back to the most varied option
        return tune(users, comps, mask, min_cov=0.0)
    return {'weights': best[2], 'val_ndcg': best[0], 'val_hr': best[1], 'val_distinct_per_slot': best[3]}


def main():
    t0 = time.time()
    store, genres, vote, ratings = ev.load()
    users_all = ev.make_users(store, ratings)
    all_uids = ratings.userId.unique().tolist()
    N = len(store.ids)
    print(f'catalogue {N} | ratings {len(ratings)} | eligible eval users {len(users_all)} of {len(all_uids)}')
    v1 = ev.V1(store, genres, vote)
    topg = {u['uid']: ev.user_top_genres(u, genres) for u in users_all}
    by_uid = {u['uid']: u for u in users_all}
    rng_r = np.random.default_rng(7)
    rmood = {u['uid']: MOODS[int(rng_r.integers(0, len(MOODS)))] for u in users_all}
    pmood = {u['uid']: ev.proxy_mood(store, u) for u in users_all}
    omood = {u['uid']: ev.oracle_mood(store, u) for u in users_all}

    seed_rows, main, gammas_by_seed = [], None, []
    for seed in SEEDS:
        tr, va, te = split_groups(all_uids, seed)
        tr_r = ratings[ratings.userId.isin(tr)]
        stats = H.build_item_stats(tr_r.row.values, tr_r.rating.values, tr_r.userId.values, N)
        priors = H.prior_components(stats)
        val_u = [by_uid[i] for i in sorted(va) if i in by_uid]
        tst_u = [by_uid[i] for i in sorted(te) if i in by_uid]

        def comps_of(us, g):
            return [H.user_components(store, stats, ev.taste_of(store, u), u['liked'], priors, cf_gamma=g) for u in us]

        def slots(us, cs, w):
            return [r for i in range(len(us)) for r in [x['row'] for x in H.rerank(store, H.combine(cs[i], w), us[i]['seen'], None, 0.0, K)]]

        # fit on VALIDATION users only: for each coverage requirement choose the popularity discount (gamma) and the
        # blend weights that give the best validation NDCG while meeting it
        comps_val = {g: comps_of(val_u, g) for g in GAMMAS}
        def fit(min_cov):
            cands = {g: tune(val_u, comps_val[g], min_cov=min_cov) for g in GAMMAS}
            g = max(cands, key=lambda x: (cands[x]['val_ndcg'], x))
            return g, cands[g], cands
        fits = {c: fit(c) for c in FRONTIER}
        gamma, main_fit, by_gamma = fits[MIN_COVERAGE]
        cv, ct = comps_val[gamma], comps_of(tst_u, gamma)
        tuned = {
            'full': main_fit,
            'no_cf': tune(val_u, cv, mask=('content', 'quality', 'pop'), min_cov=MIN_COVERAGE),
            'no_prior': tune(val_u, cv, mask=('content', 'cf'), min_cov=MIN_COVERAGE),
            'prior_only': tune(val_u, cv, mask=('quality', 'pop')),
        }
        comps_t = {g: comps_of(tst_u, g) for g in GAMMAS}
        fixed = {'content_only': {'content': 1.0, 'cf': 0, 'quality': 0, 'pop': 0},
                 'cf_only': {'content': 0, 'cf': 1.0, 'quality': 0, 'pop': 0},
                 'pop_only': {'content': 0, 'cf': 0, 'quality': 0, 'pop': 1.0}}

        def rank(i, us, cs, w, mood=None, lam=0.0):
            u = us[i]; base = H.combine(cs[i], w)
            return [x['row'] for x in H.rerank(store, base, u['seen'], store.mood_vectors.get(mood) if mood else None, lam, K)]

        def run(fn, us):
            lists = [fn(i) for i in range(len(us))]
            res = [ev.metrics(l, u, store, genres, topg[u['uid']]) for l, u in zip(lists, us)]
            out = {k: np.array([r[k] for r in res]) for k in ('hr', 'ndcg', 'div', 'seren')}
            flat = [r for l in lists for r in l]; cnt = np.bincount(flat, minlength=N)
            out['coverage'] = np.array([len(set(flat)) / len(flat)])                       # distinct titles per recommendation slot
            out['top100'] = np.array([np.sort(cnt)[::-1][:100].sum() / cnt.sum()])        # share of slots held by the 100 most-shown titles
            return out

        W = {k: v['weights'] for k, v in tuned.items()}
        systems = {
            'POP: most-liked films (train crowd)': run(lambda i: rank(i, tst_u, ct, fixed['pop_only']), tst_u),
            'B1: TF-IDF + genre (v1)': run(lambda i: v1.rank(tst_u[i], pmood[tst_u[i]['uid']], store), tst_u),
            'Content only (semantic centroid)': run(lambda i: rank(i, tst_u, ct, fixed['content_only']), tst_u),
            'Collaborative only': run(lambda i: rank(i, tst_u, ct, fixed['cf_only']), tst_u),
            'Quality + popularity prior only': run(lambda i: rank(i, tst_u, ct, W['prior_only']), tst_u),
            'Content + prior (no collaborative)': run(lambda i: rank(i, tst_u, ct, W['no_cf']), tst_u),
            'Content + collaborative (no prior)': run(lambda i: rank(i, tst_u, ct, W['no_prior']), tst_u),
            'Hybrid, no mood': run(lambda i: rank(i, tst_u, ct, W['full']), tst_u),
            'Hybrid tuned for accuracy only (no coverage requirement)': run(lambda i: rank(i, tst_u, comps_t[fits[0.0][0]], fits[0.0][1]['weights']), tst_u),
            f'Hybrid + mood re-rank (lambda={LAMBDA}), proxy mood': run(lambda i: rank(i, tst_u, ct, W['full'], pmood[tst_u[i]['uid']], LAMBDA), tst_u),
            f'Hybrid + mood re-rank (lambda={LAMBDA}), random mood': run(lambda i: rank(i, tst_u, ct, W['full'], rmood[tst_u[i]['uid']], LAMBDA), tst_u),
            f'v2.0 mood-shift, content only (lambda={LAMBDA}), proxy mood': run(lambda i: ev.emb_rank(store, tst_u[i], pmood[tst_u[i]['uid']], LAMBDA), tst_u),
        }
        frontier = {}
        for c, (g, fitc, _) in fits.items():
            m_ = run(lambda i: rank(i, tst_u, comps_t[g], fitc['weights']), tst_u)
            frontier[str(c)] = {'gamma': g, 'weights': fitc['weights'], 'val_ndcg': fitc['val_ndcg'], 'val_distinct_per_slot': fitc['val_distinct_per_slot'],
                                'hr': float(m_['hr'].mean()), 'ndcg': float(m_['ndcg'].mean()), 'coverage': float(m_['coverage'][0]), 'top100': float(m_['top100'][0])}
        seed_rows.append({'seed': seed, 'frontier': frontier, 'n_train_users': len(tr), 'n_val': len(val_u), 'n_test': len(tst_u),
                          'gamma': gamma, 'by_gamma': {str(g): {'val_ndcg': v['val_ndcg'], 'val_distinct_per_slot': v['val_distinct_per_slot'], 'weights': v['weights']} for g, v in by_gamma.items()},
                          'tuned': tuned, 'means': {k: {m: float(v[m].mean()) for m in v} for k, v in systems.items()}})
        print(f'seed {seed}: gamma {gamma} | val {len(val_u)} test {len(tst_u)} | full weights {W["full"]} val NDCG {tuned["full"]["val_ndcg"]:.4f} | '
              f'hybrid test HR {systems["Hybrid, no mood"]["hr"].mean():.3f} POP {systems["POP: most-liked films (train crowd)"]["hr"].mean():.3f} ({time.time() - t0:.0f}s)')

        gammas_by_seed.append(gamma)
        if seed == SEEDS[0]:
            # ---- main split: per-user arrays, lambda sweep, overlap, production stats
            sweep = []
            base_lists = [rank(i, tst_u, ct, W['full']) for i in range(len(tst_u))]
            for lam in SWEEP:
                for label, mm in (('proxy', pmood), ('random', rmood), ('oracle', omood)):
                    lists = [rank(i, tst_u, ct, W['full'], mm[u['uid']], lam) for i, u in enumerate(tst_u)]
                    res = [ev.metrics(l, u, store, genres, topg[u['uid']]) for l, u in zip(lists, tst_u)]
                    overlap = float(np.mean([len(set(a) & set(b)) / K for a, b in zip(lists, base_lists)]))
                    sweep.append({'lambda': lam, 'mood_source': label, 'hr': float(np.mean([r['hr'] for r in res])),
                                  'ndcg': float(np.mean([r['ndcg'] for r in res])), 'div': float(np.mean([r['div'] for r in res])),
                                  'overlap_with_no_mood': overlap})
            main = {'systems': systems, 'tuned': tuned, 'weights': W, 'sweep': sweep, 'n_test': len(tst_u), 'n_val': len(val_u)}

    # ---------------- summaries
    rng = np.random.default_rng(11)

    def boot(x, n=2000):
        x = np.asarray(x, float); idx = rng.integers(0, len(x), size=(n, len(x))); m = x[idx].mean(axis=1)
        return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))

    sysnames = list(main['systems'])
    table = []
    for name in sysnames:
        m = main['systems'][name]; row = {'system': name}
        for k in ('hr', 'ndcg', 'div', 'seren'):
            mean, lo, hi = boot(m[k]); row[k], row[k + '_lo'], row[k + '_hi'] = mean, lo, hi
        row['coverage'], row['top100'] = float(m['coverage'][0]), float(m['top100'][0])
        table.append(row)

    def paired(a, b):
        out = {}
        for k in ('hr', 'ndcg'):
            d = main['systems'][a][k] - main['systems'][b][k]; mean, lo, hi = boot(d)
            out[k] = {'diff': mean, 'lo': lo, 'hi': hi, 'better': int((d > 0).sum()), 'worse': int((d < 0).sum()), 'equal': int((d == 0).sum())}
        return out
    pro = f'Hybrid + mood re-rank (lambda={LAMBDA}), proxy mood'
    pairs = {'hybrid_vs_pop': paired('Hybrid, no mood', 'POP: most-liked films (train crowd)'),
             'hybrid_vs_v1': paired('Hybrid, no mood', 'B1: TF-IDF + genre (v1)'),
             'hybrid_vs_content': paired('Hybrid, no mood', 'Content only (semantic centroid)'),
             'mood_rerank_vs_hybrid': paired(pro, 'Hybrid, no mood'),
             'mood_rerank_vs_random_mood': paired(pro, f'Hybrid + mood re-rank (lambda={LAMBDA}), random mood')}

    multi = {}
    for name in sysnames:
        for m in ('hr', 'ndcg', 'coverage', 'top100'):
            vals = [s['means'][name][m] for s in seed_rows]
            multi.setdefault(name, {})[m] = {'mean': float(np.mean(vals)), 'sd': float(np.std(vals)), 'values': vals}

    # ---------------- production artefacts: stats from ALL ratings, weights from the main split
    allr = ratings
    stats_all = H.build_item_stats(allr.row.values, allr.rating.values, allr.userId.values, N)
    H.save_stats(stats_all, MODELS_DIR)
    # production weights: mean of the five splits' normalised tuned weights (each tuned only on its own validation users),
    # which is steadier than any single split and keeps a small content share for films with no ratings
    prod_gamma = float(np.median(gammas_by_seed))
    norm = [np.array([sr['by_gamma'][str(prod_gamma)]['weights'][k] for k in H.COMPONENTS]) for sr in seed_rows]
    norm = [w / w.sum() for w in norm]
    consensus = dict(zip(H.COMPONENTS, [round(float(x), 3) for x in np.mean(norm, axis=0)]))
    cfg = {'weights': consensus, 'cf_gamma': prod_gamma, 'candidates': H.CANDIDATES, 'default_lambda': LAMBDA,
           'tuned_on': 'mean of normalised weights tuned on the validation users of 5 random user splits (NDCG@10); gamma = median of the per-split choices',
           'stats_built_from': f'{len(allr)} MovieLens ratings, {allr.userId.nunique()} users'}
    with open(os.path.join(MODELS_DIR, H.CONFIG_FILE), 'w') as f:
        json.dump(cfg, f, indent=1)

    result = {'protocol': {'k': K, 'seeds': SEEDS, 'groups': '60/20/20 users (train/val/test)', 'weight_grid': GRID, 'lambda': LAMBDA, 'candidates': H.CANDIDATES},
              'main_split': {'n_test': main['n_test'], 'n_val': main['n_val'], 'tuned': main['tuned']},
              'systems': table, 'paired': pairs, 'gammas_chosen': gammas_by_seed, 'multi_seed': multi, 'seeds': seed_rows, 'sweep': main['sweep'], 'production_config': cfg}
    os.makedirs(OUT, exist_ok=True)
    json.dump(result, open(os.path.join(OUT, 'hybrid.json'), 'w'), indent=1)

    print('\n%-62s %7s %8s %8s %9s %9s' % ('system (seed 0 test users, n=%d)' % main['n_test'], 'HR@10', 'NDCG@10', 'Div@10', 'coverage', 'top100'))
    for r in table:
        print('%-62s %7.3f %8.3f %8.3f %9.3f %9.3f' % (r['system'], r['hr'], r['ndcg'], r['div'], r['coverage'], r['top100']))
    print('gamma chosen per split:', gammas_by_seed)
    print('\nmean +- sd over seeds (HR@10):')
    for n, v in multi.items():
        print('  %-62s %.3f +- %.3f' % (n, v['hr']['mean'], v['hr']['sd']))
    print('\nweights (full):', main['weights']['full'])
    for k, v in pairs.items():
        print(k, {m: (round(x['diff'], 4), round(x['lo'], 4), round(x['hi'], 4)) for m, x in v.items()})
    make_charts(table, main['sweep'])
    print(f'done in {time.time() - t0:.0f}s')


def make_charts(table, sweep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    NAVY, TEAL, ORANGE, GREY, BLUE = '#1F4E79', '#00897B', '#E67E22', '#8A8F98', '#7CA6C7'
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'axes.spines.top': False, 'axes.spines.right': False})
    short = {'POP': 'Popularity', 'B1': 'v1\nTF-IDF', 'Content only': 'Content\nonly', 'Collaborative only': 'Collab.\nonly',
             'Quality + popularity prior only': 'Prior\nonly', 'Content + prior': 'Content\n+ prior', 'Content + collaborative': 'Content\n+ collab.',
             'Hybrid, no mood': 'Hybrid\n(final)', 'Hybrid tuned for accuracy only': 'Hybrid\nacc. only', 'Hybrid + mood re-rank': 'Hybrid +\nmood', 'v2.0': 'v2.0 mood\nshift'}
    rows = [r for r in table if 'random mood' not in r['system']]
    lab = [next((v for k, v in short.items() if r['system'].startswith(k)), r['system'][:10]) for r in rows]
    cols = [NAVY if r['system'].startswith('Hybrid') else GREY for r in rows]
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for a, (k, t) in zip(ax, [('hr', 'Hit Rate@10'), ('ndcg', 'NDCG@10')]):
        v = [r[k] for r in rows]; err = [[r[k] - r[k + '_lo'] for r in rows], [r[k + '_hi'] - r[k] for r in rows]]
        a.bar(range(len(rows)), v, yerr=err, color=cols, capsize=3, width=.65)
        a.set_xticks(range(len(rows))); a.set_xticklabels(lab, fontsize=7.5); a.set_title(t + ' (held-out test users, 95% CI)', loc='left', fontweight='bold'); a.grid(axis='y', alpha=.25)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_7_3_hybrid.png'), dpi=160); plt.close(fig)

    df = pd.DataFrame(sweep)
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.0))
    for a, (k, t) in zip(ax, [('hr', 'Hit Rate@10'), ('ndcg', 'NDCG@10'), ('overlap_with_no_mood', 'Overlap with the no-mood list')]):
        for src, col, ls, lab_ in (('proxy', NAVY, '-', 'proxy mood'), ('random', GREY, '-', 'random mood'), ('oracle', ORANGE, '--', 'oracle (uses test items)')):
            d = df[df.mood_source == src]; a.plot(d['lambda'], d[k], marker='o', color=col, ls=ls, lw=2, label=lab_)
        a.set_xlabel('λ (mood re-ranking strength)'); a.set_title(t, loc='left', fontweight='bold'); a.grid(alpha=.25)
    ax[0].legend(frameon=False, fontsize=8)
    fig.tight_layout(); fig.savefig(os.path.join(FIG, 'fig_7_4_rerank.png'), dpi=160); plt.close(fig)
    print('charts written')


if __name__ == '__main__':
    main()

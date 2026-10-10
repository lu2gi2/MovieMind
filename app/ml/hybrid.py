"""
Hybrid recommender (v2.1): content embedding + collaborative signal + quality prior, with mood as a re-ranker.

    base(m | user) = w_c * z(cos(taste, m))            content: what the film is like
                   + w_cf * z( sum_j sim_cf(j, m) / (n_m + 10)^gamma )   collaborative, popularity-discounted
                   + w_q * z(bayes_rating(m))          quality prior
                   + w_p * z(log(1 + n_likes(m)))      popularity prior

    1. retrieve the top-N unseen movies by base score
    2. re-rank them: final = (1 - lam) * minmax(base) + lam * minmax(cos(m, mood_direction))
    3. apply the cluster diversity cap and return the top k

lam = 0 leaves the base ranking untouched, so a mood can only reorder films the engine already considers good.
All statistics (ratings, popularity, item-item similarity) come from a ratings table the caller chooses, which is how
the offline evaluation keeps them away from the users it evaluates.
"""
import json
import os
import numpy as np
import scipy.sparse as sp

LIKE_THRESHOLD = 4.0
COMPONENTS = ('content', 'cf', 'quality', 'pop')
DEFAULT_WEIGHTS = {'content': 1.0, 'cf': 1.0, 'quality': 0.5, 'pop': 0.5}
CANDIDATES = 60
STATS_FILE = 'item_stats.npz'
CONFIG_FILE = 'hybrid_config.json'


def zscore(v):
    v = np.asarray(v, dtype='float64')
    s = v.std()
    return (v - v.mean()) / s if s > 1e-12 else np.zeros_like(v)


def minmax(v):
    v = np.asarray(v, dtype='float64')
    lo, hi = v.min(), v.max()
    return (v - lo) / (hi - lo) if hi - lo > 1e-12 else np.zeros_like(v)


# ------------------------------------------------------------------ item statistics
def build_item_stats(item_rows, ratings, user_ids, n_items, k=100, shrink=10.0, prior_strength=20.0):
    """
    Item statistics from a set of ratings (arrays of equal length: item_rows, ratings, user_ids).

    Returns a dict of arrays:
      quality  Bayesian-smoothed mean rating per item (shrunk toward the global mean)
      pop      log(1 + number of likes) per item
      nbr_idx  (n_items, k) indices of each item's most similar items
      nbr_sim  (n_items, k) their similarity: co-likes / (sqrt(n_i * n_j) + shrink)
    """
    item_rows = np.asarray(item_rows); ratings = np.asarray(ratings, dtype='float64'); user_ids = np.asarray(user_ids)
    mu = ratings.mean() if len(ratings) else 3.5
    cnt = np.bincount(item_rows, minlength=n_items).astype('float64')
    tot = np.bincount(item_rows, weights=ratings, minlength=n_items)
    quality = (tot + prior_strength * mu) / (cnt + prior_strength)

    liked = ratings >= LIKE_THRESHOLD
    _, uidx = np.unique(user_ids[liked], return_inverse=True)
    M = sp.csr_matrix((np.ones(liked.sum(), dtype='float32'), (uidx, item_rows[liked])),
                      shape=(uidx.max() + 1 if liked.any() else 1, n_items))
    M.data[:] = 1.0
    n_like = np.asarray(M.sum(axis=0)).ravel().astype('float64')
    pop = np.log1p(n_like)

    co = (M.T @ M).toarray().astype('float32')
    np.fill_diagonal(co, 0.0)
    denom = np.sqrt(np.outer(n_like, n_like)).astype('float32') + shrink
    sim = co / denom
    np.fill_diagonal(sim, -1.0)                      # an item is never its own neighbour
    k = min(k, n_items - 1)
    idx = np.argpartition(-sim, k, axis=1)[:, :k]
    vals = np.maximum(np.take_along_axis(sim, idx, axis=1), 0.0)
    order = np.argsort(-vals, axis=1)
    return {'quality': quality.astype('float32'), 'pop': pop.astype('float32'),
            'nbr_idx': np.take_along_axis(idx, order, axis=1).astype('int32'),
            'nbr_sim': np.take_along_axis(vals, order, axis=1).astype('float32')}


def save_stats(stats, models_dir, name=STATS_FILE):
    np.savez_compressed(os.path.join(models_dir, name), **stats)


def load_stats(models_dir, name=STATS_FILE):
    path = os.path.join(models_dir, name)
    if not os.path.exists(path):
        return None
    with np.load(path) as z:
        return {k: z[k] for k in z.files}


def load_config(models_dir):
    path = os.path.join(models_dir, CONFIG_FILE)
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {'weights': dict(DEFAULT_WEIGHTS), 'candidates': CANDIDATES, 'default_lambda': 0.3, 'cf_gamma': 0.0}


# ------------------------------------------------------------------ scoring
def cf_scores(stats, liked_rows, n_items):
    """Sum of item-item similarities from the user's liked movies to every movie."""
    s = np.zeros(n_items, dtype='float64')
    if len(liked_rows):
        rows = np.asarray(liked_rows)
        np.add.at(s, stats['nbr_idx'][rows].ravel(), stats['nbr_sim'][rows].ravel())
    return s


def prior_components(stats):
    """User-independent z-scored quality and popularity (compute once per stats object)."""
    return {'quality': zscore(stats['quality']), 'pop': zscore(stats['pop'])}


def user_components(store, stats, taste_unit, liked_rows, priors=None, cf_gamma=0.0, cf_offset=10.0):
    """
    The four z-scored component vectors for one user.

    cf_gamma discounts the collaborative score by item popularity: cf / (n_likes + cf_offset) ** cf_gamma.
    A raw co-like sum favours films that many people liked, so everyone is shown the same few titles;
    gamma = 0 keeps the raw sum, larger values surface less-rated films that are still liked together.
    """
    priors = priors or prior_components(stats)
    content = zscore(store.E @ np.asarray(taste_unit, dtype='float32'))
    n = store.E.shape[0]
    if len(liked_rows):
        raw = cf_scores(stats, liked_rows, n)
        if cf_gamma:
            raw = raw / ((np.expm1(stats['pop'].astype('float64')) + cf_offset) ** cf_gamma)
        cf = zscore(raw)
    else:
        cf = np.zeros(n)
    return {'content': content, 'cf': cf, 'quality': priors['quality'], 'pop': priors['pop']}


def combine(comps, weights):
    return sum(weights.get(k, 0.0) * comps[k] for k in COMPONENTS)


def rerank(store, base, seen_rows, mood_dir=None, lam=0.3, k=10, n_candidates=CANDIDATES):
    """
    Retrieve by base score, re-rank the candidates by mood, apply the diversity cap.
    Returns a list of dicts: row, base, final, base_rank (position in the un-moded ranking), final_rank.
    """
    b = np.array(base, dtype='float64')
    if len(seen_rows):
        b[np.asarray(seen_rows)] = -np.inf
    order = np.argsort(-b)
    cand = order[:max(n_candidates, k)]
    cand = cand[np.isfinite(b[cand])]
    base_c = b[cand]
    if mood_dir is not None and lam > 0:
        mood_c = store.E[cand] @ np.asarray(mood_dir, dtype='float32')
        final = (1 - lam) * minmax(base_c) + lam * minmax(mood_c)
    else:
        final = minmax(base_c)
    seq = cand[np.argsort(-final, kind='stable')]
    final_by_row = dict(zip(cand.tolist(), final.tolist()))
    base_rank = {int(r): i + 1 for i, r in enumerate(cand)}

    cap, counts, picked = max(2, k // 3), {}, []
    for r in seq:
        c = int(store.clusters[r])
        if counts.get(c, 0) < cap:
            picked.append(int(r)); counts[c] = counts.get(c, 0) + 1
        if len(picked) == k:
            break
    if len(picked) < k:                                   # diversity cap left gaps: fill in order
        for r in seq:
            if int(r) not in picked:
                picked.append(int(r))
            if len(picked) == k:
                break
    return [{'row': r, 'base': float(b[r]), 'final': final_by_row[r], 'base_rank': base_rank[r], 'final_rank': i + 1}
            for i, r in enumerate(picked)]

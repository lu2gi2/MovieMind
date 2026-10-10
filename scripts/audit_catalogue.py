#!/usr/bin/env python3
"""
Audit how well every catalogue movie is connected to the rest of the system.

Checks: database <-> embeddings <-> v1 matrices alignment, metadata completeness, MovieLens rating coverage,
collaborative neighbours, mood / cluster membership, poster reachability (sampled), and whether each movie can
actually be recommended (simulated users).

    python scripts/audit_catalogue.py [--posters N] [--json out.json]
"""
import argparse
import json
import os
import pickle
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'scripts'))

from app import create_app
from app.models import Movie
from app.ml import hybrid as H
from app.ml.embeddings import EmbeddingStore, MODELS_DIR, MOODS, MOOD_KEYWORDS, mood_exemplar_rows


def pct(n, d):
    return f'{100 * n / d:.1f}%' if d else 'n/a'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--posters', type=int, default=40, help='number of poster URLs to probe over HTTP (0 = skip)')
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    out = {}
    app = create_app()

    with app.app_context():
        movies = Movie.query.order_by(Movie.movie_id).all()
    N = len(movies)
    ids = np.array([m.movie_id for m in movies])
    print(f'== 1. Alignment ==\ndatabase movies: {N}')
    store = EmbeddingStore.load()
    same = np.array_equal(ids, store.ids)
    print(f'embeddings: {len(store.ids)} rows | same ids and order as database: {same}')
    mp = pickle.load(open(os.path.join(MODELS_DIR, 'movie_id_map.pkl'), 'rb'))
    v1_ok = set(mp) == set(ids.tolist())
    print(f'v1 TF-IDF/genre matrices: {len(mp)} movies | same set: {v1_ok}')
    out['alignment'] = {'db': N, 'embeddings': int(len(store.ids)), 'ids_match': bool(same), 'v1_match': bool(v1_ok)}
    dup_t = pd.Series([m.title for m in movies]).duplicated().sum(); dup_tm = pd.Series([m.tmdb_id for m in movies]).duplicated().sum()
    print(f'duplicate titles: {dup_t} | duplicate tmdb ids: {dup_tm}')

    print('\n== 2. Metadata completeness ==')
    fields = {'overview': lambda m: bool(m.overview and m.overview.strip()), 'tagline': lambda m: bool(m.tagline),
              'genres': lambda m: bool(m.get_genres_list()), 'keywords': lambda m: bool(m.get_keywords_list()),
              'cast': lambda m: bool(m.get_cast_list()), 'director': lambda m: bool(m.director), 'poster': lambda m: bool(m.poster_path),
              'backdrop': lambda m: bool(m.backdrop_path), 'runtime': lambda m: bool(m.runtime), 'release_year': lambda m: bool(m.release_year),
              'imdb_id': lambda m: bool(m.imdb_id)}
    comp = {}
    for k, f in fields.items():
        n = sum(1 for m in movies if f(m)); comp[k] = n
        print(f'  {k:13s} {n:5d} / {N}  ({pct(n, N)})')
    out['completeness'] = comp
    rich = sum(1 for m in movies if fields['overview'](m) and fields['genres'](m) and fields['keywords'](m))
    print(f'  overview + genres + keywords all present: {rich} ({pct(rich, N)})')

    print('\n== 3. MovieLens ratings coverage ==')
    row_of = {int(i): r for r, i in enumerate(store.ids)}
    ml = pd.read_csv(os.path.join(ROOT, 'data/raw/ratings.csv'))
    total_ratings = len(ml); allmovies = ml.movieId.nunique()
    r = ml[ml.movieId.isin(row_of)].copy(); r['row'] = r.movieId.map(row_of)
    print(f'MovieLens: {total_ratings} ratings on {allmovies} movies; catalogue keeps {len(r)} ratings ({pct(len(r), total_ratings)}) on {r.movieId.nunique()} movies')
    cnt = np.bincount(r.row.values, minlength=N); likes = np.bincount(r[r.rating >= 4].row.values, minlength=N)
    for lo, hi, lab in [(0, 0, 'no ratings at all'), (1, 4, '1-4 ratings'), (5, 19, '5-19 ratings'), (20, 99, '20-99 ratings'), (100, 10 ** 9, '100+ ratings')]:
        n = int(((cnt >= lo) & (cnt <= hi)).sum()); print(f'  {lab:20s} {n:5d} ({pct(n, N)})')
    print(f'  movies with >= 1 like (rating>=4): {(likes >= 1).sum()} ({pct((likes >= 1).sum(), N)}); >= 5 likes: {(likes >= 5).sum()} ({pct((likes >= 5).sum(), N)})')
    out['ratings'] = {'ml_ratings': int(total_ratings), 'kept': int(len(r)), 'no_ratings': int((cnt == 0).sum()), 'lt5_likes': int((likes < 5).sum())}

    print('\n== 4. Collaborative neighbours ==')
    stats = H.load_stats(MODELS_DIR)
    if stats is None:
        print('  item_stats.npz missing (run scripts/build_item_stats.py)')
    else:
        nz = (stats['nbr_sim'] > 0).sum(axis=1)
        for lab, cond in [('no collaborative neighbour (cold items)', nz == 0), ('1-9 neighbours', (nz >= 1) & (nz < 10)), ('10+ neighbours', nz >= 10), ('full 100 neighbours', nz >= 100)]:
            print(f'  {lab:42s} {int(cond.sum()):5d} ({pct(int(cond.sum()), N)})')
        indeg = np.bincount(stats['nbr_idx'][stats['nbr_sim'] > 0].ravel(), minlength=N)
        print(f'  movies that appear in nobody\'s neighbour list: {(indeg == 0).sum()} ({pct((indeg == 0).sum(), N)})')
        out['cf'] = {'cold_items': int((nz == 0).sum()), 'never_a_neighbour': int((indeg == 0).sum())}

    print('\n== 5. Mood and cluster membership ==')
    kws = [m.get_keywords_list() for m in movies]
    member = np.zeros(N, dtype=int)
    for mood in MOODS:
        rows = mood_exemplar_rows(kws, mood); member[rows] += 1
    print(f'  movies that are an exemplar of at least one mood: {(member > 0).sum()} ({pct((member > 0).sum(), N)}); of none: {(member == 0).sum()}')
    sizes = np.bincount(store.clusters)
    print(f'  clusters: {len(sizes)}, sizes min/median/max = {sizes.min()}/{int(np.median(sizes))}/{sizes.max()}; unassigned movies: {int((store.clusters < 0).sum())}')
    mood_cos = store.E @ np.vstack([store.mood_vectors[m] for m in MOODS]).T
    print(f'  every movie has a mood-fit score for all {len(MOODS)} moods: {np.isfinite(mood_cos).all()}')

    print('\n== 6. Can each movie be recommended? (simulated users) ==')
    cfg = H.load_config(MODELS_DIR); priors = H.prior_components(stats)
    rng = np.random.default_rng(0)
    seen_in_recs = np.zeros(N, dtype=int); n_users = 600
    for _ in range(n_users):
        liked = rng.choice(np.where(likes >= 3)[0], size=int(rng.integers(3, 15)), replace=False).tolist()
        taste = store.centroid(store.ids[liked].tolist())
        comps = H.user_components(store, stats, taste, liked, priors)
        base = H.combine(comps, cfg['weights'])
        mood = MOODS[int(rng.integers(0, len(MOODS)))] if rng.random() < 0.6 else None
        for x in H.rerank(store, base, liked, store.mood_vectors[mood] if mood else None, 0.3 if mood else 0.0, 10, cfg.get('candidates', 60)):
            seen_in_recs[x['row']] += 1
    reach = int((seen_in_recs > 0).sum())
    top100 = np.sort(seen_in_recs)[::-1][:100].sum() / seen_in_recs.sum()
    print(f'  {n_users} simulated users x top-10: {reach} distinct movies recommended ({pct(reach, N)} of the catalogue)')
    print(f'  share of all recommendation slots taken by the 100 most-recommended movies: {100 * top100:.1f}%')
    never = np.where(seen_in_recs == 0)[0]
    never_cnt = cnt[never]
    print(f'  never recommended: {len(never)} movies; of these {(never_cnt == 0).sum()} have no ratings, median ratings = {int(np.median(never_cnt)) if len(never) else 0}')
    # content-only reachability for comparison
    seen_c = np.zeros(N, dtype=int)
    for _ in range(300):
        liked = rng.choice(np.where(likes >= 3)[0], size=8, replace=False).tolist()
        s = store.scores(store.centroid(store.ids[liked].tolist())); s[liked] = -np.inf
        seen_c[np.argsort(-s)[:10]] += 1
    print(f'  content-only ranking, 300 users: {(seen_c > 0).sum()} distinct movies ({pct((seen_c > 0).sum(), N)})')
    out['reachability'] = {'users': n_users, 'distinct_recommended': reach, 'share': reach / N, 'top100_slot_share': float(top100), 'never': int(len(never)),
                           'content_only_distinct': int((seen_c > 0).sum())}

    if a.posters:
        print(f'\n== 7. Poster URLs (sample of {a.posters}) ==')
        import requests
        with app.app_context():
            sample = [m for m in Movie.query.order_by(Movie.popularity.asc()).limit(2000).all() if m.poster_path]
        pick = [sample[i] for i in rng.choice(len(sample), size=min(a.posters, len(sample)), replace=False)]
        ok = bad = 0; failures = []
        for m in pick:
            try:
                st = requests.head(m.get_poster_url(), timeout=6, allow_redirects=True).status_code
            except Exception as e:
                st = 0
            if st == 200: ok += 1
            else: bad += 1; failures.append((m.title, st))
        print(f'  reachable: {ok}/{ok + bad}' + (f' | failures: {failures[:5]}' if failures else ''))
        out['posters_sample'] = {'ok': ok, 'bad': bad}

    if a.json:
        json.dump(out, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()

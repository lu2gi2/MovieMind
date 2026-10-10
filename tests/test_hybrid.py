"""Hybrid recommender: item statistics, blending, mood as a re-ranker, explanations."""
import numpy as np
import pytest

from app import create_app, db
from app.models import User, Movie, UserLike
from app.ml import hybrid as H
from app.ml.embeddings import EmbeddingStore, MOODS, unit
from app.ml.embedding_recommender import EmbeddingRecommender


@pytest.fixture
def store():
    rng = np.random.default_rng(3)
    E = np.vstack([unit(rng.normal(size=8)) for _ in range(40)]).astype('float32')
    moods = np.vstack([unit(rng.normal(size=8)) for _ in MOODS]).astype('float32')
    return EmbeddingStore(E, np.arange(1, 41), moods, np.arange(40))       # every film its own cluster: the cap never bites


@pytest.fixture
def stats():
    """Users 0-9 like items 0,1,2 together; users 10-19 like 3,4 together; item 5 is rated poorly."""
    rows, rat, usr = [], [], []
    for u in range(10):
        for i in (0, 1, 2):
            rows.append(i); rat.append(5.0); usr.append(u)
    for u in range(10, 20):
        for i in (3, 4):
            rows.append(i); rat.append(4.5); usr.append(u)
    for u in range(20):
        rows.append(5); rat.append(1.0); usr.append(u)
    return H.build_item_stats(rows, rat, usr, 40, k=10)


def test_item_stats_shapes_and_no_self_neighbours(stats):
    assert stats['nbr_idx'].shape == (40, 10) and stats['nbr_sim'].shape == (40, 10)
    assert all(i not in stats['nbr_idx'][i] for i in range(40))


def test_co_liked_items_are_neighbours(stats):
    assert {1, 2} <= set(stats['nbr_idx'][0][:2].tolist())
    assert not ({3, 4} & set(stats['nbr_idx'][0][:2].tolist()))


def test_quality_prior_shrinks_and_ranks_correctly(stats):
    q = stats['quality']
    assert q[5] < q[0] and q[0] > q[39]          # poorly rated < well rated; unrated item sits at the global mean


def test_popularity_counts_likes(stats):
    assert stats['pop'][0] >= stats['pop'][3] > stats['pop'][5] == 0     # item 5 is never liked (rated 1.0)


def test_cf_scores_favour_co_liked_items(stats):
    s = H.cf_scores(stats, [0], 40)
    assert s[1] > 0 and s[2] > 0 and s[3] == 0


def test_zscore_and_minmax():
    z = H.zscore([1, 2, 3, 4]); assert abs(z.mean()) < 1e-9 and abs(z.std() - 1) < 1e-9
    assert H.minmax([5, 5, 5]).sum() == 0 and H.minmax([0, 5, 10]).tolist() == [0, .5, 1]
    assert H.zscore([2, 2, 2]).sum() == 0


def _comps(store, stats, liked):
    return H.user_components(store, stats, store.centroid([i + 1 for i in liked]), liked)


def test_lambda_zero_keeps_base_order(store, stats):
    c = _comps(store, stats, [0]); base = H.combine(c, {'content': 1, 'cf': 1, 'quality': .5, 'pop': .5})
    plain = [x['row'] for x in H.rerank(store, base, [0], None, 0.0, 10)]
    zero = [x['row'] for x in H.rerank(store, base, [0], store.mood_vectors['happy'], 0.0, 10)]
    expected = [int(r) for r in np.argsort(-np.where(np.isin(np.arange(40), [0]), -np.inf, base))[:10]]
    assert plain == zero == expected


def test_mood_reorders_only_the_candidate_pool(store, stats):
    c = _comps(store, stats, [0]); base = H.combine(c, {'content': 1, 'cf': 1, 'quality': .5, 'pop': .5})
    pool = set(np.argsort(-np.where(np.arange(40) == 0, -np.inf, base))[:15].tolist())
    out = H.rerank(store, base, [0], store.mood_vectors['scared'], 1.0, 10, n_candidates=15)
    assert {x['row'] for x in out} <= pool                    # nothing outside the retrieved pool can appear
    shifted = [x['row'] for x in out]
    unshifted = [x['row'] for x in H.rerank(store, base, [0], None, 0.0, 10, n_candidates=15)]
    assert shifted != unshifted
    cos = store.E @ store.mood_vectors['scared']
    assert cos[shifted[0]] >= cos[unshifted[0]]               # the top pick leans toward the mood


def test_seen_rows_never_returned(store, stats):
    base = H.combine(_comps(store, stats, [0]), {'content': 1, 'cf': 1, 'quality': 0, 'pop': 0})
    rows = [x['row'] for x in H.rerank(store, base, [0, 1, 2], None, 0.0, 10)]
    assert not {0, 1, 2} & set(rows)


# ---- full recommender with the hybrid on a synthetic catalogue
@pytest.fixture
def app():
    app = create_app(); app.config['TESTING'] = True
    with app.app_context():
        db.create_all()
        for i in range(1, 41):
            db.session.add(Movie(movie_id=i, title=f'Movie {i}', genres='["Drama"]', vote_count=100, vote_average=7.0))
        u = User(username='h', email='h@x.com'); u.set_password('pw123456'); db.session.add(u); db.session.commit()
        yield app
        db.drop_all()


def test_hybrid_recommender_end_to_end(app, store, stats):
    u = User.query.first()
    db.session.add(UserLike(user_id=u.user_id, movie_id=1)); db.session.commit()      # likes item row 0
    rec = EmbeddingRecommender(store, stats=stats, config={'weights': {'content': .3, 'cf': 1.0, 'quality': .2, 'pop': .2}, 'candidates': 20})
    assert rec.engine == 'v2.1-hybrid'
    out = rec.get_recommendations(u.user_id, limit=5)
    assert len(out) == 5 and 1 not in [m['movie_id'] for m in out]
    assert {2, 3} & {m['movie_id'] for m in out}                                      # co-liked items (rows 1, 2) surface
    d = out[0]['explanation_detail']
    assert d['reason'] in ('content', 'cf', 'quality', 'pop') and set(d['components']) == {'content', 'cf', 'quality', 'pop'}
    assert out[0]['explanation']


def test_hybrid_explanation_mentions_mood_shift(app, store, stats):
    u = User.query.first()
    db.session.add(UserLike(user_id=u.user_id, movie_id=1)); db.session.commit()
    rec = EmbeddingRecommender(store, stats=stats, config={'weights': {'content': 1, 'cf': 1, 'quality': .5, 'pop': .5}, 'candidates': 30})
    plain = rec.get_recommendations(u.user_id, limit=8)
    moved = rec.get_recommendations(u.user_id, mood='scared', lam=1.0, limit=8)
    assert [m['movie_id'] for m in plain] != [m['movie_id'] for m in moved]
    assert any((m['explanation_detail']['places_moved_by_mood'] or 0) > 0 for m in moved)


def test_default_recommender_loads_real_stats_when_built():
    from app.ml.embeddings import EmbeddingStore as ES, MODELS_DIR
    if not (ES.available() and H.load_stats(MODELS_DIR)):
        pytest.skip('hybrid stats not built')
    cfg = H.load_config(MODELS_DIR)
    assert set(cfg['weights']) == set(H.COMPONENTS) and abs(sum(cfg['weights'].values()) - 1) < 0.02

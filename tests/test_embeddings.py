"""
Tests for the MovieMind v2 embedding space (report test cases TC-01 .. TC-12).

Logic tests use a tiny synthetic store so they are fast and deterministic; the
TC-01 / TC-02 / TC-09 / TC-12 checks run against the real built artefacts and are
skipped if scripts/build_embeddings.py has not been run.
"""
import time
import numpy as np
import pytest

from app import create_app, db
from app.models import User, Movie, UserFavorite, UserLike
from app.ml.embeddings import EmbeddingStore, MOODS, MOOD_KEYWORDS, unit
from app.ml.embedding_recommender import EmbeddingRecommender
from app.ml.explainer import geometric_explanation

real = pytest.mark.skipif(not EmbeddingStore.available(), reason='embeddings not built')


# ------------------------------------------------------------------ fixtures
@pytest.fixture
def synth():
    """60 unit vectors in 8-d forming 3 clear regions; one mood direction per MOODS entry."""
    rng = np.random.default_rng(0)
    centres = np.eye(8)[:3]
    E = np.vstack([unit(centres[i // 20] + 0.15 * rng.normal(size=8)) for i in range(60)]).astype('float32')
    moods = np.vstack([unit(rng.normal(size=8)) for _ in MOODS]).astype('float32')
    clusters = np.repeat([0, 1, 2], 20)
    return EmbeddingStore(E, np.arange(1, 61), moods, clusters)


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def world(app, synth):
    """DB with 60 movies matching the synthetic store and one user."""
    for i in range(1, 61):
        db.session.add(Movie(movie_id=i, title=f'Movie {i}', genres='["Drama"]', vote_count=100, vote_average=7.0))
    u = User(username='u', email='u@x.com'); u.set_password('pw123456')
    db.session.add(u); db.session.commit()
    return u, EmbeddingRecommender(synth)


def like(user, *ids):
    for i in ids:
        db.session.add(UserLike(user_id=user.user_id, movie_id=i))
    db.session.commit()


# ------------------------------------------------------------------ TC-01 .. TC-02
@real
def test_tc01_embedding_shape_and_norm():
    s = EmbeddingStore.load()
    assert s.E.ndim == 2 and s.E.shape[1] == 384 and s.E.shape[0] == len(s.ids)
    assert np.allclose(np.linalg.norm(s.E, axis=1), 1.0, atol=1e-4)
    assert len(s.mood_vectors) == 10
    assert all(abs(np.linalg.norm(v) - 1) < 1e-4 for v in s.mood_vectors.values())


@real
def test_tc02_synonyms_close_in_embedding_space_but_zero_in_tfidf():
    from sentence_transformers import SentenceTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    a, b, c = 'a bank heist', 'a robbery', 'a quiet romance in the countryside'
    m = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    ea, eb, ec = m.encode([a, b, c], normalize_embeddings=True)
    assert float(ea @ eb) > float(ea @ ec) + 0.1
    tf = TfidfVectorizer().fit_transform([a, b])
    assert cosine_similarity(tf[0], tf[1])[0, 0] < 0.2   # shares only the stop word 'a'


# ------------------------------------------------------------------ TC-03 .. TC-05 taste
def test_tc03_new_user_gets_recommendations(world):
    u, rec = world
    out = rec.get_recommendations(u.user_id, limit=5)
    assert len(out) == 5 and all('explanation_detail' in r for r in out)


def test_tc04_single_like_taste_equals_that_embedding(world, synth):
    u, _ = world
    like(u, 7)
    assert np.allclose(synth.centroid([7]), synth.embedding(7), atol=1e-5)


def test_tc05_unlike_recomputes_centroid_from_remaining(world, synth):
    u, _ = world
    from app.ml.taste_updater import refresh_embedding_taste
    like(u, 3, 25, 45)
    refresh_embedding_taste(u.user_id, synth)
    UserLike.query.filter_by(user_id=u.user_id, movie_id=25).delete(); db.session.commit()
    t = refresh_embedding_taste(u.user_id, synth)
    assert np.allclose(t, synth.centroid([3, 45]), atol=1e-5)
    assert np.allclose(np.array(User.query.get(u.user_id).get_taste_embedding()), t, atol=1e-4)


# ------------------------------------------------------------------ TC-06 .. TC-08 ranking
def test_tc06_lambda_zero_equals_no_mood(world):
    u, rec = world
    like(u, 2, 5)
    base = [r['movie_id'] for r in rec.get_recommendations(u.user_id, limit=10)]
    zero = [r['movie_id'] for r in rec.get_recommendations(u.user_id, mood='excited', lam=0.0, limit=10)]
    assert base == zero


def test_tc07_liked_movies_never_returned(world):
    u, rec = world
    like(u, 1, 2, 3, 21, 22)
    ids = {r['movie_id'] for r in rec.get_recommendations(u.user_id, mood='sad', lam=0.5, limit=30)}
    assert not ids & {1, 2, 3, 21, 22}


def test_tc08_diversity_cap_per_cluster(world, synth):
    u, rec = world
    like(u, 1)
    k = 9
    out = rec.get_recommendations(u.user_id, limit=k)
    counts = {}
    for r in out:
        c = int(synth.clusters[synth.id_to_row[r['movie_id']]])
        counts[c] = counts.get(c, 0) + 1
    assert max(counts.values()) <= max(2, k // 3)


# ------------------------------------------------------------------ TC-09 mood direction
def test_search_position_math(synth):
    taste = synth.embedding(1)
    pos = synth.search_position(taste, 'happy', 0.4)
    assert np.allclose(pos, unit(taste) + 0.4 * synth.mood_vectors['happy'], atol=1e-6)
    assert np.allclose(synth.search_position(taste, None, 0.9), unit(taste))


def test_tc09_mood_shift_raises_mood_aligned_movies(world, synth):
    u, rec = world
    like(u, 1)
    mood = 'sad'
    d = synth.mood_vectors[mood]
    aligned = int(np.argmax(synth.E @ d))            # row most aligned with the mood direction
    mid = int(synth.ids[aligned])
    if mid == 1:
        pytest.skip('degenerate synthetic case')

    def rank(lam):
        out = rec.get_recommendations(u.user_id, mood=mood, lam=lam, limit=59)
        order = [r['movie_id'] for r in out]
        return order.index(mid) if mid in order else 10 ** 6
    assert rank(1.0) <= rank(0.0)


# ------------------------------------------------------------------ TC-10 .. TC-11 API
@pytest.fixture
def client(app):
    return app.test_client()


@pytest.fixture
def logged_in(client, app):
    client.post('/api/auth/signup', json={'username': 'apiu', 'email': 'a@x.com', 'password': 'password123'})
    return client


@real
@pytest.mark.parametrize('bad', ['-0.1', '1.5', 'abc'])
def test_tc10_lambda_out_of_range_is_400(logged_in, bad):
    r = logged_in.get(f'/api/recommend?mood=happy&lambda={bad}')
    assert r.status_code == 400 and 'lambda' in r.get_json()['error']


@real
def test_unknown_mood_is_400(logged_in):
    assert logged_in.get('/api/recommend?mood=bogus').status_code == 400


def test_tc11_explanation_fields(world, synth):
    u, rec = world
    like(u, 4, 5)
    r = rec.get_recommendations(u.user_id, mood='happy', lam=0.5, limit=3)[0]
    d = r['explanation_detail']
    assert {'taste_match', 'mood_effect', 'nearest_liked', 'rank_without_mood', 'rank_with_mood', 'text'} <= set(d)
    assert d['nearest_liked']['title'] in ('Movie 4', 'Movie 5')
    assert isinstance(r['explanation'], str) and r['explanation']


def test_mood_effect_is_zero_without_mood(world, synth):
    u, _ = world
    taste = synth.embedding(10)
    d = geometric_explanation(synth, 30, taste, None, 0.3, [synth.id_to_row[10]], {10: 'Movie 10'}, 5, 5)
    assert abs(d['mood_effect']) < 1e-6


# ------------------------------------------------------------------ TC-12 latency
@real
def test_tc12_query_latency_is_small():
    s = EmbeddingStore.load()
    taste = s.centroid(list(s.ids[:10]))
    t = time.perf_counter()
    for _ in range(50):
        sims = s.scores(s.search_position(taste, 'excited', 0.3))
        np.argsort(-sims)[:10]
    per = (time.perf_counter() - t) / 50
    assert per < 0.05          # well under 50 ms for ~5k movies on CPU


def test_mood_rules_cover_all_moods():
    assert set(MOOD_KEYWORDS) == set(MOODS) and len(MOODS) == 10

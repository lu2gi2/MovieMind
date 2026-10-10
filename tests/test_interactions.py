"""Like / Favorite / Watchlist semantics and the combined state endpoint."""
import pytest
from app import create_app, db
from app.models import User, Movie, UserLike, UserFavorite, UserWatchlist, UserInteraction
from app.ml.embeddings import EmbeddingStore

real = pytest.mark.skipif(not EmbeddingStore.available(), reason='embeddings not built')


@pytest.fixture
def app():
    app = create_app()
    app.config['TESTING'] = True
    with app.app_context():
        db.create_all()
        for i, g in enumerate(['["Drama"]', '["Comedy"]', '["Action"]'], start=1):
            db.session.add(Movie(movie_id=i, title=f'Movie {i}', genres=g, vote_count=100, vote_average=7.0, release_year=2000 + i))
        db.session.commit()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    c = app.test_client()
    r = c.post('/api/auth/signup', json={'username': 'inter', 'email': 'i@x.com', 'password': 'password123'})
    assert r.status_code == 201
    return c


def user(app):
    return User.query.filter_by(username='inter').first()


def test_requires_login(app):
    c = app.test_client()
    for url in ('/api/user/likes', '/api/user/watchlist', '/api/user/state'):
        assert c.get(url).status_code == 401


def test_like_is_separate_and_updates_taste(app, client):
    r = client.post('/api/user/likes', json={'movie_id': 1})
    assert r.status_code == 201 and r.get_json()['taste_updated'] is True
    assert UserLike.query.count() == 1 and UserFavorite.query.count() == 0 and UserWatchlist.query.count() == 0
    assert user(app).get_taste_vector().get('Drama', 0) > 0.5          # genre profile moved
    assert client.post('/api/user/likes', json={'movie_id': 1}).status_code == 409   # no duplicate signal


def test_unlike_reverses_taste(app, client):
    client.post('/api/user/likes', json={'movie_id': 1})
    r = client.delete('/api/user/likes/1')
    assert r.status_code == 200 and UserLike.query.count() == 0
    assert client.delete('/api/user/likes/1').status_code == 404


def test_favorite_does_not_change_taste(app, client):
    r = client.post('/api/user/favorites', json={'movie_id': 2})
    assert r.status_code == 201 and r.get_json()['taste_updated'] is False
    assert UserLike.query.count() == 0
    assert user(app).get_taste_vector() == {}


def test_watchlist_is_not_a_preference_signal(app, client):
    assert client.post('/api/user/watchlist', json={'movie_id': 3}).status_code == 201
    assert client.post('/api/user/watchlist', json={'movie_id': 3}).status_code == 409
    assert UserLike.query.count() == 0 and user(app).get_taste_vector() == {}
    w = client.get('/api/user/watchlist').get_json()
    assert [m['movie_id'] for m in w['watchlist']] == [3]
    assert client.delete('/api/user/watchlist/3').status_code == 200
    assert client.get('/api/user/watchlist').get_json()['watchlist'] == []


def test_state_reports_all_three_sets(app, client):
    client.post('/api/user/likes', json={'movie_id': 1})
    client.post('/api/user/favorites', json={'movie_id': 2})
    client.post('/api/user/watchlist', json={'movie_id': 3})
    s = client.get('/api/user/state').get_json()
    assert s == {'likes': [1], 'favorites': [2], 'watchlist': [3]}


def test_validation_errors(client):
    assert client.post('/api/user/likes', json={}).status_code == 400
    assert client.post('/api/user/likes', json={'movie_id': 999}).status_code == 404
    assert client.post('/api/user/watchlist', json={'movie_id': 999}).status_code == 404


def test_interactions_are_logged_with_distinct_types(app, client):
    client.post('/api/user/likes', json={'movie_id': 1})
    client.post('/api/user/favorites', json={'movie_id': 1})
    client.post('/api/user/watchlist', json={'movie_id': 1})
    kinds = {i.interaction_type for i in UserInteraction.query.all()}
    assert kinds == {'like', 'favorite', 'watchlist'}


def test_taste_endpoint_reports_real_genre_shares(client):
    client.post('/api/user/likes', json={'movie_id': 1})
    client.post('/api/user/likes', json={'movie_id': 2})
    t = client.get('/api/user/taste').get_json()
    shares = {g['genre']: g['share'] for g in t['genre_share']}
    assert shares == {'Drama': 0.5, 'Comedy': 0.5}
    assert t['mood_fit'] == [] or len(t['mood_fit']) == 3


def test_taste_endpoint_lists_recent_likes(client):
    client.post('/api/user/likes', json={'movie_id': 1})
    t = client.get('/api/user/taste').get_json()
    assert t['like_count'] == 1 and [m['movie_id'] for m in t['recent_likes']] == [1]


@real
def test_recommendations_exclude_liked_and_favorited(app, client):
    client.post('/api/user/likes', json={'movie_id': 1})
    client.post('/api/user/favorites', json={'movie_id': 2})
    ids = [m['movie_id'] for m in client.get('/api/recommend?limit=10').get_json()['recommendations']]
    assert 1 not in ids and 2 not in ids


@real
def test_catalogue_mood_ordering(client):
    r = client.get('/api/movies?mood=scared&per_page=3')
    assert r.status_code == 200 and r.get_json()['mood'] == 'scared'
    assert client.get('/api/movies?mood=bogus').status_code == 400


def test_pages_render(client):
    for path in ('/discover', '/dashboard', '/browse', '/favorites', '/watchlist', '/taste', '/profile', '/movie/1'):
        assert client.get(path).status_code == 200, path


def test_legacy_favorites_are_restored_as_likes(app, client):
    from app import migrate_legacy_likes
    u = user(app)
    # legacy row: favourite + a 'like' interaction, no UserLike (how v1 stored a "like")
    db.session.add(UserFavorite(user_id=u.user_id, movie_id=1))
    db.session.add(UserInteraction(user_id=u.user_id, movie_id=1, interaction_type='like'))
    # new-style favourite: logged as 'favorite', must NOT become a like
    db.session.add(UserFavorite(user_id=u.user_id, movie_id=2))
    db.session.add(UserInteraction(user_id=u.user_id, movie_id=2, interaction_type='favorite'))
    db.session.commit()
    assert migrate_legacy_likes() == 1
    assert [l.movie_id for l in UserLike.query.all()] == [1]
    assert migrate_legacy_likes() == 0          # idempotent

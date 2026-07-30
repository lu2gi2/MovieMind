"""
Tests for API endpoints
"""
import pytest
import json
from app import create_app, db
from app.models import User, Movie, UserFavorite


@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'
    app.config['WTF_CSRF_ENABLED'] = False

    with app.app_context():
        db.create_all()
        yield app
        db.drop_all()


@pytest.fixture
def client(app):
    """Create test client"""
    return app.test_client()


@pytest.fixture
def sample_user(app):
    """Create a sample user"""
    with app.app_context():
        user = User(
            username='apitest',
            email='api@test.com'
        )
        user.set_password('password123')
        user.set_preferred_genres(['Action', 'Comedy'])
        db.session.add(user)
        db.session.commit()
        return {'id': user.user_id, 'email': 'api@test.com', 'password': 'password123'}


@pytest.fixture
def sample_movies(app):
    """Create sample movies"""
    with app.app_context():
        movies = []
        for i in range(5):
            movie = Movie(
                movie_id=i + 1,
                tmdb_id=1000 + i,
                title=f'Test Movie {i + 1}',
                release_year=2020 + i,
                overview=f'Description for movie {i + 1}',
                genres=json.dumps(['Action', 'Comedy'] if i % 2 == 0 else ['Drama', 'Romance']),
                vote_average=7.0 + i * 0.5,
                popularity=100 - i * 10
            )
            movies.append(movie)
            db.session.add(movie)
        db.session.commit()
        return [m.movie_id for m in movies]


def login(client, email, password):
    """Helper to login"""
    return client.post('/api/auth/login', json={
        'email': email,
        'password': password
    })


class TestAuthAPI:
    """Tests for authentication endpoints"""

    def test_signup_success(self, client):
        """Test successful signup"""
        response = client.post('/api/auth/signup', json={
            'username': 'newuser',
            'email': 'new@example.com',
            'password': 'password123',
            'preferred_genres': ['Action', 'Drama']
        })

        assert response.status_code == 201
        data = response.get_json()
        assert 'user' in data
        assert data['user']['username'] == 'newuser'

    def test_signup_duplicate_email(self, client, sample_user):
        """Test signup with existing email"""
        response = client.post('/api/auth/signup', json={
            'username': 'another',
            'email': 'api@test.com',  # Already exists
            'password': 'password123'
        })

        assert response.status_code == 409
        assert 'error' in response.get_json()

    def test_signup_missing_fields(self, client):
        """Test signup with missing fields"""
        response = client.post('/api/auth/signup', json={
            'username': 'incomplete'
            # Missing email and password
        })

        assert response.status_code == 400

    def test_login_success(self, client, sample_user):
        """Test successful login"""
        response = login(client, sample_user['email'], sample_user['password'])

        assert response.status_code == 200
        data = response.get_json()
        assert 'user' in data

    def test_login_wrong_password(self, client, sample_user):
        """Test login with wrong password"""
        response = login(client, sample_user['email'], 'wrongpassword')

        assert response.status_code == 401

    def test_login_nonexistent_user(self, client):
        """Test login with nonexistent email"""
        response = login(client, 'nonexistent@example.com', 'password')

        assert response.status_code == 401

    def test_logout(self, client, sample_user):
        """Test logout"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.post('/api/auth/logout')

        assert response.status_code == 200

    def test_check_auth_logged_in(self, client, sample_user):
        """Test auth check when logged in"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.get('/api/auth/check')

        assert response.status_code == 200
        data = response.get_json()
        assert data['authenticated'] is True

    def test_check_auth_logged_out(self, client):
        """Test auth check when logged out"""
        response = client.get('/api/auth/check')

        assert response.status_code == 200
        data = response.get_json()
        assert data['authenticated'] is False


class TestMoviesAPI:
    """Tests for movies endpoints"""

    def test_get_movies(self, client, sample_movies):
        """Test getting movie list"""
        response = client.get('/api/movies')

        assert response.status_code == 200
        data = response.get_json()
        assert 'movies' in data
        assert len(data['movies']) == 5

    def test_get_movies_pagination(self, client, sample_movies):
        """Test movie pagination"""
        response = client.get('/api/movies?page=1&per_page=2')

        assert response.status_code == 200
        data = response.get_json()
        assert len(data['movies']) == 2
        assert data['pages'] == 3  # 5 movies / 2 per page = 3 pages

    def test_get_movies_search(self, client, sample_movies):
        """Test movie search"""
        response = client.get('/api/movies?search=Movie%201')

        assert response.status_code == 200
        data = response.get_json()
        assert len(data['movies']) >= 1

    def test_get_movies_filter_genre(self, client, sample_movies):
        """Test filtering by genre"""
        response = client.get('/api/movies?genre=Action')

        assert response.status_code == 200
        data = response.get_json()
        # Should only get movies with Action genre
        for movie in data['movies']:
            assert 'Action' in movie['genres']

    def test_get_movie_detail(self, client, sample_movies):
        """Test getting single movie"""
        response = client.get(f'/api/movies/{sample_movies[0]}')

        assert response.status_code == 200
        data = response.get_json()
        assert 'movie' in data
        assert data['movie']['movie_id'] == sample_movies[0]

    def test_get_movie_not_found(self, client):
        """Test getting nonexistent movie"""
        response = client.get('/api/movies/99999')

        assert response.status_code == 404

    def test_get_genres(self, client, sample_movies):
        """Test getting genre list"""
        response = client.get('/api/genres')

        assert response.status_code == 200
        data = response.get_json()
        assert 'genres' in data
        assert 'Action' in data['genres']

    def test_get_moods(self, client):
        """Test getting mood list"""
        response = client.get('/api/moods')

        assert response.status_code == 200
        data = response.get_json()
        assert 'moods' in data
        assert len(data['moods']) == 10


class TestUserAPI:
    """Tests for user endpoints"""

    def test_get_profile(self, client, sample_user):
        """Test getting user profile"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.get('/api/user/profile')

        assert response.status_code == 200
        data = response.get_json()
        assert 'profile' in data
        assert data['profile']['username'] == 'apitest'

    def test_get_profile_unauthorized(self, client):
        """Test profile access without login"""
        response = client.get('/api/user/profile')

        assert response.status_code == 401

    def test_update_profile(self, client, sample_user):
        """Test updating profile"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.put('/api/user/profile', json={
            'preferred_genres': ['Horror', 'Thriller']
        })

        assert response.status_code == 200

    def test_get_favorites_empty(self, client, sample_user):
        """Test getting empty favorites"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.get('/api/user/favorites')

        assert response.status_code == 200
        data = response.get_json()
        assert data['favorites'] == []

    def test_add_favorite(self, client, sample_user, sample_movies):
        """Test adding a favorite"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.post('/api/user/favorites', json={
            'movie_id': sample_movies[0]
        })

        assert response.status_code == 201
        data = response.get_json()
        assert 'taste_updated' in data

    def test_add_favorite_duplicate(self, client, sample_user, sample_movies):
        """Test adding duplicate favorite"""
        login(client, sample_user['email'], sample_user['password'])

        # Add once
        client.post('/api/user/favorites', json={'movie_id': sample_movies[0]})

        # Try to add again
        response = client.post('/api/user/favorites', json={
            'movie_id': sample_movies[0]
        })

        assert response.status_code == 409

    def test_remove_favorite(self, client, sample_user, sample_movies):
        """Test removing a favorite"""
        login(client, sample_user['email'], sample_user['password'])

        # Add first
        client.post('/api/user/favorites', json={'movie_id': sample_movies[0]})

        # Remove
        response = client.delete(f'/api/user/favorites/{sample_movies[0]}')

        assert response.status_code == 200

    def test_get_taste(self, client, sample_user):
        """Test getting taste vector"""
        login(client, sample_user['email'], sample_user['password'])

        response = client.get('/api/user/taste')

        assert response.status_code == 200
        data = response.get_json()
        assert 'taste_vector' in data


class TestHealthAPI:
    """Tests for health endpoint"""

    def test_health_check(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')

        assert response.status_code == 200
        data = response.get_json()
        assert data['status'] == 'healthy'

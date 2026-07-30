"""
Tests for database models
"""
import pytest
import json
from app import create_app, db
from app.models import User, Movie, UserFavorite, UserInteraction


@pytest.fixture
def app():
    """Create application for testing"""
    app = create_app()
    app.config['TESTING'] = True
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///:memory:'

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
            username='testuser',
            email='test@example.com'
        )
        user.set_password('password123')
        user.set_preferred_genres(['Action', 'Comedy'])
        user.set_taste_vector({'Action': 0.8, 'Comedy': 0.7})
        db.session.add(user)
        db.session.commit()
        return user.user_id


@pytest.fixture
def sample_movie(app):
    """Create a sample movie"""
    with app.app_context():
        movie = Movie(
            movie_id=1,
            tmdb_id=12345,
            title='Test Movie',
            release_year=2023,
            overview='A test movie for testing purposes.',
            genres=json.dumps(['Action', 'Comedy']),
            cast=json.dumps(['Actor 1', 'Actor 2']),
            director='Test Director',
            vote_average=8.5,
            popularity=100.0
        )
        db.session.add(movie)
        db.session.commit()
        return movie.movie_id


class TestUserModel:
    """Tests for User model"""

    def test_create_user(self, app):
        """Test user creation"""
        with app.app_context():
            user = User(username='newuser', email='new@example.com')
            user.set_password('secret123')
            db.session.add(user)
            db.session.commit()

            assert user.user_id is not None
            assert user.username == 'newuser'
            assert user.email == 'new@example.com'

    def test_password_hashing(self, app):
        """Test password is hashed correctly"""
        with app.app_context():
            user = User(username='hashtest', email='hash@example.com')
            user.set_password('mypassword')

            assert user.password_hash != 'mypassword'
            assert user.check_password('mypassword') is True
            assert user.check_password('wrongpassword') is False

    def test_preferred_genres(self, app, sample_user):
        """Test genre preferences storage"""
        with app.app_context():
            user = User.query.get(sample_user)
            genres = user.get_preferred_genres()

            assert isinstance(genres, list)
            assert 'Action' in genres
            assert 'Comedy' in genres

    def test_taste_vector(self, app, sample_user):
        """Test taste vector storage and retrieval"""
        with app.app_context():
            user = User.query.get(sample_user)
            taste = user.get_taste_vector()

            assert isinstance(taste, dict)
            assert taste.get('Action') == 0.8
            assert taste.get('Comedy') == 0.7

    def test_update_taste_vector(self, app, sample_user):
        """Test taste vector update with EMA"""
        with app.app_context():
            user = User.query.get(sample_user)
            old_action = user.get_taste_vector().get('Action', 0.5)

            user.update_taste_vector(['Action', 'Drama'], alpha=0.2)
            db.session.commit()

            new_taste = user.get_taste_vector()
            # Action should increase (was 0.8, target 1.0)
            assert new_taste['Action'] > old_action
            # Drama should be added
            assert 'Drama' in new_taste

    def test_to_dict(self, app, sample_user):
        """Test user to_dict conversion"""
        with app.app_context():
            user = User.query.get(sample_user)
            data = user.to_dict()

            assert 'user_id' in data
            assert 'username' in data
            assert 'email' in data
            assert 'password_hash' not in data  # Should not expose password


class TestMovieModel:
    """Tests for Movie model"""

    def test_create_movie(self, app, sample_movie):
        """Test movie creation"""
        with app.app_context():
            movie = Movie.query.get(sample_movie)

            assert movie is not None
            assert movie.title == 'Test Movie'
            assert movie.release_year == 2023

    def test_get_genres_list(self, app, sample_movie):
        """Test genre list retrieval"""
        with app.app_context():
            movie = Movie.query.get(sample_movie)
            genres = movie.get_genres_list()

            assert isinstance(genres, list)
            assert 'Action' in genres
            assert 'Comedy' in genres

    def test_get_cast_list(self, app, sample_movie):
        """Test cast list retrieval"""
        with app.app_context():
            movie = Movie.query.get(sample_movie)
            cast = movie.get_cast_list()

            assert isinstance(cast, list)
            assert 'Actor 1' in cast

    def test_poster_url(self, app):
        """Test poster URL generation"""
        with app.app_context():
            movie = Movie(
                movie_id=2,
                title='Poster Test',
                poster_path='/abc123.jpg'
            )
            db.session.add(movie)
            db.session.commit()

            url = movie.get_poster_url()
            assert url is not None
            assert 'abc123.jpg' in url
            assert 'tmdb.org' in url

    def test_to_dict(self, app, sample_movie):
        """Test movie to_dict conversion"""
        with app.app_context():
            movie = Movie.query.get(sample_movie)
            data = movie.to_dict()

            assert 'movie_id' in data
            assert 'title' in data
            assert 'genres' in data
            assert isinstance(data['genres'], list)


class TestUserFavorite:
    """Tests for UserFavorite model"""

    def test_add_favorite(self, app, sample_user, sample_movie):
        """Test adding a favorite"""
        with app.app_context():
            favorite = UserFavorite(
                user_id=sample_user,
                movie_id=sample_movie
            )
            db.session.add(favorite)
            db.session.commit()

            assert favorite.id is not None
            assert favorite.user_id == sample_user
            assert favorite.movie_id == sample_movie

    def test_unique_constraint(self, app, sample_user, sample_movie):
        """Test that user can only favorite a movie once"""
        with app.app_context():
            fav1 = UserFavorite(user_id=sample_user, movie_id=sample_movie)
            db.session.add(fav1)
            db.session.commit()

            fav2 = UserFavorite(user_id=sample_user, movie_id=sample_movie)
            db.session.add(fav2)

            with pytest.raises(Exception):  # IntegrityError
                db.session.commit()


class TestUserInteraction:
    """Tests for UserInteraction model"""

    def test_create_interaction(self, app, sample_user, sample_movie):
        """Test creating an interaction"""
        with app.app_context():
            interaction = UserInteraction(
                user_id=sample_user,
                movie_id=sample_movie,
                interaction_type='like'
            )
            db.session.add(interaction)
            db.session.commit()

            assert interaction.id is not None
            assert interaction.interaction_type == 'like'
            assert interaction.timestamp is not None

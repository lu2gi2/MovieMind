"""
Tests for ML components - Vectorizer, Similarity, Recommender
"""
import pytest
import numpy as np
import json
from app import create_app, db
from app.models import User, Movie, UserFavorite
from app.ml.vectorizer import MovieVectorizer, create_combined_text
from app.ml.similarity import SimilarityEngine, cosine_sim
from app.ml.taste_updater import TasteUpdater
from app.ml.explainer import RecommendationExplainer


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
def sample_movies(app):
    """Create sample movies for testing"""
    with app.app_context():
        movies = [
            Movie(
                movie_id=1,
                title='Action Movie',
                overview='An exciting action movie with explosions.',
                genres=json.dumps(['Action', 'Thriller']),
                cast=json.dumps(['Actor 1']),
                director='Director 1',
                vote_average=7.5
            ),
            Movie(
                movie_id=2,
                title='Comedy Film',
                overview='A funny comedy film that makes you laugh.',
                genres=json.dumps(['Comedy', 'Romance']),
                cast=json.dumps(['Actor 2']),
                director='Director 2',
                vote_average=8.0
            ),
            Movie(
                movie_id=3,
                title='Action Comedy',
                overview='An action movie with comedy elements.',
                genres=json.dumps(['Action', 'Comedy']),
                cast=json.dumps(['Actor 3']),
                director='Director 3',
                vote_average=7.0
            ),
        ]
        for movie in movies:
            db.session.add(movie)
        db.session.commit()
        return [m.movie_id for m in movies]


@pytest.fixture
def sample_user(app):
    """Create sample user for testing"""
    with app.app_context():
        user = User(
            username='mltest',
            email='ml@test.com'
        )
        user.set_password('test123')
        user.set_preferred_genres(['Action', 'Comedy'])
        user.set_taste_vector({'Action': 0.8, 'Comedy': 0.6})
        db.session.add(user)
        db.session.commit()
        return user.user_id


class TestMovieVectorizer:
    """Tests for MovieVectorizer"""

    def test_genre_encoding(self):
        """Test genre one-hot encoding"""
        vectorizer = MovieVectorizer()

        genres_lists = [
            ['Action', 'Comedy'],
            ['Drama'],
            ['Action', 'Thriller']
        ]

        matrix = vectorizer.fit_transform_genres(genres_lists)

        assert matrix.shape[0] == 3  # 3 movies
        assert matrix.shape[1] == 4  # 4 unique genres
        assert 'Action' in vectorizer.all_genres
        assert 'Comedy' in vectorizer.all_genres

    def test_tfidf_vectorization(self):
        """Test TF-IDF vectorization"""
        # Create a custom vectorizer with lower min_df for testing
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = MovieVectorizer()
        vectorizer.tfidf_vectorizer = TfidfVectorizer(
            max_features=100,
            stop_words='english',
            min_df=1  # Lower for testing with few documents
        )

        plots = [
            'An action movie with exciting scenes and explosions.',
            'A comedy film with funny jokes and laughter.',
            'A drama about life and love and emotions.'
        ]

        matrix = vectorizer.fit_transform_plots(plots)

        assert matrix.shape[0] == 3  # 3 movies
        assert matrix.shape[1] > 0  # Some features
        # Matrix should be sparse
        assert hasattr(matrix, 'toarray')

    def test_combined_text(self, app, sample_movies):
        """Test combined text creation"""
        with app.app_context():
            movie = Movie.query.get(sample_movies[0])
            text = create_combined_text(movie)

            assert 'exciting' in text.lower()
            assert 'action' in text.lower()
            assert 'Director 1' in text


class TestSimilarityEngine:
    """Tests for SimilarityEngine"""

    def test_cosine_similarity(self):
        """Test cosine similarity calculation"""
        vec1 = [1, 0, 1]
        vec2 = [1, 1, 0]

        sim = cosine_sim(vec1, vec2)

        assert 0 <= sim <= 1
        assert sim == pytest.approx(0.5, rel=0.01)

    def test_identical_vectors(self):
        """Test similarity of identical vectors"""
        vec = [1, 2, 3]
        sim = cosine_sim(vec, vec)

        assert sim == pytest.approx(1.0, rel=0.01)

    def test_orthogonal_vectors(self):
        """Test similarity of orthogonal vectors"""
        vec1 = [1, 0, 0]
        vec2 = [0, 1, 0]

        sim = cosine_sim(vec1, vec2)

        assert sim == pytest.approx(0.0, rel=0.01)

    def test_zero_vector(self):
        """Test handling of zero vector"""
        vec1 = [0, 0, 0]
        vec2 = [1, 2, 3]

        sim = cosine_sim(vec1, vec2)

        assert sim == 0.0


class TestTasteUpdater:
    """Tests for TasteUpdater"""

    def test_update_on_like(self, app, sample_user, sample_movies):
        """Test taste update when liking a movie"""
        with app.app_context():
            movie = Movie.query.get(sample_movies[0])  # Action, Thriller
            user = User.query.get(sample_user)
            old_action = user.get_taste_vector().get('Action', 0.5)

            updater = TasteUpdater(sample_user)
            result = updater.update_on_like(movie)

            assert 'changes' in result
            assert 'taste_vector' in result
            # Action should increase (target is 1.0)
            new_action = result['taste_vector'].get('Action', 0)
            assert new_action >= old_action

    def test_update_on_unlike(self, app, sample_user, sample_movies):
        """Test taste update when unliking a movie"""
        with app.app_context():
            movie = Movie.query.get(sample_movies[0])

            updater = TasteUpdater(sample_user)
            # First like, then unlike
            updater.update_on_like(movie)
            old_taste = updater.taste_vector.copy()

            result = updater.update_on_unlike(movie)

            # Genres should move towards neutral (0.5)
            for genre in movie.get_genres_list():
                if genre in old_taste and old_taste[genre] > 0.5:
                    assert result['taste_vector'][genre] <= old_taste[genre]

    def test_get_top_genres(self, app, sample_user):
        """Test getting top genres"""
        with app.app_context():
            updater = TasteUpdater(sample_user)
            top = updater.get_top_genres(n=2)

            assert len(top) <= 2
            assert 'Action' in top  # Action has highest score (0.8)

    def test_taste_summary(self, app, sample_user):
        """Test taste summary generation"""
        with app.app_context():
            updater = TasteUpdater(sample_user)
            summary = updater.get_taste_summary()

            assert 'status' in summary
            assert 'top_genres' in summary
            assert 'message' in summary


class TestRecommendationExplainer:
    """Tests for RecommendationExplainer"""

    def test_explain_genre_match(self, app, sample_user, sample_movies):
        """Test genre-based explanation"""
        with app.app_context():
            movie = Movie.query.get(sample_movies[0])  # Action movie

            explainer = RecommendationExplainer(sample_user)
            explanation = explainer.explain(movie, similarity_score=0.5)

            assert len(explanation) > 0
            # Should mention Action since user prefers it
            assert 'Action' in explanation or 'action' in explanation.lower()

    def test_explain_mood(self, app, sample_movies):
        """Test mood-based explanation"""
        with app.app_context():
            movie = Movie.query.get(sample_movies[0])  # Action, Thriller

            explainer = RecommendationExplainer()
            explanation = explainer.explain(movie, mood='excited')

            assert len(explanation) > 0

    def test_explain_highly_rated(self, app):
        """Test rating-based explanation"""
        with app.app_context():
            movie = Movie(
                movie_id=99,
                title='Highly Rated',
                genres=json.dumps(['Drama']),
                vote_average=9.0
            )
            db.session.add(movie)
            db.session.commit()

            explainer = RecommendationExplainer()
            explanation = explainer.explain(movie)

            assert '9.0' in explanation or 'rated' in explanation.lower()

    def test_explain_taste_change(self, app, sample_user, sample_movies):
        """Test taste change explanation"""
        with app.app_context():
            changes = {
                'Action': {'old': 0.7, 'new': 0.8, 'change': 0.1},
                'Comedy': {'old': 0.5, 'new': 0.6, 'change': 0.1}
            }

            explainer = RecommendationExplainer(sample_user)
            message = explainer.explain_taste_change(changes, 'Test Movie')

            assert len(message) > 0
            assert 'Action' in message or 'taste' in message.lower()

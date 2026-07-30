"""Movie model for storing movie data"""
import json
from app import db


class Movie(db.Model):
    """Movie model with TMDB enriched data"""
    __tablename__ = 'movies'

    movie_id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True, index=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    original_title = db.Column(db.String(255))
    release_year = db.Column(db.Integer, index=True)
    overview = db.Column(db.Text)
    tagline = db.Column(db.String(500))

    # Stored as JSON strings
    genres = db.Column(db.Text)
    cast = db.Column(db.Text)
    keywords = db.Column(db.Text)

    director = db.Column(db.String(100))
    runtime = db.Column(db.Integer)
    original_language = db.Column(db.String(10))

    # Ratings and popularity
    vote_average = db.Column(db.Float, default=0)
    vote_count = db.Column(db.Integer, default=0)
    popularity = db.Column(db.Float, default=0)

    # Images
    poster_path = db.Column(db.String(255))
    backdrop_path = db.Column(db.String(255))

    # ML feature vectors (stored as JSON)
    genre_vector = db.Column(db.Text)
    feature_vector = db.Column(db.Text)

    def get_genres_list(self):
        """Return genres as a Python list"""
        if not self.genres:
            return []
        try:
            return json.loads(self.genres)
        except json.JSONDecodeError:
            return []

    def get_cast_list(self):
        """Return cast as a Python list"""
        if not self.cast:
            return []
        try:
            return json.loads(self.cast)
        except json.JSONDecodeError:
            return []

    def get_keywords_list(self):
        """Return keywords as a Python list"""
        if not self.keywords:
            return []
        try:
            return json.loads(self.keywords)
        except json.JSONDecodeError:
            return []

    def get_poster_url(self, size='w500'):
        """Return full poster URL"""
        if self.poster_path:
            return f"https://image.tmdb.org/t/p/{size}{self.poster_path}"
        return None

    def get_backdrop_url(self, size='w1280'):
        """Return full backdrop URL"""
        if self.backdrop_path:
            return f"https://image.tmdb.org/t/p/{size}{self.backdrop_path}"
        return None

    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            'movie_id': self.movie_id,
            'tmdb_id': self.tmdb_id,
            'title': self.title,
            'release_year': self.release_year,
            'overview': self.overview,
            'tagline': self.tagline,
            'genres': self.get_genres_list(),
            'cast': self.get_cast_list(),
            'director': self.director,
            'runtime': self.runtime,
            'vote_average': self.vote_average,
            'popularity': self.popularity,
            'poster_url': self.get_poster_url(),
            'backdrop_url': self.get_backdrop_url()
        }

    def __repr__(self):
        return f'<Movie {self.title} ({self.release_year})>'

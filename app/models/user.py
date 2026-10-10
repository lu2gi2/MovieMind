"""User models for authentication and preferences"""
import json
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import UserMixin
from app import db, login_manager


class User(UserMixin, db.Model):
    """User model with preferences and taste vector"""
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False, index=True)
    email = db.Column(db.String(100), unique=True, nullable=False, index=True)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # User preferences (collected during signup)
    preferred_genres = db.Column(db.Text)  # JSON list
    preferred_industry = db.Column(db.String(50))  # Hollywood, Bollywood, etc.
    preferred_format = db.Column(db.String(20))  # movies, series, both

    # Adaptive learning: taste vector (updated as user interacts)
    taste_vector = db.Column(db.Text)  # JSON dict: {genre: weight}  (v1 genre profile)

    # v2: centroid of liked-movie embeddings (JSON list of 384 floats)
    taste_embedding = db.Column(db.Text)

    # Relationships
    favorites = db.relationship('UserFavorite', backref='user', lazy='dynamic',
                                cascade='all, delete-orphan')
    likes = db.relationship('UserLike', backref='user', lazy='dynamic',
                            cascade='all, delete-orphan')
    watchlist = db.relationship('UserWatchlist', backref='user', lazy='dynamic',
                                cascade='all, delete-orphan')
    interactions = db.relationship('UserInteraction', backref='user', lazy='dynamic',
                                   cascade='all, delete-orphan')

    def get_id(self):
        """Required for Flask-Login"""
        return str(self.user_id)

    def set_password(self, password):
        """Hash and set password"""
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        """Verify password"""
        return check_password_hash(self.password_hash, password)

    def get_preferred_genres(self):
        """Return preferred genres as list"""
        if not self.preferred_genres:
            return []
        try:
            return json.loads(self.preferred_genres)
        except json.JSONDecodeError:
            return []

    def set_preferred_genres(self, genres_list):
        """Set preferred genres from list"""
        self.preferred_genres = json.dumps(genres_list)

    def get_taste_vector(self):
        """Return taste vector as dict"""
        if not self.taste_vector:
            return {}
        try:
            return json.loads(self.taste_vector)
        except json.JSONDecodeError:
            return {}

    def set_taste_vector(self, vector_dict):
        """Set taste vector from dict"""
        self.taste_vector = json.dumps(vector_dict)

    def get_taste_embedding(self):
        """Return the v2 taste centroid as a list of floats ([] if not set)"""
        if not self.taste_embedding:
            return []
        try:
            return json.loads(self.taste_embedding)
        except json.JSONDecodeError:
            return []

    def set_taste_embedding(self, vector):
        """Store the v2 taste centroid (iterable of floats)"""
        self.taste_embedding = json.dumps([round(float(x), 6) for x in vector])

    def update_taste_vector(self, genres, alpha=0.2):
        """
        Update taste vector using Exponential Moving Average

        Args:
            genres: List of genres from a liked movie
            alpha: Learning rate (0.2 = 20% weight to new data)
        """
        current_taste = self.get_taste_vector()

        for genre in genres:
            old_value = current_taste.get(genre, 0.5)
            new_value = (1 - alpha) * old_value + alpha * 1.0
            current_taste[genre] = round(new_value, 4)

        self.set_taste_vector(current_taste)

    def to_dict(self):
        """Convert to dictionary for JSON responses"""
        return {
            'user_id': self.user_id,
            'username': self.username,
            'email': self.email,
            'preferred_genres': self.get_preferred_genres(),
            'preferred_industry': self.preferred_industry,
            'preferred_format': self.preferred_format,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

    def __repr__(self):
        return f'<User {self.username}>'


class UserFavorite(db.Model):
    """User's favorite/liked movies"""
    __tablename__ = 'user_favorites'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, index=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False, index=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Unique constraint: user can only favorite a movie once
    __table_args__ = (
        db.UniqueConstraint('user_id', 'movie_id', name='unique_user_movie_favorite'),
    )

    # Relationship to movie
    movie = db.relationship('Movie', backref='favorited_by')

    def __repr__(self):
        return f'<UserFavorite user={self.user_id} movie={self.movie_id}>'


class UserLike(db.Model):
    """A positive preference signal. Drives the taste profile (genre profile and embedding centroid)."""
    __tablename__ = 'user_likes'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, index=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False, index=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'movie_id', name='unique_user_movie_like'),
    )

    movie = db.relationship('Movie', backref='liked_by')

    def __repr__(self):
        return f'<UserLike user={self.user_id} movie={self.movie_id}>'


class UserWatchlist(db.Model):
    """Titles saved for later. Not a preference signal."""
    __tablename__ = 'user_watchlist'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, index=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False, index=True)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('user_id', 'movie_id', name='unique_user_movie_watchlist'),
    )

    movie = db.relationship('Movie', backref='watchlisted_by')

    def __repr__(self):
        return f'<UserWatchlist user={self.user_id} movie={self.movie_id}>'


class UserInteraction(db.Model):
    """Track all user interactions for analytics and learning"""
    __tablename__ = 'user_interactions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False, index=True)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False, index=True)
    interaction_type = db.Column(db.String(20), nullable=False)  # 'view', 'like', 'unlike', 'favorite', 'unfavorite', 'watchlist', 'unwatchlist', 'rate'
    rating = db.Column(db.Float)  # Optional: 1-5 star rating
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationship to movie
    movie = db.relationship('Movie', backref='interactions')

    def __repr__(self):
        return f'<UserInteraction {self.interaction_type} user={self.user_id} movie={self.movie_id}>'


@login_manager.user_loader
def load_user(user_id):
    """Load user by ID for Flask-Login"""
    return User.query.get(int(user_id))

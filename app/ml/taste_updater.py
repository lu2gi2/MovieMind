"""
Taste Updater - Adaptive learning from user interactions

Uses Exponential Moving Average (EMA) to update user taste vectors
based on their likes, views, and ratings.
"""
import json
from datetime import datetime, timedelta
from app import db
from app.models import User, Movie, UserFavorite, UserInteraction


class TasteUpdater:
    """
    Manages user taste vector updates using adaptive learning.

    The taste vector represents user preferences for each genre on a 0-1 scale:
    - 0.0 = Strong dislike / Never watches
    - 0.5 = Neutral / No data
    - 1.0 = Strong preference / Frequently watches

    Learning rates:
    - Like: α = 0.2 (20% weight to new data)
    - Unlike: α = 0.15 (15% negative adjustment)
    - View: α = 0.05 (5% small positive signal)
    """

    # Learning rates for different interaction types
    LEARNING_RATES = {
        'like': 0.20,      # Strong positive signal
        'unlike': 0.15,    # Moderate negative signal
        'view': 0.05,      # Weak positive signal
        'rate_high': 0.25, # Rating >= 4 stars
        'rate_low': 0.10,  # Rating < 3 stars
    }

    # Decay rate for inactive genres (applied periodically)
    DECAY_RATE = 0.02
    DECAY_THRESHOLD_DAYS = 30

    def __init__(self, user_id):
        self.user = User.query.get(user_id)
        if not self.user:
            raise ValueError(f"User {user_id} not found")
        self.taste_vector = self.user.get_taste_vector()

    def update_on_like(self, movie):
        """
        Update taste vector when user likes a movie.

        Args:
            movie: Movie object that was liked

        Returns:
            dict: Updated taste vector and changes made
        """
        genres = movie.get_genres_list()
        alpha = self.LEARNING_RATES['like']

        changes = {}
        for genre in genres:
            old_value = self.taste_vector.get(genre, 0.5)
            # EMA formula: new = (1 - α) * old + α * target
            # Target is 1.0 for likes (maximum preference)
            new_value = (1 - alpha) * old_value + alpha * 1.0
            new_value = round(min(new_value, 1.0), 4)

            if new_value != old_value:
                changes[genre] = {
                    'old': old_value,
                    'new': new_value,
                    'change': round(new_value - old_value, 4)
                }

            self.taste_vector[genre] = new_value

        self._save()

        return {
            'taste_vector': self.taste_vector,
            'changes': changes,
            'interaction': 'like',
            'movie': movie.title
        }

    def update_on_unlike(self, movie):
        """
        Update taste vector when user unlikes/removes a movie.

        Args:
            movie: Movie object that was unliked

        Returns:
            dict: Updated taste vector and changes made
        """
        genres = movie.get_genres_list()
        alpha = self.LEARNING_RATES['unlike']

        changes = {}
        for genre in genres:
            old_value = self.taste_vector.get(genre, 0.5)
            # Move towards neutral (0.5) when unliking
            new_value = (1 - alpha) * old_value + alpha * 0.5
            new_value = round(max(new_value, 0.0), 4)

            if new_value != old_value:
                changes[genre] = {
                    'old': old_value,
                    'new': new_value,
                    'change': round(new_value - old_value, 4)
                }

            self.taste_vector[genre] = new_value

        self._save()

        return {
            'taste_vector': self.taste_vector,
            'changes': changes,
            'interaction': 'unlike',
            'movie': movie.title
        }

    def update_on_view(self, movie):
        """
        Update taste vector when user views a movie (weak signal).

        Args:
            movie: Movie object that was viewed
        """
        genres = movie.get_genres_list()
        alpha = self.LEARNING_RATES['view']

        for genre in genres:
            old_value = self.taste_vector.get(genre, 0.5)
            # Slight increase towards preference
            new_value = (1 - alpha) * old_value + alpha * 0.7
            self.taste_vector[genre] = round(min(new_value, 1.0), 4)

        self._save()

    def update_on_rating(self, movie, rating):
        """
        Update taste vector based on user rating.

        Args:
            movie: Movie object
            rating: Float between 1 and 5
        """
        genres = movie.get_genres_list()

        # Determine learning rate and target based on rating
        if rating >= 4:
            alpha = self.LEARNING_RATES['rate_high']
            target = 0.9
        elif rating >= 3:
            alpha = self.LEARNING_RATES['view']
            target = 0.6
        else:
            alpha = self.LEARNING_RATES['rate_low']
            target = 0.3

        for genre in genres:
            old_value = self.taste_vector.get(genre, 0.5)
            new_value = (1 - alpha) * old_value + alpha * target
            self.taste_vector[genre] = round(max(0, min(new_value, 1.0)), 4)

        self._save()

    def apply_decay(self):
        """
        Apply decay to genres that haven't been interacted with recently.
        This prevents stale preferences from dominating.
        """
        # Get recent interactions
        threshold = datetime.utcnow() - timedelta(days=self.DECAY_THRESHOLD_DAYS)
        recent_interactions = UserInteraction.query.filter(
            UserInteraction.user_id == self.user.user_id,
            UserInteraction.timestamp >= threshold
        ).all()

        # Get genres from recent interactions
        recent_genres = set()
        for interaction in recent_interactions:
            movie = Movie.query.get(interaction.movie_id)
            if movie:
                recent_genres.update(movie.get_genres_list())

        # Decay genres not in recent interactions
        for genre, value in self.taste_vector.items():
            if genre not in recent_genres and value > 0.5:
                # Decay towards neutral
                new_value = value - self.DECAY_RATE
                self.taste_vector[genre] = round(max(new_value, 0.5), 4)

        self._save()

    def get_top_genres(self, n=5):
        """Get user's top N preferred genres."""
        sorted_genres = sorted(
            self.taste_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return [genre for genre, score in sorted_genres[:n]]

    def get_taste_summary(self):
        """
        Get a human-readable summary of user's taste.

        Returns:
            dict: Summary with top genres, preferences, etc.
        """
        if not self.taste_vector:
            return {
                'status': 'new_user',
                'message': 'Start liking movies to build your taste profile!',
                'top_genres': [],
                'taste_vector': {}
            }

        sorted_genres = sorted(
            self.taste_vector.items(),
            key=lambda x: x[1],
            reverse=True
        )

        top_genres = [g for g, s in sorted_genres if s > 0.6][:5]
        avoided_genres = [g for g, s in sorted_genres if s < 0.4]

        return {
            'status': 'active',
            'top_genres': top_genres,
            'avoided_genres': avoided_genres,
            'total_genres': len(self.taste_vector),
            'taste_vector': dict(sorted_genres),
            'message': f"You love {', '.join(top_genres[:3])}!" if top_genres else "Keep liking movies to refine your taste."
        }

    def _save(self):
        """Save taste vector to database."""
        self.user.set_taste_vector(self.taste_vector)
        db.session.commit()


def update_user_taste(user_id, movie_id, interaction_type, rating=None):
    """
    Convenience function to update user taste.

    Args:
        user_id: User ID
        movie_id: Movie ID
        interaction_type: 'like', 'unlike', 'view', or 'rate'
        rating: Optional rating (1-5) if interaction_type is 'rate'

    Returns:
        dict: Update result with changes
    """
    movie = Movie.query.get(movie_id)
    if not movie:
        return {'error': 'Movie not found'}

    updater = TasteUpdater(user_id)

    if interaction_type == 'like':
        return updater.update_on_like(movie)
    elif interaction_type == 'unlike':
        return updater.update_on_unlike(movie)
    elif interaction_type == 'view':
        updater.update_on_view(movie)
        return {'status': 'updated'}
    elif interaction_type == 'rate' and rating is not None:
        updater.update_on_rating(movie, rating)
        return {'status': 'updated'}

    return {'error': 'Invalid interaction type'}


def refresh_embedding_taste(user_id, store=None):
    """
    v2 taste: unit centroid of the embeddings of every movie the user currently likes.

    Always recomputed from the favourites table, so an unlike never has to "subtract"
    a vector. A user with no likes gets the global mean embedding (cold start).
    Returns the centroid as a NumPy array, or None if embeddings are not built.
    """
    from app.ml.embeddings import get_store
    store = store or get_store()
    user = User.query.get(user_id)
    if store is None or user is None:
        return None
    liked = [f.movie_id for f in UserFavorite.query.filter_by(user_id=user_id).all()]
    taste = store.centroid(liked)
    user.set_taste_embedding(taste)
    db.session.commit()
    return taste

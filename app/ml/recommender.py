"""
Recommendation Engine - Generate personalized movie recommendations
"""
import numpy as np
from app import db
from app.models import Movie, User, UserFavorite
from app.ml.similarity import SimilarityEngine


class RecommendationEngine:
    """
    Hybrid recommendation engine combining:
    - Content-based filtering (plot similarity)
    - Genre matching
    - Mood-based boosting
    - User taste vector
    """

    # Mood to genre mapping with boost factors
    MOOD_GENRE_MAP = {
        'happy': {
            'Comedy': 1.5, 'Animation': 1.3, 'Family': 1.3,
            'Adventure': 1.2, 'Fantasy': 1.1, 'Music': 1.2
        },
        'sad': {
            'Drama': 1.4, 'Romance': 1.3, 'Music': 1.2,
            'Documentary': 1.1
        },
        'excited': {
            'Action': 1.5, 'Adventure': 1.4, 'Thriller': 1.3,
            'Science Fiction': 1.2, 'War': 1.1
        },
        'relaxed': {
            'Comedy': 1.3, 'Animation': 1.3, 'Family': 1.2,
            'Documentary': 1.2, 'Music': 1.2
        },
        'romantic': {
            'Romance': 1.6, 'Drama': 1.3, 'Comedy': 1.2,
            'Music': 1.1
        },
        'thrilled': {
            'Thriller': 1.5, 'Horror': 1.4, 'Mystery': 1.4,
            'Crime': 1.3, 'Action': 1.2
        },
        'nostalgic': {
            'Drama': 1.3, 'Family': 1.3, 'Animation': 1.2,
            'Comedy': 1.2, 'Romance': 1.2
        },
        'adventurous': {
            'Adventure': 1.5, 'Action': 1.4, 'Science Fiction': 1.3,
            'Fantasy': 1.3, 'Western': 1.2
        },
        'thoughtful': {
            'Documentary': 1.5, 'Drama': 1.4, 'History': 1.3,
            'Biography': 1.3, 'Mystery': 1.2
        },
        'scared': {
            'Horror': 1.6, 'Thriller': 1.4, 'Mystery': 1.3
        }
    }

    def __init__(self):
        self.similarity_engine = SimilarityEngine()
        self._loaded = False

    def load(self):
        """Load similarity matrices"""
        if not self._loaded:
            self.similarity_engine.load_matrices()
            self._loaded = True
        return self

    def get_recommendations(self, user_id, mood=None, limit=10, exclude_favorites=True):
        """
        Generate personalized recommendations for a user

        Args:
            user_id: User ID
            mood: Optional mood string (e.g., 'happy', 'sad', 'excited')
            limit: Number of recommendations to return
            exclude_favorites: Whether to exclude already favorited movies

        Returns:
            List of dicts with movie info and explanation
        """
        if not self._loaded:
            self.load()

        user = User.query.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Get user's favorite movies
        favorite_ids = []
        if exclude_favorites:
            favorites = UserFavorite.query.filter_by(user_id=user_id).all()
            favorite_ids = [f.movie_id for f in favorites]

        # Get user's taste vector
        taste_vector = user.get_taste_vector()

        # If user has favorites, use content-based recommendations
        if favorite_ids:
            candidates = self._get_content_based_candidates(favorite_ids, limit * 3)
        else:
            # Cold start: use popular movies matching preferred genres
            candidates = self._get_cold_start_candidates(user, limit * 3)

        # Score and rank candidates
        scored_movies = []
        for candidate in candidates:
            movie = Movie.query.get(candidate['movie_id'])
            if not movie:
                continue

            # Skip favorites
            if movie.movie_id in favorite_ids:
                continue

            # Calculate final score
            score, explanation = self._calculate_score(
                movie=movie,
                base_similarity=candidate.get('similarity', 0.5),
                taste_vector=taste_vector,
                mood=mood,
                favorite_ids=favorite_ids
            )

            scored_movies.append({
                'movie': movie,
                'score': score,
                'explanation': explanation,
                'base_similarity': candidate.get('similarity', 0.5)
            })

        # Sort by score
        scored_movies.sort(key=lambda x: x['score'], reverse=True)

        # Apply diversity filter
        diverse_results = self._apply_diversity(scored_movies, limit)

        # Format results
        results = []
        for item in diverse_results:
            movie = item['movie']
            results.append({
                'movie_id': movie.movie_id,
                'title': movie.title,
                'release_year': movie.release_year,
                'genres': movie.get_genres_list(),
                'overview': movie.overview,
                'poster_url': movie.get_poster_url(),
                'vote_average': movie.vote_average,
                'director': movie.director,
                'cast': movie.get_cast_list()[:3],
                'score': round(item['score'], 3),
                'explanation': item['explanation']
            })

        return results

    def _get_content_based_candidates(self, favorite_ids, n):
        """Get candidates based on similarity to favorites"""
        return self.similarity_engine.get_similar_to_multiple(
            favorite_ids, n=n, method='hybrid'
        )

    def _get_cold_start_candidates(self, user, n):
        """Get candidates for users with no favorites (cold start)"""
        preferred_genres = user.get_preferred_genres()

        # Query popular movies
        query = Movie.query.filter(Movie.vote_count > 50)

        # If user has preferred genres, prioritize those
        if preferred_genres:
            # Get movies that match any preferred genre
            # (SQLite doesn't have native JSON support, so we use LIKE)
            genre_filters = []
            for genre in preferred_genres:
                genre_filters.append(Movie.genres.like(f'%"{genre}"%'))

            from sqlalchemy import or_
            query = query.filter(or_(*genre_filters))

        # Order by popularity and rating
        movies = query.order_by(
            Movie.vote_average.desc(),
            Movie.popularity.desc()
        ).limit(n).all()

        return [{'movie_id': m.movie_id, 'similarity': 0.5} for m in movies]

    def _calculate_score(self, movie, base_similarity, taste_vector, mood, favorite_ids):
        """
        Calculate final recommendation score

        Returns:
            tuple: (score, explanation_string)
        """
        explanations = []

        # Start with base similarity score (0.4 weight)
        score = 0.4 * base_similarity

        if base_similarity > 0.3:
            explanations.append(f"Similar to movies you like")

        # Taste vector matching (0.3 weight)
        movie_genres = movie.get_genres_list()
        taste_score = self._compute_taste_match(movie_genres, taste_vector)
        score += 0.3 * taste_score

        if taste_score > 0.5 and taste_vector:
            matching_genres = [g for g in movie_genres if taste_vector.get(g, 0) > 0.5]
            if matching_genres:
                explanations.append(f"Matches your taste in {matching_genres[0]}")

        # Quality factor (0.15 weight)
        quality_score = min(movie.vote_average / 10.0, 1.0) if movie.vote_average else 0.5
        score += 0.15 * quality_score

        if movie.vote_average and movie.vote_average >= 7.5:
            explanations.append(f"Highly rated ({movie.vote_average:.1f}/10)")

        # Mood boost (0.15 weight, multiplicative on top)
        mood_multiplier = self._get_mood_boost(mood, movie_genres)
        if mood_multiplier > 1.0:
            score *= mood_multiplier
            mood_genres = self.MOOD_GENRE_MAP.get(mood.lower(), {}) if mood else {}
            boosted = [g for g in movie_genres if g in mood_genres]
            if boosted:
                explanations.append(f"Great for {mood} mood")

        # Combine explanations
        if not explanations:
            explanations.append("Popular choice")

        explanation = " | ".join(explanations[:2])  # Max 2 reasons

        return score, explanation

    def _compute_taste_match(self, movie_genres, taste_vector):
        """Compute how well movie matches user's taste vector"""
        if not taste_vector or not movie_genres:
            return 0.5  # Neutral

        scores = [taste_vector.get(genre, 0.5) for genre in movie_genres]
        return sum(scores) / len(scores) if scores else 0.5

    def _get_mood_boost(self, mood, movie_genres):
        """Get mood-based multiplier for a movie"""
        if not mood or not movie_genres:
            return 1.0

        mood = mood.lower()
        if mood not in self.MOOD_GENRE_MAP:
            return 1.0

        mood_boosts = self.MOOD_GENRE_MAP[mood]
        boosts = [mood_boosts.get(genre, 1.0) for genre in movie_genres]

        # Return max boost (most relevant genre)
        return max(boosts) if boosts else 1.0

    def _apply_diversity(self, scored_movies, n):
        """
        Ensure diversity in recommendations

        Limits movies per primary genre to avoid homogeneous results
        """
        diverse_set = []
        genre_counts = {}
        max_per_genre = max(2, n // 3)  # At least 2, or 1/3 of results

        for item in scored_movies:
            movie = item['movie']
            genres = movie.get_genres_list()
            primary_genre = genres[0] if genres else 'Unknown'

            # Check genre limit
            if genre_counts.get(primary_genre, 0) < max_per_genre:
                diverse_set.append(item)
                genre_counts[primary_genre] = genre_counts.get(primary_genre, 0) + 1

            if len(diverse_set) >= n:
                break

        return diverse_set

    def get_similar_movies(self, movie_id, n=10):
        """
        Get movies similar to a given movie

        Args:
            movie_id: Source movie ID
            n: Number of results

        Returns:
            List of similar movies with scores
        """
        if not self._loaded:
            self.load()

        similar = self.similarity_engine.get_similar_movies(movie_id, n=n, method='hybrid')

        results = []
        for item in similar:
            movie = Movie.query.get(item['movie_id'])
            if movie:
                results.append({
                    'movie_id': movie.movie_id,
                    'title': movie.title,
                    'release_year': movie.release_year,
                    'genres': movie.get_genres_list(),
                    'poster_url': movie.get_poster_url(),
                    'vote_average': movie.vote_average,
                    'similarity': round(item['similarity'], 3)
                })

        return results


# Singleton instance
_recommender = None


def get_recommender():
    """Get or create the recommender singleton"""
    global _recommender
    if _recommender is None:
        _recommender = RecommendationEngine()
        _recommender.load()
    return _recommender

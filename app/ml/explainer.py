"""
Explainer - Generate human-readable explanations for recommendations

Provides clear, understandable reasons for why each movie was recommended.
"""
from app.models import Movie, User, UserFavorite


class RecommendationExplainer:
    """
    Generates explanations for movie recommendations.

    Explanations are based on:
    1. Genre matching with user preferences
    2. Similarity to liked movies
    3. Mood boosting
    4. Popularity/ratings
    """

    # Templates for different explanation types
    TEMPLATES = {
        'genre_match': [
            "Matches your love for {genre}",
            "Perfect for {genre} fans like you",
            "Right up your alley - {genre}",
            "Because you enjoy {genre}",
        ],
        'similar_to': [
            "Similar to {movie} which you liked",
            "Fans of {movie} also loved this",
            "If you liked {movie}, try this",
            "Has the same vibe as {movie}",
        ],
        'mood': [
            "Great for your {mood} mood",
            "Perfect when you're feeling {mood}",
            "Ideal for a {mood} evening",
            "{mood} mood? This is for you",
        ],
        'highly_rated': [
            "Highly rated ({rating}/10)",
            "Critics and audiences love it ({rating}/10)",
            "A crowd favorite ({rating}/10)",
        ],
        'director': [
            "From director {director}",
            "Directed by {director}",
        ],
        'cast': [
            "Starring {actor}",
            "Features {actor}",
        ],
        'popular': [
            "A popular choice",
            "Trending now",
            "Fan favorite",
        ],
        'hidden_gem': [
            "A hidden gem you might love",
            "Underrated but matches your taste",
            "Discover something new",
        ],
    }

    def __init__(self, user_id=None):
        self.user = User.query.get(user_id) if user_id else None
        self.taste_vector = self.user.get_taste_vector() if self.user else {}
        self.favorite_movies = self._get_favorite_movies() if self.user else []

    def _get_favorite_movies(self):
        """Get user's favorite movies."""
        if not self.user:
            return []
        favorites = UserFavorite.query.filter_by(user_id=self.user.user_id).limit(10).all()
        return [Movie.query.get(f.movie_id) for f in favorites if Movie.query.get(f.movie_id)]

    def explain(self, movie, similarity_score=0, mood=None):
        """
        Generate explanation for why a movie was recommended.

        Args:
            movie: Movie object
            similarity_score: Content similarity score (0-1)
            mood: Optional mood string

        Returns:
            str: Human-readable explanation
        """
        reasons = []

        # 1. Check genre match
        genre_reason = self._explain_genre_match(movie)
        if genre_reason:
            reasons.append(genre_reason)

        # 2. Check similarity to favorites
        similar_reason = self._explain_similarity(movie, similarity_score)
        if similar_reason:
            reasons.append(similar_reason)

        # 3. Check mood match
        if mood:
            mood_reason = self._explain_mood(movie, mood)
            if mood_reason:
                reasons.append(mood_reason)

        # 4. Check rating
        if movie.vote_average and movie.vote_average >= 7.5:
            reasons.append(self._format_template('highly_rated', rating=f"{movie.vote_average:.1f}"))

        # 5. Fallback to director/cast if no other reasons
        if not reasons:
            if movie.director:
                reasons.append(self._format_template('director', director=movie.director))
            elif movie.get_cast_list():
                reasons.append(self._format_template('cast', actor=movie.get_cast_list()[0]))
            else:
                reasons.append(self._format_template('popular'))

        # Combine reasons (max 2)
        return " | ".join(reasons[:2])

    def _explain_genre_match(self, movie):
        """Explain based on genre preferences."""
        if not self.taste_vector:
            return None

        movie_genres = movie.get_genres_list()

        # Find the genre with highest user preference
        best_match = None
        best_score = 0

        for genre in movie_genres:
            score = self.taste_vector.get(genre, 0)
            if score > best_score and score > 0.6:
                best_match = genre
                best_score = score

        if best_match:
            return self._format_template('genre_match', genre=best_match)

        return None

    def _explain_similarity(self, movie, similarity_score):
        """Explain based on similarity to favorites."""
        if similarity_score < 0.3 or not self.favorite_movies:
            return None

        # Find which favorite it's most similar to
        # (simplified - just pick a random favorite with matching genre)
        movie_genres = set(movie.get_genres_list())

        for fav in self.favorite_movies:
            fav_genres = set(fav.get_genres_list())
            if movie_genres & fav_genres:  # Intersection
                return self._format_template('similar_to', movie=fav.title)

        return "Similar to movies you like"

    def _explain_mood(self, movie, mood):
        """Explain based on mood match."""
        MOOD_GENRES = {
            'happy': ['Comedy', 'Animation', 'Family'],
            'sad': ['Drama', 'Romance'],
            'excited': ['Action', 'Adventure', 'Thriller'],
            'relaxed': ['Comedy', 'Documentary', 'Family'],
            'romantic': ['Romance', 'Drama'],
            'thrilled': ['Thriller', 'Horror', 'Mystery'],
            'nostalgic': ['Drama', 'Family', 'Animation'],
            'adventurous': ['Adventure', 'Action', 'Fantasy'],
            'thoughtful': ['Documentary', 'Drama', 'Mystery'],
            'scared': ['Horror', 'Thriller'],
        }

        mood_lower = mood.lower()
        if mood_lower not in MOOD_GENRES:
            return None

        movie_genres = movie.get_genres_list()
        mood_genres = MOOD_GENRES[mood_lower]

        if any(g in mood_genres for g in movie_genres):
            mood_display = mood.capitalize()
            return self._format_template('mood', mood=mood_display)

        return None

    def _format_template(self, template_type, **kwargs):
        """Format a random template of the given type."""
        import random
        templates = self.TEMPLATES.get(template_type, [])
        if not templates:
            return ""

        template = random.choice(templates)
        try:
            return template.format(**kwargs)
        except KeyError:
            return template

    def explain_taste_change(self, changes, movie_title):
        """
        Generate explanation for taste vector changes.

        Args:
            changes: Dict of genre changes from TasteUpdater
            movie_title: Title of the movie that caused the change

        Returns:
            str: Human-readable explanation of changes
        """
        if not changes:
            return f"Added '{movie_title}' to favorites."

        # Find the most significant change
        biggest_change = max(changes.items(), key=lambda x: abs(x[1]['change']))
        genre = biggest_change[0]
        change_info = biggest_change[1]

        direction = "increased" if change_info['change'] > 0 else "decreased"
        percentage = abs(int(change_info['change'] * 100))

        explanations = [
            f"Your taste for {genre} {direction} by {percentage}%",
            f"We've noted your interest in {genre}!",
            f"Your {genre} preference has been updated",
        ]

        import random
        return random.choice(explanations)


def generate_explanation(user_id, movie, similarity_score=0, mood=None):
    """
    Convenience function to generate an explanation.

    Args:
        user_id: User ID
        movie: Movie object
        similarity_score: Content similarity score
        mood: Optional mood string

    Returns:
        str: Explanation string
    """
    explainer = RecommendationExplainer(user_id)
    return explainer.explain(movie, similarity_score, mood)


# ---------------------------------------------------------------------------
# v2: explanations computed from the embedding geometry (no random templates)
# ---------------------------------------------------------------------------
def geometric_explanation(store, row, taste, mood, lam, liked_rows, titles,
                          rank_without_mood=None, rank_with_mood=None):
    """
    Explain one recommendation from vector relationships.

        taste_match   = cos(user_taste, movie)
        mood_effect   = cos(search_position, movie) - cos(user_taste, movie)
        nearest_liked = argmax over liked movies of cos(movie, liked)

    titles: dict movie_id -> title (used only to name the nearest liked movie).
    """
    import numpy as np
    from app.ml.embeddings import unit

    m = store.E[row]
    taste_u = unit(taste)
    pos = unit(store.search_position(taste_u, mood, lam))
    taste_match = float(m @ taste_u)
    mood_effect = float(m @ pos) - taste_match

    nearest = None
    if liked_rows:
        sims = store.E[liked_rows] @ m
        j = int(np.argmax(sims))
        nearest = {'movie_id': int(store.ids[liked_rows[j]]),
                   'title': titles.get(int(store.ids[liked_rows[j]])),
                   'similarity': round(float(sims[j]), 3)}

    moved = None
    if rank_without_mood is not None and rank_with_mood is not None:
        moved = int(rank_without_mood - rank_with_mood)   # >0 = moved up because of mood

    parts = []
    if nearest and nearest['title']:
        parts.append(f"Close to {nearest['title']}, which you liked")
    elif not liked_rows:
        parts.append("Near the centre of the catalogue - like a few films to sharpen this")
    if mood and lam > 0:
        if moved and moved > 0:
            parts.append(f"{mood} mood lifted it {moved} place{'s' if moved != 1 else ''}")
        elif mood_effect > 0.005:
            parts.append(f"nudged toward {mood}")
        else:
            parts.append(f"stays put under a {mood} nudge")
    parts.append(f"taste match {taste_match:.2f}")

    return {
        'taste_match': round(taste_match, 3),
        'mood_effect': round(mood_effect, 3),
        'nearest_liked': nearest,
        'rank_without_mood': rank_without_mood,
        'rank_with_mood': rank_with_mood,
        'text': ' · '.join(parts),
    }

"""
MovieMind v2 recommender: mood-shifted nearest-neighbour search in one embedding space.

    search_position = unit(user_taste) + lambda * mood_direction
    ranking         = cosine(search_position, every movie), liked movies removed,
                      then a cluster-based diversity cap of max(2, k // 3) per region.
"""
import numpy as np
from app.models import Movie, User, UserFavorite
from app.ml.embeddings import get_store, unit, EmbeddingStore
from app.ml.explainer import geometric_explanation

DEFAULT_LAMBDA = 0.3


class EmbeddingRecommender:
    engine = 'v2-embedding'

    def __init__(self, store=None):
        self.store = store or get_store()
        if self.store is None:
            raise FileNotFoundError('Embeddings not built - run scripts/build_embeddings.py')

    # -- taste ------------------------------------------------------------
    def _cold_start_taste(self, user):
        """No likes yet: centroid of well-rated movies in the user's preferred genres, else global mean."""
        genres = user.get_preferred_genres()
        if not genres:
            return self.store.global_mean.copy()
        from sqlalchemy import or_
        q = Movie.query.filter(Movie.vote_count > 50,
                               or_(*[Movie.genres.like(f'%"{g}"%') for g in genres]))
        ids = [m.movie_id for m in q.order_by(Movie.vote_average.desc()).limit(40).all()]
        return self.store.centroid(ids)

    def taste_for(self, user, liked_ids):
        stored = user.get_taste_embedding()
        if liked_ids:
            if len(stored) == self.store.E.shape[1]:
                return unit(np.array(stored, dtype='float32'))
            return self.store.centroid(liked_ids)
        return self._cold_start_taste(user)

    # -- main entry -------------------------------------------------------
    def get_recommendations(self, user_id, mood=None, limit=10, lam=DEFAULT_LAMBDA, exclude_favorites=True):
        user = User.query.get(user_id)
        if not user:
            raise ValueError(f'User {user_id} not found')
        if mood and mood.lower() not in self.store.mood_vectors:
            raise ValueError(f"Unknown mood '{mood}'")
        mood = mood.lower() if mood else None
        store = self.store

        liked_ids = [f.movie_id for f in UserFavorite.query.filter_by(user_id=user_id).all()]
        liked_rows = [store.id_to_row[i] for i in liked_ids if i in store.id_to_row]
        taste = self.taste_for(user, liked_ids)

        sims = store.scores(store.search_position(taste, mood, lam))
        sims_plain = store.scores(taste)
        if exclude_favorites:
            sims[liked_rows] = -np.inf
            sims_plain[liked_rows] = -np.inf

        order = np.argsort(-sims)
        rank_with = np.empty(len(order), dtype='int64'); rank_with[order] = np.arange(1, len(order) + 1)
        order0 = np.argsort(-sims_plain)
        rank_plain = np.empty(len(order0), dtype='int64'); rank_plain[order0] = np.arange(1, len(order0) + 1)

        picked = self._diverse(order, sims, limit)
        movies = {m.movie_id: m for m in Movie.query.filter(Movie.movie_id.in_([int(store.ids[r]) for r in picked])).all()}
        liked_titles = {m.movie_id: m.title for m in Movie.query.filter(Movie.movie_id.in_(liked_ids)).all()} if liked_ids else {}

        results = []
        for r in picked:
            movie = movies.get(int(store.ids[r]))
            if movie is None:
                continue
            ex = geometric_explanation(store, r, taste, mood, lam, liked_rows, liked_titles,
                                       int(rank_plain[r]), int(rank_with[r]))
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
                'score': round(float(sims[r]), 3),
                'explanation': ex['text'],
                'explanation_detail': ex,
            })
        return results

    def _diverse(self, order, sims, k):
        cap, counts, picked = max(2, k // 3), {}, []
        for r in order:
            if not np.isfinite(sims[r]):
                break
            c = int(self.store.clusters[r])
            if counts.get(c, 0) < cap:
                picked.append(int(r)); counts[c] = counts.get(c, 0) + 1
            if len(picked) == k:
                break
        return picked

    # -- similar movies (same space) ---------------------------------------
    def get_similar_movies(self, movie_id, n=10):
        if not self.store.has(movie_id):
            return []
        pairs = self.store.similar_to(movie_id, n)
        movies = {m.movie_id: m for m in Movie.query.filter(Movie.movie_id.in_([i for i, _ in pairs])).all()}
        return [{'movie_id': i, 'title': movies[i].title, 'release_year': movies[i].release_year,
                 'genres': movies[i].get_genres_list(), 'poster_url': movies[i].get_poster_url(),
                 'vote_average': movies[i].vote_average, 'similarity': round(s, 3)}
                for i, s in pairs if i in movies]

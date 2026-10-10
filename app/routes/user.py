"""
User routes - Profile management and favorites
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import User, Movie, UserFavorite, UserInteraction, UserLike, UserWatchlist
from app.ml.taste_updater import TasteUpdater, refresh_embedding_taste
from app.ml.explainer import RecommendationExplainer
from datetime import datetime

user_bp = Blueprint('user', __name__)


@user_bp.route('/profile', methods=['GET'])
@login_required
def get_profile():
    """
    Get current user's profile

    Returns:
        User profile with preferences and stats
    """
    # Get favorite count
    favorite_count = UserFavorite.query.filter_by(user_id=current_user.user_id).count()

    # Get interaction count
    interaction_count = UserInteraction.query.filter_by(user_id=current_user.user_id).count()

    profile = current_user.to_dict()
    profile['favorite_count'] = favorite_count
    profile['like_count'] = UserLike.query.filter_by(user_id=current_user.user_id).count()
    profile['watchlist_count'] = UserWatchlist.query.filter_by(user_id=current_user.user_id).count()
    profile['interaction_count'] = interaction_count
    profile['taste_vector'] = current_user.get_taste_vector()

    return jsonify({'profile': profile}), 200


@user_bp.route('/profile', methods=['PUT'])
@login_required
def update_profile():
    """
    Update current user's profile

    Expected JSON (all fields optional):
    {
        "username": "new_username",
        "email": "new_email@example.com",
        "preferred_genres": ["Action", "Comedy"],
        "preferred_industry": "Hollywood",
        "preferred_format": "movies"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    # Update username if provided
    if 'username' in data:
        new_username = data['username']
        if new_username != current_user.username:
            # Check if username is taken
            if User.query.filter_by(username=new_username).first():
                return jsonify({'error': 'Username already taken'}), 409
            current_user.username = new_username

    # Update email if provided
    if 'email' in data:
        new_email = data['email']
        if new_email != current_user.email:
            # Check if email is taken
            if User.query.filter_by(email=new_email).first():
                return jsonify({'error': 'Email already registered'}), 409
            current_user.email = new_email

    # Update preferences
    if 'preferred_genres' in data:
        current_user.set_preferred_genres(data['preferred_genres'])

    if 'preferred_industry' in data:
        current_user.preferred_industry = data['preferred_industry']

    if 'preferred_format' in data:
        current_user.preferred_format = data['preferred_format']

    try:
        current_user.updated_at = datetime.utcnow()
        db.session.commit()

        return jsonify({
            'message': 'Profile updated successfully',
            'profile': current_user.to_dict()
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Update failed: {str(e)}'}), 500


@user_bp.route('/favorites', methods=['GET'])
@login_required
def get_favorites():
    """
    Get current user's favorite movies

    Query params:
        page: Page number (default 1)
        per_page: Results per page (default 20)

    Returns:
        Paginated list of favorite movies
    """
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(max(per_page, 1), 100)

    # Get favorites with movie data
    favorites = UserFavorite.query.filter_by(user_id=current_user.user_id)\
        .order_by(UserFavorite.added_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    movies = []
    for fav in favorites.items:
        movie = Movie.query.get(fav.movie_id)
        if movie:
            movie_data = movie.to_dict()
            movie_data['added_at'] = fav.added_at.isoformat()
            movies.append(movie_data)

    return jsonify({
        'favorites': movies,
        'total': favorites.total,
        'page': page,
        'per_page': per_page,
        'pages': favorites.pages
    }), 200


def _json_movie_id():
    data = request.get_json(silent=True) or {}
    mid = data.get('movie_id')
    return mid if isinstance(mid, int) else None


def _log(movie_id, kind):
    db.session.add(UserInteraction(user_id=current_user.user_id, movie_id=movie_id, interaction_type=kind))


@user_bp.route('/favorites', methods=['POST'])
@login_required
def add_favorite():
    """
    Add a movie to the curated Favorites collection.

    Expected JSON: {"movie_id": 123}

    A favourite is NOT a preference signal; use POST /api/user/likes for that.
    """
    movie_id = _json_movie_id()
    if movie_id is None:
        return jsonify({'error': 'movie_id is required'}), 400
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    if UserFavorite.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first():
        return jsonify({'error': 'Movie already in favorites'}), 409
    try:
        db.session.add(UserFavorite(user_id=current_user.user_id, movie_id=movie_id))
        _log(movie_id, 'favorite')
        db.session.commit()
        return jsonify({'message': 'Movie added to favorites', 'movie': movie.to_dict(),
                        'taste_updated': False}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to add favorite: {str(e)}'}), 500


@user_bp.route('/favorites/<int:movie_id>', methods=['DELETE'])
@login_required
def remove_favorite(movie_id):
    """Remove a movie from Favorites (does not change taste)."""
    favorite = UserFavorite.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first()
    if not favorite:
        return jsonify({'error': 'Movie not in favorites'}), 404
    try:
        db.session.delete(favorite)
        _log(movie_id, 'unfavorite')
        db.session.commit()
        return jsonify({'message': 'Movie removed from favorites', 'taste_updated': False}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to remove favorite: {str(e)}'}), 500


# ---------------------------------------------------------------- likes (preference signal)
@user_bp.route('/likes', methods=['GET'])
@login_required
def get_likes():
    """Liked movies, newest first (paginated)."""
    page = request.args.get('page', 1, type=int)
    per_page = min(max(request.args.get('per_page', 20, type=int), 1), 100)
    q = UserLike.query.filter_by(user_id=current_user.user_id).order_by(UserLike.created_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    movies = []
    for like in q.items:
        m = Movie.query.get(like.movie_id)
        if m:
            d = m.to_dict(); d['added_at'] = like.created_at.isoformat(); movies.append(d)
    return jsonify({'likes': movies, 'total': q.total, 'page': page, 'per_page': per_page, 'pages': q.pages}), 200


@user_bp.route('/likes', methods=['POST'])
@login_required
def add_like():
    """
    Like a movie. This is the signal that teaches the recommender: it updates the
    genre profile and the embedding taste centroid.

    Expected JSON: {"movie_id": 123}
    """
    movie_id = _json_movie_id()
    if movie_id is None:
        return jsonify({'error': 'movie_id is required'}), 400
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    if UserLike.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first():
        return jsonify({'error': 'Movie already liked'}), 409
    try:
        db.session.add(UserLike(user_id=current_user.user_id, movie_id=movie_id))
        _log(movie_id, 'like')
        db.session.commit()

        update_result = TasteUpdater(current_user.user_id).update_on_like(movie)   # genre profile
        refresh_embedding_taste(current_user.user_id)                              # embedding centroid
        taste_message = RecommendationExplainer(current_user.user_id).explain_taste_change(
            update_result.get('changes', {}), movie.title)
        return jsonify({'message': 'Movie liked', 'movie': movie.to_dict(), 'taste_updated': True,
                        'taste_message': taste_message, 'changes': update_result.get('changes', {})}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to like movie: {str(e)}'}), 500


@user_bp.route('/likes/<int:movie_id>', methods=['DELETE'])
@login_required
def remove_like(movie_id):
    """Unlike a movie; the centroid is recomputed from the remaining likes."""
    like = UserLike.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first()
    if not like:
        return jsonify({'error': 'Movie not liked'}), 404
    try:
        movie = Movie.query.get(movie_id)
        db.session.delete(like)
        _log(movie_id, 'unlike')
        db.session.commit()
        if movie:
            TasteUpdater(current_user.user_id).update_on_unlike(movie)
        refresh_embedding_taste(current_user.user_id)
        return jsonify({'message': 'Like removed', 'taste_updated': True}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to remove like: {str(e)}'}), 500


# ---------------------------------------------------------------- watchlist (save for later)
@user_bp.route('/watchlist', methods=['GET'])
@login_required
def get_watchlist():
    """Watchlist, newest first (paginated). Not a preference signal."""
    page = request.args.get('page', 1, type=int)
    per_page = min(max(request.args.get('per_page', 20, type=int), 1), 100)
    q = UserWatchlist.query.filter_by(user_id=current_user.user_id).order_by(UserWatchlist.added_at.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)
    movies = []
    for w in q.items:
        m = Movie.query.get(w.movie_id)
        if m:
            d = m.to_dict(); d['added_at'] = w.added_at.isoformat(); movies.append(d)
    return jsonify({'watchlist': movies, 'total': q.total, 'page': page, 'per_page': per_page, 'pages': q.pages}), 200


@user_bp.route('/watchlist', methods=['POST'])
@login_required
def add_watchlist():
    """Save a movie for later. Expected JSON: {"movie_id": 123}"""
    movie_id = _json_movie_id()
    if movie_id is None:
        return jsonify({'error': 'movie_id is required'}), 400
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404
    if UserWatchlist.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first():
        return jsonify({'error': 'Movie already in watchlist'}), 409
    try:
        db.session.add(UserWatchlist(user_id=current_user.user_id, movie_id=movie_id))
        _log(movie_id, 'watchlist')
        db.session.commit()
        return jsonify({'message': 'Saved to watchlist', 'movie': movie.to_dict(), 'taste_updated': False}), 201
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to save: {str(e)}'}), 500


@user_bp.route('/watchlist/<int:movie_id>', methods=['DELETE'])
@login_required
def remove_watchlist(movie_id):
    item = UserWatchlist.query.filter_by(user_id=current_user.user_id, movie_id=movie_id).first()
    if not item:
        return jsonify({'error': 'Movie not in watchlist'}), 404
    try:
        db.session.delete(item)
        _log(movie_id, 'unwatchlist')
        db.session.commit()
        return jsonify({'message': 'Removed from watchlist'}), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to remove: {str(e)}'}), 500


@user_bp.route('/state', methods=['GET'])
@login_required
def get_state():
    """IDs of every liked / favourited / watchlisted movie, so any page can render consistent button states."""
    uid = current_user.user_id
    ids = lambda model: [r.movie_id for r in model.query.filter_by(user_id=uid).all()]
    return jsonify({'likes': ids(UserLike), 'favorites': ids(UserFavorite), 'watchlist': ids(UserWatchlist)}), 200


@user_bp.route('/favorites/<int:movie_id>/check', methods=['GET'])
@login_required
def check_favorite(movie_id):
    """
    Check if a movie is in user's favorites

    Path params:
        movie_id: The movie ID to check

    Returns:
        Whether the movie is a favorite
    """
    is_favorite = UserFavorite.query.filter_by(
        user_id=current_user.user_id,
        movie_id=movie_id
    ).first() is not None

    return jsonify({'is_favorite': is_favorite}), 200


@user_bp.route('/history', methods=['GET'])
@login_required
def get_history():
    """
    Get user's interaction history

    Query params:
        type: Filter by interaction type (like, unlike, view, rate)
        page: Page number (default 1)
        per_page: Results per page (default 20)

    Returns:
        List of interactions with movie data
    """
    interaction_type = request.args.get('type')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(max(per_page, 1), 100)

    query = UserInteraction.query.filter_by(user_id=current_user.user_id)

    if interaction_type:
        query = query.filter_by(interaction_type=interaction_type)

    interactions = query.order_by(UserInteraction.timestamp.desc())\
        .paginate(page=page, per_page=per_page, error_out=False)

    history = []
    for interaction in interactions.items:
        movie = Movie.query.get(interaction.movie_id)
        if movie:
            history.append({
                'interaction_type': interaction.interaction_type,
                'timestamp': interaction.timestamp.isoformat(),
                'movie': movie.to_dict()
            })

    return jsonify({
        'history': history,
        'total': interactions.total,
        'page': page,
        'per_page': per_page
    }), 200


@user_bp.route('/taste', methods=['GET'])
@login_required
def get_taste_vector():
    """
    Get user's current taste vector

    Returns:
        Taste vector showing genre preferences (0-1 scale)
    """
    taste = current_user.get_taste_vector()

    # Sort by preference strength
    sorted_taste = sorted(taste.items(), key=lambda x: x[1], reverse=True)

    # v2: the films that sit closest to the user's taste centroid in embedding space
    neighbours = []
    emb = current_user.get_taste_embedding()
    from app.ml.embeddings import get_store
    store = get_store()
    liked = {l.movie_id for l in UserLike.query.filter_by(user_id=current_user.user_id).all()}
    if store is not None and emb and len(emb) == store.E.shape[1] and liked:
        import numpy as np
        sims = store.scores(np.array(emb, dtype='float32'))
        for i in np.argsort(-sims):
            mid = int(store.ids[i])
            if mid in liked:
                continue
            m = Movie.query.get(mid)
            if m:
                neighbours.append({'movie_id': mid, 'title': m.title, 'release_year': m.release_year,
                                   'similarity': round(float(sims[i]), 3)})
            if len(neighbours) == 6:
                break

    recent = []
    for l in UserLike.query.filter_by(user_id=current_user.user_id).order_by(UserLike.created_at.desc()).limit(8).all():
        m = Movie.query.get(l.movie_id)
        if m:
            recent.append(m.to_dict())

    # share of liked movies that carry each genre (real counts, so it can be shown as a percentage)
    like_count = UserLike.query.filter_by(user_id=current_user.user_id).count()
    counts = {}
    for l in UserLike.query.filter_by(user_id=current_user.user_id).all():
        m = Movie.query.get(l.movie_id)
        for g in (m.get_genres_list() if m else []):
            counts[g] = counts.get(g, 0) + 1
    genre_share = [{'genre': g, 'count': c, 'share': round(c / like_count, 3)}
                   for g, c in sorted(counts.items(), key=lambda x: -x[1])[:8]] if like_count else []

    # moods whose direction lies closest to the user's taste centroid
    mood_fit = []
    if store is not None and emb and len(emb) == store.E.shape[1] and like_count:
        import numpy as np
        from app.ml.embeddings import unit
        t = unit(np.array(emb, dtype='float32'))
        fit = sorted(((float(t @ d), name) for name, d in store.mood_vectors.items()), reverse=True)
        mood_fit = [{'mood': name, 'score': round(sc, 3)} for sc, name in fit[:3]]

    return jsonify({
        'taste_vector': dict(sorted_taste),
        'top_genres': [g for g, _ in sorted_taste[:5]] if sorted_taste else [],
        'taste_neighbours': neighbours,
        'recent_likes': recent,
        'like_count': like_count,
        'genre_share': genre_share,
        'mood_fit': mood_fit
    }), 200


@user_bp.route('/taste/reset', methods=['POST'])
@login_required
def reset_taste_vector():
    """
    Reset taste vector to initial state based on preferred genres
    """
    preferred = current_user.get_preferred_genres()

    if preferred:
        new_taste = {genre: 0.7 for genre in preferred}
    else:
        new_taste = {}

    current_user.set_taste_vector(new_taste)

    try:
        db.session.commit()
        return jsonify({
            'message': 'Taste vector reset',
            'taste_vector': new_taste
        }), 200
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Reset failed: {str(e)}'}), 500

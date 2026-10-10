"""
Recommendation routes - Get personalized movie recommendations
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import Movie, UserFavorite
from app.ml.recommender import get_recommender
from app.ml.embedding_recommender import DEFAULT_LAMBDA

recommendation_bp = Blueprint('recommendation', __name__)


@recommendation_bp.route('/recommend', methods=['GET'])
@login_required
def get_recommendations():
    """
    Get personalized recommendations for current user

    Query params:
        lambda: Mood strength 0.0-1.0 (default 0.3)
        mood: Optional mood string (happy, sad, excited, relaxed, romantic, thrilled, nostalgic, adventurous, thoughtful, scared)
        limit: Number of recommendations (default 10, max 50)

    Returns:
        List of recommended movies with explanations
    """
    # Get query parameters
    mood = request.args.get('mood')
    limit = request.args.get('limit', 10, type=int)
    limit = min(max(limit, 1), 50)  # Clamp between 1 and 50

    # lambda: how far the mood moves the search away from the user's taste (0 = pure taste, 1 = pure mood)
    lam_raw = request.args.get('lambda', DEFAULT_LAMBDA)
    try:
        lam = float(lam_raw)
    except (TypeError, ValueError):
        return jsonify({'error': 'lambda must be a number between 0 and 1'}), 400
    if not 0.0 <= lam <= 1.0:
        return jsonify({'error': 'lambda must be between 0 and 1'}), 400

    try:
        recommender = get_recommender()
        if getattr(recommender, 'engine', '').startswith('v2'):
            recommendations = recommender.get_recommendations(
                user_id=current_user.user_id, mood=mood, limit=limit, lam=lam
            )
        else:  # v1 fallback (no embeddings built)
            recommendations = recommender.get_recommendations(
                user_id=current_user.user_id, mood=mood, limit=limit
            )

        return jsonify({
            'recommendations': recommendations,
            'count': len(recommendations),
            'mood': mood,
            'lambda': lam,
            'engine': getattr(recommender, 'engine', 'v1-tfidf')
        }), 200

    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        return jsonify({'error': f'Failed to get recommendations: {str(e)}'}), 500


@recommendation_bp.route('/similar/<int:movie_id>', methods=['GET'])
def get_similar_movies(movie_id):
    """
    Get movies similar to a given movie

    Path params:
        movie_id: The movie ID to find similar movies for

    Query params:
        limit: Number of results (default 10, max 20)

    Returns:
        List of similar movies with similarity scores
    """
    # Check if movie exists
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404

    limit = request.args.get('limit', 10, type=int)
    limit = min(max(limit, 1), 20)

    try:
        recommender = get_recommender()
        similar = recommender.get_similar_movies(movie_id, n=limit)

        return jsonify({
            'source_movie': movie.to_dict(),
            'similar_movies': similar,
            'count': len(similar)
        }), 200

    except Exception as e:
        return jsonify({'error': f'Failed to get similar movies: {str(e)}'}), 500


@recommendation_bp.route('/movies', methods=['GET'])
def get_movies():
    """
    Get movies with optional filtering

    Query params:
        genre: Filter by genre
        year: Filter by release year
        search: Search by title
        mood: Order by closeness to a mood direction in the embedding space (overrides sort)
        sort: Sort by 'rating', 'popularity', 'year', 'title' (default: popularity)
        order: 'asc' or 'desc' (default: desc)
        page: Page number (default 1)
        per_page: Results per page (default 20, max 100)

    Returns:
        Paginated list of movies
    """
    # Get query parameters
    genre = request.args.get('genre')
    year = request.args.get('year', type=int)
    search = request.args.get('search')
    sort = request.args.get('sort', 'popularity')
    order = request.args.get('order', 'desc')
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 20, type=int)
    per_page = min(max(per_page, 1), 100)

    # Build query
    query = Movie.query

    # Apply filters
    if genre:
        query = query.filter(Movie.genres.like(f'%"{genre}"%'))

    if year:
        query = query.filter(Movie.release_year == year)

    if search:
        query = query.filter(Movie.title.ilike(f'%{search}%'))

    # Mood ordering: rank the filtered films by cosine similarity to the mood direction
    mood = request.args.get('mood')
    if mood:
        import math
        import numpy as np
        from app.ml.embeddings import get_store
        store = get_store()
        if store is None or mood.lower() not in store.mood_vectors:
            return jsonify({'error': f"Unknown mood '{mood}'"}), 400
        pairs = [(store.id_to_row[i], i) for (i,) in query.with_entities(Movie.movie_id).all() if i in store.id_to_row]
        if pairs:
            scores = store.E[[r for r, _ in pairs]] @ store.mood_vectors[mood.lower()]
            ordered = [pairs[j][1] for j in np.argsort(-scores)]
        else:
            ordered = []
        total = len(ordered)
        window = ordered[(page - 1) * per_page: page * per_page]
        by_id = {m.movie_id: m for m in Movie.query.filter(Movie.movie_id.in_(window)).all()} if window else {}
        pages_n = math.ceil(total / per_page) if total else 0
        return jsonify({
            'movies': [by_id[i].to_dict() for i in window if i in by_id],
            'total': total, 'page': page, 'per_page': per_page, 'pages': pages_n,
            'has_next': page < pages_n, 'has_prev': page > 1, 'mood': mood.lower()
        }), 200

    # Apply sorting
    sort_columns = {
        'rating': Movie.vote_average,
        'popularity': Movie.popularity,
        'year': Movie.release_year,
        'title': Movie.title
    }
    sort_column = sort_columns.get(sort, Movie.popularity)

    if order == 'asc':
        query = query.order_by(sort_column.asc())
    else:
        query = query.order_by(sort_column.desc())

    # Paginate
    pagination = query.paginate(page=page, per_page=per_page, error_out=False)

    movies = [movie.to_dict() for movie in pagination.items]

    return jsonify({
        'movies': movies,
        'total': pagination.total,
        'page': page,
        'per_page': per_page,
        'pages': pagination.pages,
        'has_next': pagination.has_next,
        'has_prev': pagination.has_prev
    }), 200


@recommendation_bp.route('/movies/<int:movie_id>', methods=['GET'])
def get_movie(movie_id):
    """
    Get details for a specific movie

    Path params:
        movie_id: The movie ID

    Returns:
        Movie details
    """
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404

    movie_data = movie.to_dict()

    # Add additional details
    movie_data['keywords'] = movie.get_keywords_list()
    movie_data['full_cast'] = movie.get_cast_list()

    # Check if current user has favorited this movie
    if current_user.is_authenticated:
        is_favorite = UserFavorite.query.filter_by(
            user_id=current_user.user_id,
            movie_id=movie_id
        ).first() is not None
        movie_data['is_favorite'] = is_favorite

    return jsonify({'movie': movie_data}), 200


@recommendation_bp.route('/genres', methods=['GET'])
def get_genres():
    """
    Get list of all available genres

    Returns:
        List of genre names
    """
    # Get unique genres from database
    movies = Movie.query.all()
    all_genres = set()

    for movie in movies:
        genres = movie.get_genres_list()
        all_genres.update(genres)

    # Sort alphabetically
    genres_list = sorted(list(all_genres))

    return jsonify({
        'genres': genres_list,
        'count': len(genres_list)
    }), 200


@recommendation_bp.route('/moods', methods=['GET'])
def get_moods():
    """
    Get list of available moods for recommendations

    Returns:
        List of mood options with descriptions
    """
    moods = [
        {'id': 'happy', 'label': 'Happy', 'description': 'Warm, funny, easy company'},
        {'id': 'sad', 'label': 'Sad', 'description': 'Quiet, aching, human'},
        {'id': 'excited', 'label': 'Excited', 'description': 'Fast, loud, high stakes'},
        {'id': 'relaxed', 'label': 'Relaxed', 'description': 'Slow, gentle, low pressure'},
        {'id': 'romantic', 'label': 'Romantic', 'description': 'Love stories of every shape'},
        {'id': 'thrilled', 'label': 'Thrilled', 'description': 'Tense, clever, edge of the seat'},
        {'id': 'nostalgic', 'label': 'Nostalgic', 'description': 'Growing up, other decades'},
        {'id': 'adventurous', 'label': 'Adventurous', 'description': 'Journeys, treasure, far places'},
        {'id': 'thoughtful', 'label': 'Thoughtful', 'description': 'Ideas that stay with you'},
        {'id': 'scared', 'label': 'Scared', 'description': 'Ghosts, monsters, dread'}
    ]

    return jsonify({'moods': moods}), 200

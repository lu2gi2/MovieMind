"""
User routes - Profile management and favorites
"""
from flask import Blueprint, request, jsonify
from flask_login import login_required, current_user
from app import db
from app.models import User, Movie, UserFavorite, UserInteraction
from app.ml.taste_updater import TasteUpdater
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


@user_bp.route('/favorites', methods=['POST'])
@login_required
def add_favorite():
    """
    Add a movie to favorites (like a movie)

    Expected JSON:
    {
        "movie_id": 123
    }

    This also updates the user's taste vector based on the movie's genres.
    """
    data = request.get_json()

    if not data or 'movie_id' not in data:
        return jsonify({'error': 'movie_id is required'}), 400

    movie_id = data['movie_id']

    # Check if movie exists
    movie = Movie.query.get(movie_id)
    if not movie:
        return jsonify({'error': 'Movie not found'}), 404

    # Check if already favorited
    existing = UserFavorite.query.filter_by(
        user_id=current_user.user_id,
        movie_id=movie_id
    ).first()

    if existing:
        return jsonify({'error': 'Movie already in favorites'}), 409

    try:
        # Add to favorites
        favorite = UserFavorite(
            user_id=current_user.user_id,
            movie_id=movie_id
        )
        db.session.add(favorite)

        # Log interaction
        interaction = UserInteraction(
            user_id=current_user.user_id,
            movie_id=movie_id,
            interaction_type='like'
        )
        db.session.add(interaction)
        db.session.commit()

        # Update taste vector using TasteUpdater (adaptive learning)
        updater = TasteUpdater(current_user.user_id)
        update_result = updater.update_on_like(movie)

        # Generate explanation for taste change
        explainer = RecommendationExplainer(current_user.user_id)
        taste_message = explainer.explain_taste_change(
            update_result.get('changes', {}),
            movie.title
        )

        return jsonify({
            'message': 'Movie added to favorites',
            'movie': movie.to_dict(),
            'taste_updated': True,
            'taste_message': taste_message,
            'changes': update_result.get('changes', {}),
            'new_taste_vector': update_result.get('taste_vector', {})
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to add favorite: {str(e)}'}), 500


@user_bp.route('/favorites/<int:movie_id>', methods=['DELETE'])
@login_required
def remove_favorite(movie_id):
    """
    Remove a movie from favorites

    Path params:
        movie_id: The movie ID to remove
    """
    favorite = UserFavorite.query.filter_by(
        user_id=current_user.user_id,
        movie_id=movie_id
    ).first()

    if not favorite:
        return jsonify({'error': 'Movie not in favorites'}), 404

    try:
        # Get movie before deleting
        movie = Movie.query.get(movie_id)

        db.session.delete(favorite)

        # Log interaction
        interaction = UserInteraction(
            user_id=current_user.user_id,
            movie_id=movie_id,
            interaction_type='unlike'
        )
        db.session.add(interaction)
        db.session.commit()

        # Update taste vector (reduce preference slightly)
        if movie:
            updater = TasteUpdater(current_user.user_id)
            updater.update_on_unlike(movie)

        return jsonify({
            'message': 'Movie removed from favorites',
            'taste_updated': True
        }), 200

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Failed to remove favorite: {str(e)}'}), 500


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

    return jsonify({
        'taste_vector': dict(sorted_taste),
        'top_genres': [g for g, _ in sorted_taste[:5]] if sorted_taste else []
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

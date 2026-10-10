from flask import Flask, jsonify, request, redirect, url_for
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from flask_login import LoginManager
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize extensions
db = SQLAlchemy()
login_manager = LoginManager()


def create_app(config_name=None):
    """Application factory pattern"""
    app = Flask(__name__,
                template_folder='../templates',
                static_folder='../static')

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key')
    app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///movies.db')
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    app.config['SESSION_COOKIE_HTTPONLY'] = True
    app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'

    # Initialize extensions with app
    db.init_app(app)
    CORS(app, supports_credentials=True)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Handle unauthorized requests
    @login_manager.unauthorized_handler
    def unauthorized():
        # Return JSON for API requests, redirect for page requests
        if request.path.startswith('/api/'):
            return jsonify({'error': 'Authentication required'}), 401
        return redirect(url_for('pages.login'))

    # Import models (needed for user_loader)
    from app.models import User, Movie, UserFavorite, UserInteraction

    # Register blueprints
    from app.routes.auth import auth_bp
    from app.routes.recommendation import recommendation_bp
    from app.routes.user import user_bp
    from app.routes.pages import pages_bp

    app.register_blueprint(auth_bp, url_prefix='/api/auth')
    app.register_blueprint(recommendation_bp, url_prefix='/api')
    app.register_blueprint(user_bp, url_prefix='/api/user')
    app.register_blueprint(pages_bp)  # No prefix - serves at root

    # Health check endpoint
    @app.route('/api/health')
    def health_check():
        return jsonify({
            'status': 'healthy',
            'database': 'connected'
        }), 200

    # Create database tables
    with app.app_context():
        db.create_all()
        _ensure_columns()
        migrate_legacy_likes()

    return app


def _ensure_columns():
    """Lightweight migration: add columns introduced after a database was first created."""
    from sqlalchemy import inspect, text
    cols = {c['name'] for c in inspect(db.engine).get_columns('users')}
    if 'taste_embedding' not in cols:
        with db.engine.begin() as conn:
            conn.execute(text('ALTER TABLE users ADD COLUMN taste_embedding TEXT'))


def migrate_legacy_likes():
    """
    Before Likes existed, adding a favourite also counted as a like (it logged a 'like' interaction and
    taught the recommender). Restore exactly those: favourites whose interaction was logged as 'like'
    and that have no UserLike yet. New-style favourites log 'favorite', so they are never promoted.
    Idempotent; returns the number of likes created.
    """
    from app.models import UserFavorite, UserLike, UserInteraction
    created, users = 0, set()
    try:
        for fav in UserFavorite.query.all():
            if UserLike.query.filter_by(user_id=fav.user_id, movie_id=fav.movie_id).first():
                continue
            legacy = UserInteraction.query.filter_by(user_id=fav.user_id, movie_id=fav.movie_id,
                                                     interaction_type='like').first()
            if legacy:
                db.session.add(UserLike(user_id=fav.user_id, movie_id=fav.movie_id, created_at=fav.added_at))
                created += 1; users.add(fav.user_id)
        db.session.commit()
        if users:
            from app.ml.taste_updater import refresh_embedding_taste
            for uid in users:
                refresh_embedding_taste(uid)
    except Exception:
        db.session.rollback()
    return created

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

    return app

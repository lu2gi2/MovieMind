"""Route blueprints"""
from app.routes.auth import auth_bp
from app.routes.recommendation import recommendation_bp
from app.routes.user import user_bp
from app.routes.pages import pages_bp

__all__ = ['auth_bp', 'recommendation_bp', 'user_bp', 'pages_bp']

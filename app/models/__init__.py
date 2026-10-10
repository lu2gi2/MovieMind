"""Database models"""
from app.models.movie import Movie
from app.models.user import User, UserFavorite, UserInteraction, UserLike, UserWatchlist

__all__ = ['Movie', 'User', 'UserFavorite', 'UserInteraction', 'UserLike', 'UserWatchlist']

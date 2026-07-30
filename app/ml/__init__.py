"""Machine Learning modules for movie recommendations"""
from app.ml.vectorizer import MovieVectorizer, create_combined_text
from app.ml.similarity import SimilarityEngine, cosine_sim
from app.ml.recommender import RecommendationEngine, get_recommender
from app.ml.taste_updater import TasteUpdater, update_user_taste
from app.ml.explainer import RecommendationExplainer, generate_explanation

__all__ = [
    'MovieVectorizer',
    'create_combined_text',
    'SimilarityEngine',
    'cosine_sim',
    'RecommendationEngine',
    'get_recommender',
    'TasteUpdater',
    'update_user_taste',
    'RecommendationExplainer',
    'generate_explanation'
]

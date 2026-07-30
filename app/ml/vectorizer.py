"""
Movie Vectorizer - Convert movie data into numerical vectors for ML
"""
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import json
import pickle
import os


class MovieVectorizer:
    """Convert movie data (genres, plots) into numerical vectors"""

    def __init__(self):
        self.genre_encoder = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,      # Limit vocabulary size
            stop_words='english',   # Remove common words
            ngram_range=(1, 2),     # Use single words and bigrams
            min_df=2,               # Word must appear in at least 2 docs
            max_df=0.85             # Ignore words in >85% of docs
        )
        self.all_genres = []
        self.is_fitted = False

    def fit_genres(self, genres_lists):
        """
        Fit the genre encoder on a list of genre lists

        Args:
            genres_lists: List of lists, e.g. [['Action', 'Comedy'], ['Drama']]
        """
        self.genre_encoder.fit(genres_lists)
        self.all_genres = list(self.genre_encoder.classes_)
        return self

    def transform_genres(self, genres_lists):
        """
        Transform genre lists to binary vectors

        Args:
            genres_lists: List of lists of genres

        Returns:
            numpy array of shape (n_movies, n_genres)
        """
        return self.genre_encoder.transform(genres_lists)

    def fit_transform_genres(self, genres_lists):
        """Fit and transform in one step"""
        self.fit_genres(genres_lists)
        return self.transform_genres(genres_lists)

    def fit_plots(self, plot_texts):
        """
        Fit the TF-IDF vectorizer on plot texts

        Args:
            plot_texts: List of plot/overview strings
        """
        # Clean texts - replace NaN/None with empty string
        clean_texts = [str(t) if t and str(t) != 'nan' else '' for t in plot_texts]
        self.tfidf_vectorizer.fit(clean_texts)
        return self

    def transform_plots(self, plot_texts):
        """
        Transform plot texts to TF-IDF vectors

        Args:
            plot_texts: List of plot strings

        Returns:
            Sparse matrix of shape (n_movies, n_features)
        """
        clean_texts = [str(t) if t and str(t) != 'nan' else '' for t in plot_texts]
        return self.tfidf_vectorizer.transform(clean_texts)

    def fit_transform_plots(self, plot_texts):
        """Fit and transform in one step"""
        clean_texts = [str(t) if t and str(t) != 'nan' else '' for t in plot_texts]
        self.is_fitted = True
        return self.tfidf_vectorizer.fit_transform(clean_texts)

    def get_genre_names(self):
        """Get list of all genre names in order"""
        return self.all_genres

    def get_tfidf_feature_names(self):
        """Get TF-IDF vocabulary"""
        return self.tfidf_vectorizer.get_feature_names_out()

    def save(self, filepath):
        """Save vectorizer state to disk"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        state = {
            'genre_encoder': self.genre_encoder,
            'tfidf_vectorizer': self.tfidf_vectorizer,
            'all_genres': self.all_genres,
            'is_fitted': self.is_fitted
        }
        with open(filepath, 'wb') as f:
            pickle.dump(state, f)
        print(f"Saved vectorizer to {filepath}")

    def load(self, filepath):
        """Load vectorizer state from disk"""
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
        self.genre_encoder = state['genre_encoder']
        self.tfidf_vectorizer = state['tfidf_vectorizer']
        self.all_genres = state['all_genres']
        self.is_fitted = state.get('is_fitted', True)
        print(f"Loaded vectorizer from {filepath}")
        return self


def create_combined_text(movie):
    """
    Create a rich text representation of a movie for TF-IDF

    Combines: overview, genres, cast, director, keywords
    """
    parts = []

    # Overview (main content)
    if movie.overview:
        parts.append(movie.overview)

    # Genres (repeated for emphasis)
    genres = movie.get_genres_list()
    if genres:
        parts.append(' '.join(genres))
        parts.append(' '.join(genres))  # Double weight

    # Director
    if movie.director:
        parts.append(movie.director)

    # Cast
    cast = movie.get_cast_list()
    if cast:
        parts.append(' '.join(cast[:3]))  # Top 3 actors

    # Keywords
    keywords = movie.get_keywords_list()
    if keywords:
        parts.append(' '.join(keywords))

    return ' '.join(parts)

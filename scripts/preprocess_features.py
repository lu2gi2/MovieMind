#!/usr/bin/env python3
"""
Preprocess movie features for the recommendation engine

This script:
1. Extracts genres from all movies and creates one-hot encodings
2. Combines plot + genres + cast + keywords into rich text
3. Creates TF-IDF vectors from the combined text
4. Saves all matrices and mappings to disk
"""
import os
import sys
import numpy as np
import scipy.sparse as sp
import pickle

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Movie
from app.ml.vectorizer import MovieVectorizer, create_combined_text


def preprocess_features():
    """Extract and save all movie features"""
    print("=" * 60)
    print("Feature Preprocessing")
    print("=" * 60)

    app = create_app()

    with app.app_context():
        # Load all movies
        movies = Movie.query.all()
        n_movies = len(movies)
        print(f"\nLoaded {n_movies} movies from database")

        if n_movies == 0:
            print("ERROR: No movies in database. Run load_movies.py first.")
            sys.exit(1)

        # Create output directory
        models_dir = 'data/models'
        os.makedirs(models_dir, exist_ok=True)

        # Initialize vectorizer
        vectorizer = MovieVectorizer()

        # ============================================
        # Step 1: Genre encoding
        # ============================================
        print("\n[1/4] Encoding genres...")

        # Extract genre lists from all movies
        genres_lists = [movie.get_genres_list() for movie in movies]

        # Remove empty lists (replace with ['Unknown'])
        genres_lists = [g if g else ['Unknown'] for g in genres_lists]

        # Fit and transform
        genre_matrix = vectorizer.fit_transform_genres(genres_lists)

        print(f"  Genre matrix shape: {genre_matrix.shape}")
        print(f"  Unique genres: {len(vectorizer.all_genres)}")
        print(f"  Genres: {vectorizer.all_genres}")

        # ============================================
        # Step 2: Create combined text for each movie
        # ============================================
        print("\n[2/4] Creating combined text representations...")

        combined_texts = []
        for movie in movies:
            text = create_combined_text(movie)
            combined_texts.append(text)

        avg_length = sum(len(t.split()) for t in combined_texts) / len(combined_texts)
        print(f"  Average text length: {avg_length:.0f} words")

        # ============================================
        # Step 3: TF-IDF vectorization
        # ============================================
        print("\n[3/4] Creating TF-IDF vectors...")

        tfidf_matrix = vectorizer.fit_transform_plots(combined_texts)

        print(f"  TF-IDF matrix shape: {tfidf_matrix.shape}")
        print(f"  Vocabulary size: {len(vectorizer.get_tfidf_feature_names())}")
        print(f"  Non-zero entries: {tfidf_matrix.nnz}")
        print(f"  Sparsity: {100 * (1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1])):.2f}%")

        # ============================================
        # Step 4: Create movie ID mapping
        # ============================================
        print("\n[4/4] Creating ID mappings...")

        movie_id_map = {movie.movie_id: idx for idx, movie in enumerate(movies)}

        print(f"  Mapped {len(movie_id_map)} movie IDs")

        # ============================================
        # Save everything
        # ============================================
        print("\n" + "-" * 40)
        print("Saving to disk...")

        # Save vectorizer
        vectorizer.save(os.path.join(models_dir, 'vectorizer.pkl'))

        # Save TF-IDF matrix (sparse)
        tfidf_path = os.path.join(models_dir, 'tfidf_matrix.npz')
        sp.save_npz(tfidf_path, tfidf_matrix)
        print(f"  Saved TF-IDF matrix to {tfidf_path}")

        # Save genre matrix (dense, small enough)
        genre_path = os.path.join(models_dir, 'genre_matrix.npy')
        np.save(genre_path, genre_matrix)
        print(f"  Saved genre matrix to {genre_path}")

        # Save movie ID mapping
        mapping_path = os.path.join(models_dir, 'movie_id_map.pkl')
        with open(mapping_path, 'wb') as f:
            pickle.dump(movie_id_map, f)
        print(f"  Saved movie ID mapping to {mapping_path}")

        # ============================================
        # Verification
        # ============================================
        print("\n" + "-" * 40)
        print("Verification:")

        # Test similarity computation
        from sklearn.metrics.pairwise import cosine_similarity

        # Pick two movies to test
        test_idx1 = 0
        test_idx2 = 1

        content_sim = cosine_similarity(
            tfidf_matrix[test_idx1],
            tfidf_matrix[test_idx2]
        )[0][0]

        genre_sim = cosine_similarity(
            genre_matrix[test_idx1].reshape(1, -1),
            genre_matrix[test_idx2].reshape(1, -1)
        )[0][0]

        print(f"\n  Test: {movies[test_idx1].title} vs {movies[test_idx2].title}")
        print(f"  Content similarity: {content_sim:.4f}")
        print(f"  Genre similarity: {genre_sim:.4f}")

        # Find most similar to first movie
        print(f"\n  Top 5 similar to '{movies[0].title}':")
        similarities = cosine_similarity(tfidf_matrix[0], tfidf_matrix)[0]
        top_indices = similarities.argsort()[::-1][1:6]  # Skip self

        for idx in top_indices:
            print(f"    - {movies[idx].title} (sim: {similarities[idx]:.3f})")

        # ============================================
        # Summary
        # ============================================
        print("\n" + "=" * 60)
        print("Preprocessing complete!")
        print("=" * 60)
        print(f"\nFiles saved to {models_dir}/:")
        print(f"  - vectorizer.pkl ({os.path.getsize(os.path.join(models_dir, 'vectorizer.pkl')) / 1024:.1f} KB)")
        print(f"  - tfidf_matrix.npz ({os.path.getsize(tfidf_path) / 1024 / 1024:.1f} MB)")
        print(f"  - genre_matrix.npy ({os.path.getsize(genre_path) / 1024:.1f} KB)")
        print(f"  - movie_id_map.pkl ({os.path.getsize(mapping_path) / 1024:.1f} KB)")


if __name__ == '__main__':
    preprocess_features()

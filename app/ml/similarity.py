"""
Similarity Engine - Calculate similarities between movies
"""
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import scipy.sparse as sp
import pickle
import os


class SimilarityEngine:
    """Calculate and cache movie similarities"""

    def __init__(self):
        self.tfidf_matrix = None
        self.genre_matrix = None
        self.movie_id_to_idx = {}
        self.idx_to_movie_id = {}
        self.is_loaded = False

    def load_matrices(self, models_dir='data/models'):
        """Load pre-computed matrices from disk"""
        # Load TF-IDF matrix
        tfidf_path = os.path.join(models_dir, 'tfidf_matrix.npz')
        if os.path.exists(tfidf_path):
            self.tfidf_matrix = sp.load_npz(tfidf_path)
            print(f"Loaded TF-IDF matrix: {self.tfidf_matrix.shape}")

        # Load genre matrix
        genre_path = os.path.join(models_dir, 'genre_matrix.npy')
        if os.path.exists(genre_path):
            self.genre_matrix = np.load(genre_path)
            print(f"Loaded genre matrix: {self.genre_matrix.shape}")

        # Load movie ID mapping
        mapping_path = os.path.join(models_dir, 'movie_id_map.pkl')
        if os.path.exists(mapping_path):
            with open(mapping_path, 'rb') as f:
                self.movie_id_to_idx = pickle.load(f)
            self.idx_to_movie_id = {v: k for k, v in self.movie_id_to_idx.items()}
            print(f"Loaded {len(self.movie_id_to_idx)} movie mappings")

        self.is_loaded = True
        return self

    def compute_content_similarity(self, movie_id_1, movie_id_2):
        """
        Compute content similarity between two movies using TF-IDF

        Returns:
            float: Similarity score between 0 and 1
        """
        if not self.is_loaded or self.tfidf_matrix is None:
            return 0.0

        if movie_id_1 not in self.movie_id_to_idx or movie_id_2 not in self.movie_id_to_idx:
            return 0.0

        idx1 = self.movie_id_to_idx[movie_id_1]
        idx2 = self.movie_id_to_idx[movie_id_2]

        vec1 = self.tfidf_matrix[idx1]
        vec2 = self.tfidf_matrix[idx2]

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)

    def compute_genre_similarity(self, movie_id_1, movie_id_2):
        """
        Compute genre similarity between two movies

        Returns:
            float: Jaccard similarity of genres (0 to 1)
        """
        if not self.is_loaded or self.genre_matrix is None:
            return 0.0

        if movie_id_1 not in self.movie_id_to_idx or movie_id_2 not in self.movie_id_to_idx:
            return 0.0

        idx1 = self.movie_id_to_idx[movie_id_1]
        idx2 = self.movie_id_to_idx[movie_id_2]

        vec1 = self.genre_matrix[idx1]
        vec2 = self.genre_matrix[idx2]

        # Jaccard similarity: intersection / union
        intersection = np.sum(np.minimum(vec1, vec2))
        union = np.sum(np.maximum(vec1, vec2))

        if union == 0:
            return 0.0

        return float(intersection / union)

    def compute_hybrid_similarity(self, movie_id_1, movie_id_2, content_weight=0.7, genre_weight=0.3):
        """
        Compute weighted hybrid similarity

        Args:
            movie_id_1, movie_id_2: Movie IDs to compare
            content_weight: Weight for TF-IDF similarity (default 0.7)
            genre_weight: Weight for genre similarity (default 0.3)

        Returns:
            float: Combined similarity score (0 to 1)
        """
        content_sim = self.compute_content_similarity(movie_id_1, movie_id_2)
        genre_sim = self.compute_genre_similarity(movie_id_1, movie_id_2)

        return content_weight * content_sim + genre_weight * genre_sim

    def get_similar_movies(self, movie_id, n=10, method='hybrid'):
        """
        Find top N most similar movies

        Args:
            movie_id: Source movie ID
            n: Number of similar movies to return
            method: 'content', 'genre', or 'hybrid'

        Returns:
            List of dicts with movie_id and similarity score
        """
        if not self.is_loaded:
            return []

        if movie_id not in self.movie_id_to_idx:
            return []

        idx = self.movie_id_to_idx[movie_id]

        # Calculate similarities based on method
        if method == 'content' and self.tfidf_matrix is not None:
            movie_vec = self.tfidf_matrix[idx]
            similarities = cosine_similarity(movie_vec, self.tfidf_matrix)[0]

        elif method == 'genre' and self.genre_matrix is not None:
            movie_vec = self.genre_matrix[idx].reshape(1, -1)
            # Use cosine similarity for genre vectors too
            similarities = cosine_similarity(movie_vec, self.genre_matrix)[0]

        elif method == 'hybrid':
            # Compute both and combine
            content_sims = np.zeros(len(self.movie_id_to_idx))
            genre_sims = np.zeros(len(self.movie_id_to_idx))

            if self.tfidf_matrix is not None:
                movie_vec = self.tfidf_matrix[idx]
                content_sims = cosine_similarity(movie_vec, self.tfidf_matrix)[0]

            if self.genre_matrix is not None:
                movie_vec = self.genre_matrix[idx].reshape(1, -1)
                genre_sims = cosine_similarity(movie_vec, self.genre_matrix)[0]

            similarities = 0.7 * content_sims + 0.3 * genre_sims

        else:
            return []

        # Get top N indices (excluding self)
        similar_indices = np.argsort(similarities)[::-1]

        results = []
        for sim_idx in similar_indices:
            if sim_idx == idx:  # Skip self
                continue

            results.append({
                'movie_id': self.idx_to_movie_id[sim_idx],
                'similarity': float(similarities[sim_idx])
            })

            if len(results) >= n:
                break

        return results

    def get_similar_to_multiple(self, movie_ids, n=10, method='hybrid'):
        """
        Find movies similar to a set of movies (e.g., user's favorites)

        Args:
            movie_ids: List of movie IDs
            n: Number of results
            method: Similarity method

        Returns:
            List of movie_ids with aggregated similarity scores
        """
        if not movie_ids:
            return []

        # Aggregate similarities across all input movies
        all_scores = {}

        for movie_id in movie_ids:
            similar = self.get_similar_movies(movie_id, n=50, method=method)
            for item in similar:
                mid = item['movie_id']
                if mid not in movie_ids:  # Don't recommend movies already in input
                    if mid not in all_scores:
                        all_scores[mid] = []
                    all_scores[mid].append(item['similarity'])

        # Average the scores
        averaged = []
        for mid, scores in all_scores.items():
            averaged.append({
                'movie_id': mid,
                'similarity': sum(scores) / len(scores),
                'match_count': len(scores)
            })

        # Sort by similarity
        averaged.sort(key=lambda x: x['similarity'], reverse=True)

        return averaged[:n]


def cosine_sim(vec1, vec2):
    """
    Compute cosine similarity between two vectors

    Works with both dense and sparse vectors
    """
    vec1 = np.asarray(vec1).flatten()
    vec2 = np.asarray(vec2).flatten()

    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return float(dot_product / (norm1 * norm2))

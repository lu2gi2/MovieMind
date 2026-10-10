#!/usr/bin/env python3
"""
Offline pipeline for MovieMind v2 (run once, or when the catalogue changes):

    movies (SQLite) -> rich text -> all-MiniLM-L6-v2 -> movie_embeddings.npy
                                 -> mood_vectors.npy (10 x 384)
                                 -> clusters.npy     (diversity regions)
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app
from app.models import Movie
from app.ml import embeddings as E


def main():
    app = create_app()
    with app.app_context():
        movies = Movie.query.order_by(Movie.movie_id).all()
        if not movies:
            sys.exit('No movies in the database. Run scripts/load_movies.py first.')
        print(f'Encoding {len(movies)} movies with {E.MODEL_NAME} ...')
        texts = [E.movie_text(m) for m in movies]
        keyword_lists = [m.get_keywords_list() for m in movies]
        ids = [m.movie_id for m in movies]

    t0 = time.time()
    emb = E.encode_texts(texts)
    enc_s = time.time() - t0
    print(f'  embeddings {emb.shape} {emb.dtype} in {enc_s:.1f}s  ({emb.nbytes / 1e6:.2f} MB)')

    mood_matrix, counts = E.compute_mood_vectors(emb, keyword_lists)
    print('  mood exemplar counts:', json.dumps(counts))
    clusters = E.compute_clusters(emb)
    print(f'  {len(set(clusters.tolist()))} clusters; sizes min/max = '
          f'{min((clusters == c).sum() for c in set(clusters.tolist()))}/'
          f'{max((clusters == c).sum() for c in set(clusters.tolist()))}')

    meta = {'model': E.MODEL_NAME, 'n_movies': len(ids), 'dim': int(emb.shape[1]),
            'moods': E.MOODS, 'exemplar_counts': counts, 'n_clusters': E.N_CLUSTERS,
            'encode_seconds': round(enc_s, 2), 'embedding_mb': round(emb.nbytes / 1e6, 3),
            'mood_vectors_kb': round(mood_matrix.nbytes / 1e3, 2)}
    E.save_artifacts(emb, ids, mood_matrix, clusters, meta)
    print('Saved to', E.MODELS_DIR)


if __name__ == '__main__':
    main()

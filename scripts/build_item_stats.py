#!/usr/bin/env python3
"""
Build the collaborative / quality / popularity statistics used by the hybrid recommender
from the MovieLens ratings (all users), writing data/models/item_stats.npz.

    python scripts/build_item_stats.py

hybrid_config.json (blend weights) is committed; scripts/evaluate_hybrid.py regenerates both after an evaluation.
"""
import os
import sys
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from app.ml import hybrid as H
from app.ml.embeddings import MODELS_DIR

ids = np.load(os.path.join(MODELS_DIR, 'movie_ids.npy'))
row_of = {int(m): i for i, m in enumerate(ids)}
r = pd.read_csv(os.path.join(ROOT, 'data', 'raw', 'ratings.csv'))
r = r[r.movieId.isin(row_of)].copy()
r['row'] = r.movieId.map(row_of)
stats = H.build_item_stats(r.row.values, r.rating.values, r.userId.values, len(ids))
H.save_stats(stats, MODELS_DIR)
print(f'{len(r)} ratings, {r.userId.nunique()} users -> {os.path.join(MODELS_DIR, H.STATS_FILE)}')

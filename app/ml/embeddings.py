"""
Semantic embedding space for MovieMind v2.

Movies, users and moods all live in the same 384-d space produced by the
all-MiniLM-L6-v2 Sentence Transformer:

    movie  -> unit embedding of its rich text
    user   -> unit centroid of liked-movie embeddings
    mood   -> unit direction  mean(mood exemplars) - mean(all movies)

    search_position = unit(user_taste) + lambda * mood_direction
"""
import json
import os
import numpy as np

MODEL_NAME = 'all-MiniLM-L6-v2'
MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                          'data', 'models')
N_CLUSTERS = 20
MIN_EXEMPLARS = 30

# Exemplar rules: TMDB keywords (exact, lower-case). Keywords only - no genres - so a
# mood direction describes *feel* and is not just a re-labelled genre direction.
MOOD_KEYWORDS = {
    'happy': ['friendship', 'family', 'road trip', 'holiday', 'dog', 'musical', 'summer', 'buddy',
              'best friend', 'christmas', 'party', 'wedding', 'cartoon', 'talking animal', 'fun'],
    'sad': ['loss of loved one', 'dying and death', 'grief', 'terminal illness', 'tragedy', 'cancer',
            'funeral', 'loneliness', 'suicide', 'widow', 'depression', 'orphan', 'death of a child', 'mourning'],
    'excited': ['chase', 'heist', 'car race', 'explosion', 'fight', 'martial arts', 'shootout', 'spy',
                'superhero', 'battle', 'survival', 'assassin', 'hitman', 'showdown', 'rescue', 'helicopter'],
    'relaxed': ['small town', 'vacation', 'island', 'countryside', 'nature', 'cooking', 'friendship',
                'road trip', 'garden', 'beach', 'farm', 'fishing', 'travel'],
    'romantic': ['love', 'romance', 'wedding', 'marriage', 'love triangle', 'falling in love', 'first love',
                 "love of one's life", 'romcom', 'dating', 'couple', 'unrequited love', 'long-distance relationship'],
    'thrilled': ['conspiracy', 'kidnapping', 'investigation', 'detective', 'hostage', 'blackmail', 'paranoia',
                 'serial killer', 'psychopath', 'undercover', 'fbi', 'murder', 'twist', 'suspense', 'cat and mouse'],
    'nostalgic': ['coming of age', 'high school', '1950s', '1960s', '1970s', '1980s', 'nostalgia', 'childhood',
                  'school', 'summer camp', 'first love', 'teenager', 'college', 'small town'],
    'adventurous': ['treasure', 'jungle', 'pirate', 'expedition', 'island', 'dragon', 'space', 'journey', 'quest',
                    'treasure hunt', 'alien', 'magic', 'time travel', 'exploration', 'survival', 'sword fight'],
    'thoughtful': ['philosophy', 'identity', 'psychology', 'dystopia', 'time travel', 'obsession', 'biography',
                   'memory', 'dreams', 'artificial intelligence', 'existentialism', 'moral dilemma', 'ethics',
                   'based on true story', 'mental illness'],
    'scared': ['ghost', 'slasher', 'possession', 'demon', 'monster', 'zombie', 'supernatural', 'vampire',
               'paranormal', 'gore', 'creature', 'curse', 'haunted house', 'witch', 'serial killer', 'halloween',
               'psychopath'],
}
MOODS = list(MOOD_KEYWORDS)


# ------------------------------------------------------------------ offline helpers
def movie_text(movie):
    """Rich text for one movie: plot + tagline + genres (x2) + keywords + director + top cast."""
    genres = ' '.join(movie.get_genres_list())
    parts = [movie.overview, movie.tagline, genres, genres,
             ' '.join(movie.get_keywords_list()), movie.director,
             ' '.join(movie.get_cast_list()[:5])]
    return ' '.join(p for p in parts if p)


def encode_texts(texts, model_name=MODEL_NAME, batch_size=64, show_progress=True):
    """Encode texts to L2-normalised float32 embeddings (cosine == dot product)."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device='cpu')
    emb = model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                       show_progress_bar=show_progress, convert_to_numpy=True)
    return emb.astype('float32')


def mood_exemplar_rows(keyword_lists, mood):
    """Row indices of movies whose TMDB keywords hit this mood's rule set."""
    rules = set(MOOD_KEYWORDS[mood])
    return [i for i, kws in enumerate(keyword_lists) if rules.intersection(k.lower() for k in kws)]


def compute_mood_vectors(emb, keyword_lists, min_exemplars=MIN_EXEMPLARS):
    """mood_direction = unit( mean(exemplars) - mean(all) ). Returns (matrix, counts)."""
    global_mean = emb.mean(axis=0)
    vecs, counts = [], {}
    for mood in MOODS:
        rows = mood_exemplar_rows(keyword_lists, mood)
        counts[mood] = len(rows)
        if len(rows) < min_exemplars:
            raise ValueError(f"mood '{mood}' has only {len(rows)} exemplars (< {min_exemplars}); widen its keywords")
        d = emb[rows].mean(axis=0) - global_mean
        vecs.append(d / np.linalg.norm(d))
    return np.vstack(vecs).astype('float32'), counts


def compute_clusters(emb, n_clusters=N_CLUSTERS, seed=0):
    """K-means 'vibe regions' used by the diversity filter."""
    from sklearn.cluster import KMeans
    return KMeans(n_clusters=n_clusters, n_init=10, random_state=seed).fit_predict(emb).astype('int32')


def save_artifacts(emb, movie_ids, mood_matrix, clusters, meta, models_dir=MODELS_DIR):
    os.makedirs(models_dir, exist_ok=True)
    np.save(os.path.join(models_dir, 'movie_embeddings.npy'), emb)
    np.save(os.path.join(models_dir, 'movie_ids.npy'), np.asarray(movie_ids, dtype='int64'))
    np.save(os.path.join(models_dir, 'mood_vectors.npy'), mood_matrix)
    np.save(os.path.join(models_dir, 'clusters.npy'), clusters)
    with open(os.path.join(models_dir, 'embeddings_meta.json'), 'w') as f:
        json.dump(meta, f, indent=2)


# ------------------------------------------------------------------ online store
def unit(v):
    v = np.asarray(v, dtype='float32')
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class EmbeddingStore:
    """In-memory view of the stored matrices; all query-time maths is NumPy."""

    def __init__(self, embeddings, movie_ids, mood_matrix, clusters, moods=None):
        self.E = np.asarray(embeddings, dtype='float32')
        self.ids = np.asarray(movie_ids, dtype='int64')
        self.id_to_row = {int(m): i for i, m in enumerate(self.ids)}
        self.moods = list(moods or MOODS)
        self.mood_vectors = {m: np.asarray(mood_matrix[i], dtype='float32') for i, m in enumerate(self.moods)}
        self.clusters = np.asarray(clusters)
        self.global_mean = unit(self.E.mean(axis=0))

    @classmethod
    def load(cls, models_dir=MODELS_DIR):
        p = lambda n: os.path.join(models_dir, n)
        if not os.path.exists(p('movie_embeddings.npy')):
            raise FileNotFoundError('movie_embeddings.npy not found - run scripts/build_embeddings.py')
        return cls(np.load(p('movie_embeddings.npy')), np.load(p('movie_ids.npy')),
                   np.load(p('mood_vectors.npy')), np.load(p('clusters.npy')))

    @staticmethod
    def available(models_dir=MODELS_DIR):
        return all(os.path.exists(os.path.join(models_dir, n))
                   for n in ('movie_embeddings.npy', 'movie_ids.npy', 'mood_vectors.npy', 'clusters.npy'))

    def has(self, movie_id):
        return int(movie_id) in self.id_to_row

    def embedding(self, movie_id):
        return self.E[self.id_to_row[int(movie_id)]]

    def centroid(self, movie_ids):
        """Unit centroid of the given movies' embeddings; global mean if none are known."""
        rows = [self.id_to_row[int(m)] for m in movie_ids if int(m) in self.id_to_row]
        return unit(self.E[rows].mean(axis=0)) if rows else self.global_mean.copy()

    def mood_direction(self, mood):
        return self.mood_vectors.get(mood.lower()) if mood else None

    def search_position(self, taste, mood=None, lam=0.3):
        """unit(taste) + lam * mood_direction  (taste alone if no mood or lam == 0)."""
        pos = unit(taste)
        d = self.mood_direction(mood)
        if d is not None and lam > 0:
            pos = pos + float(lam) * d
        return pos

    def scores(self, position):
        """Cosine similarity to every movie (rows are unit length)."""
        return self.E @ unit(position)

    def similar_to(self, movie_id, n=10):
        row = self.id_to_row[int(movie_id)]
        s = self.E @ self.E[row]
        s[row] = -np.inf
        top = np.argsort(-s)[:n]
        return [(int(self.ids[i]), float(s[i])) for i in top]


_store = None


def get_store():
    """Lazy singleton; None when the embeddings have not been built yet."""
    global _store
    if _store is None and EmbeddingStore.available():
        _store = EmbeddingStore.load()
    return _store

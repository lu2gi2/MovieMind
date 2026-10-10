# MovieMind — Mood-Aware, Explainable Movie Recommendations

MovieMind recommends movies from what you like and how you feel, and tells you **why** each film was picked.
It is a Flask web app with a hybrid recommender: content embeddings, a collaborative signal and a quality prior
produce a ranked shortlist, and your chosen mood (with a strength slider) **re-ranks** it. It recommends; it does not stream.

> Built for the *24ME511 Product Development Lab III* project at R.M.K. Engineering College.
> The full report generator is in `docs/report/` (see *Project report* below).

## What it does

- **Home:** a featured pick, recommendations with plain-language reasons ("Liked by people who also liked Heat"),
  a mood browser with a strength slider, popular films, and a right-hand rail with your **My Taste** summary and **Watchlist**.
- **Three distinct actions:** **Like** teaches the recommender; **Favorite** is a curated collection; **Watchlist** saves for later.
  Only likes change your taste.
- **Mood control:** ten moods (happy, sad, excited, relaxed, romantic, thrilled, nostalgic, adventurous, thoughtful, scared).
  The mood only reorders a shortlist the engine already trusts, so it changes the list without lowering quality.
- **Explainable:** every pick states its main reason (collaborative, content, quality or popularity) and how far the mood moved it.
- **Movies, Favorites, Watchlist, My Taste, Details, Profile** pages, all on one consistent state.

## How the recommender works (v2.1)

```
base(film | you) = w_content * z(similarity of film to your taste)         what the film is like
                 + w_collab  * z(liked-together score, popularity-discounted)
                 + w_quality * z(smoothed average rating)
                 + w_pop     * z(log(1 + likes))

1. take the top 60 unseen films by base score
2. re-rank them:  final = (1 - lambda) * base + lambda * mood_fit      (lambda = 0 leaves the order unchanged)
3. apply a diversity cap per "vibe region" and return the top k with explanations
```

- **Embeddings:** each film is encoded by `all-MiniLM-L6-v2` (384-d) from its plot, tagline, genres, keywords, director and cast.
  Your taste is the centroid of the films you like; each mood is a direction learned from the data
  (`mean(mood exemplars) - mean(all films)`).
- **Collaborative signal:** item-item "liked together" similarity from the MovieLens ratings, discounted by popularity so everyone
  is not shown the same few titles.
- **Weights** are tuned on held-out validation users, subject to a minimum-variety requirement; see `data/models/hybrid_config.json`.

### Measured results (offline, MovieLens `ml-latest-small`, users held out, mean over 5 random splits)

| System | Hit rate @10 |
|---|---|
| v1 TF-IDF + genres (the original engine) | 0.15 |
| Content embeddings only | 0.14 |
| Most-popular films | 0.35 |
| **Hybrid (final)** | **0.44** |
| Hybrid + mood re-rank (λ = 0.3) | 0.47 |

The hybrid is ahead of the popularity reference in all five splits, but the margin on a single split is not statistically
conclusive, and nothing here measures user satisfaction (no user study was run). Reach into the long tail is limited:
a simulation of 600 users recommended about 8% of the catalogue, because half the films have only 1 to 4 ratings.
Full tables, the accuracy-versus-variety frontier and limitations are in the report (Chapter 7).

## Tech stack

Python 3.10+, Flask 3, SQLAlchemy 2 / SQLite, NumPy, SciPy, scikit-learn, sentence-transformers (CPU PyTorch),
plain HTML/CSS/JavaScript (a custom "Sky Light" design system, no CSS framework). Film data from TMDB; ratings from MovieLens.

## Quick start

```bash
git clone <repository-url> && cd PDL
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt            # add torch from https://download.pytorch.org/whl/cpu if you want a small CPU build

cp .env.example .env                       # set SECRET_KEY (TMDB_API_KEY only needed to refetch metadata)
python scripts/setup_database.py
python scripts/load_movies.py              # loads data/raw/movies_enriched.csv (4,988 films)

python scripts/build_embeddings.py         # embeddings, mood vectors, clusters   (~2 min on CPU, one-time)
python scripts/build_item_stats.py         # quality / popularity / item-item similarity from MovieLens ratings

python run.py                              # http://localhost:5000
```

Set `RECOMMENDER=v1` to run the original TF-IDF + genre engine instead (kept as the evaluation baseline).

## Project structure

```
PDL/
├── app/
│   ├── __init__.py            # app factory, start-up migrations
│   ├── models/                # Movie, User, UserLike, UserFavorite, UserWatchlist, UserInteraction
│   ├── routes/                # auth, recommendation (+ /movies, /similar, /moods), user (likes, favorites, watchlist, taste), pages
│   └── ml/
│       ├── embeddings.py          # embedding space, mood rules, EmbeddingStore
│       ├── hybrid.py              # item statistics, blending, mood re-ranking
│       ├── embedding_recommender.py  # the live recommender (hybrid, with a content-only fallback)
│       ├── explainer.py           # explanations from the numbers
│       ├── taste_updater.py       # taste profile / centroid updates
│       └── vectorizer.py, similarity.py, recommender.py   # v1 TF-IDF engine (baseline)
├── scripts/
│   ├── build_embeddings.py        # offline: embeddings, mood vectors, clusters
│   ├── build_item_stats.py        # offline: collaborative and prior statistics
│   ├── evaluate.py                # initial (v2.0) evaluation
│   ├── evaluate_hybrid.py         # held-out-user evaluation of the hybrid; writes data/models/hybrid_config.json
│   ├── audit_catalogue.py         # catalogue connectivity and reachability audit
│   └── gen_botanical.py           # generates the background artwork (SVG)
├── templates/, static/            # Sky Light interface
├── tests/                         # 99 tests (pytest)
├── data/{raw,models}/             # MovieLens + TMDB data; embeddings and statistics
└── docs/report/                   # PDL III report generator, diagrams, measured results
```

## API

| Endpoint | Method | Description |
|---|---|---|
| `/api/auth/signup`, `/login`, `/logout`, `/check` | POST / GET | Authentication |
| `/api/recommend?mood=&lambda=&limit=` | GET | Recommendations with explanations (`lambda` 0–1, default 0.3; 400 on bad values) |
| `/api/similar/<id>` | GET | Films nearest to a film |
| `/api/movies?search=&genre=&year=&mood=&sort=&page=` | GET | Catalogue (`mood` orders by mood fit) |
| `/api/movies/<id>`, `/api/genres`, `/api/moods` | GET | Details and lists |
| `/api/user/likes` | GET / POST / DELETE | Likes: update your taste |
| `/api/user/favorites` | GET / POST / DELETE | Favorites collection (no effect on taste) |
| `/api/user/watchlist` | GET / POST / DELETE | Saved for later (no effect on taste) |
| `/api/user/state` | GET | Ids of liked / favorited / watchlisted titles |
| `/api/user/taste` | GET | Genre shares, closest films to your taste, moods that suit it, recent likes |
| `/api/user/profile` | GET / PUT | Profile and starting genres |

## Tests

```bash
pytest tests/ -q        # 99 tests; they run against an in-memory database and never touch your data
```

## Evaluation and audit

```bash
python scripts/evaluate_hybrid.py    # held-out evaluation, charts, and production blend weights (~2.5 min)
python scripts/audit_catalogue.py    # are all films wired in, and can each be recommended?
```

## Project report

`docs/report/` builds the college report (`.docx`) from the R.M.K. sample format. Numbers in it are read from
`docs/report/results/*.json`, and the code listings are extracted from the source. Rebuild with
`cd docs/report && python build.py`.

## Limitations

Small dataset (610 MovieLens users); offline accuracy is not user satisfaction; no mood labels exist offline, so the benefit of a
user-chosen mood is untested; weight selection is noisy; reach into the long tail is limited by sparse ratings; English-centred
text encoder. See the report for details.

## Data and credits

Film metadata: [TMDB](https://www.themoviedb.org/) (this product uses the TMDB API but is not endorsed or certified by TMDB).
Ratings: [MovieLens](https://grouplens.org/datasets/movielens/) (GroupLens). Embeddings: `sentence-transformers/all-MiniLM-L6-v2`.

## License

For educational purposes.

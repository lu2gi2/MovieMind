# Implementation Plan
# AI-Based Movie Recommendation System

**Start Date**: February 25, 2026
**Target Completion**: 6-8 weeks
**Current Status**: 🟡 Planning Phase

---

## 📋 Quick Navigation

- [Overview](#overview)
- [Phase 1: Foundation & Data Setup (Week 1-2)](#phase-1-foundation--data-setup-week-1-2)
- [Phase 2: Core ML Engine (Week 2-3)](#phase-2-core-ml-engine-week-2-3)
- [Phase 3: Backend API (Week 3-4)](#phase-3-backend-api-week-3-4)
- [Phase 4: Frontend & Integration (Week 4-5)](#phase-4-frontend--integration-week-4-5)
- [Phase 5: Adaptive Learning (Week 5-6)](#phase-5-adaptive-learning-week-5-6)
- [Phase 6: Testing & Polish (Week 6-8)](#phase-6-testing--polish-week-6-8)
- [Daily Checklist Template](#daily-checklist-template)
- [Quick Commands Reference](#quick-commands-reference)

---

## Overview

### Project Structure
```
PDL/
├── app/                    # Flask application
├── data/                   # Datasets and models
├── notebooks/              # Jupyter experiments
├── scripts/                # Setup scripts
├── tests/                  # Unit tests
├── templates/              # HTML files
├── static/                 # CSS, JS, images
├── requirements.txt
├── run.py
├── PRD.md                  # Full requirements
└── PLAN.md                 # This file
```

### Development Approach
1. **Build incrementally** - Each phase produces working code
2. **Test continuously** - Verify each feature before moving on
3. **Document as you go** - Comment code, update README
4. **Commit frequently** - Git commit after each major task

### Time Allocation
- **Week 1-2**: Data setup (20%)
- **Week 2-3**: ML engine (25%)
- **Week 3-4**: Backend API (20%)
- **Week 4-5**: Frontend (20%)
- **Week 5-6**: Adaptive learning (10%)
- **Week 6-8**: Testing & polish (5%)

---

## Phase 1: Foundation & Data Setup (Week 1-2)

**Goal**: Set up environment and populate database with 5,000+ movies

### Week 1: Environment & Data Collection

#### Day 1-2: Environment Setup

**Tasks**:
- [ ] Create project directory structure
- [ ] Set up Git repository
- [ ] Create virtual environment
- [ ] Install dependencies
- [ ] Create `.env` file
- [ ] Initialize Git with proper `.gitignore`

**Commands**:
```bash
# Create project structure
mkdir -p PDL/{app/{routes,models,ml,utils},data/{raw,processed,models},notebooks,scripts,tests,templates,static/{css,js,images}}
cd PDL

# Initialize Git
git init
echo "Creating .gitignore..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install flask flask-sqlalchemy flask-cors flask-login
pip install pandas numpy scikit-learn
pip install python-dotenv requests
pip install jupyter pytest

# Save dependencies
pip freeze > requirements.txt

# Create .env file
touch .env
```

**Files to Create**:
1. `.gitignore`
2. `.env`
3. `requirements.txt`
4. `README.md` (basic)
5. `app/__init__.py` (empty for now)

**Verification**:
```bash
# Test Python environment
python --version  # Should be 3.10+
pip list          # Verify all packages installed

# Test imports
python -c "import flask, pandas, sklearn; print('All imports successful')"
```

**Commit**: `git commit -m "Initial project setup with dependencies"`

---

#### Day 3-5: Data Collection

**Tasks**:
- [ ] Sign up for TMDB API key
- [ ] Download MovieLens dataset
- [ ] Create data fetching script
- [ ] Fetch movie details from TMDB
- [ ] Save raw data to CSV

**Step-by-Step**:

**1. Get TMDB API Key**:
- Go to https://www.themoviedb.org/settings/api
- Register for free account
- Request API key (select "Developer")
- Add to `.env` file:
  ```
  TMDB_API_KEY=your_key_here
  ```

**2. Download MovieLens**:
```bash
cd data/raw
wget https://files.grouplens.org/datasets/movielens/ml-latest-small.zip
unzip ml-latest-small.zip
mv ml-latest-small/* .
rm -rf ml-latest-small ml-latest-small.zip

# You should now have:
# - movies.csv
# - ratings.csv
# - links.csv
```

**3. Create Data Fetching Script**:

Create `scripts/fetch_tmdb_data.py`:
```python
import requests
import pandas as pd
import time
import os
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.getenv('TMDB_API_KEY')
BASE_URL = "https://api.themoviedb.org/3"

def fetch_movie_details(tmdb_id):
    """Fetch movie details from TMDB API"""
    url = f"{BASE_URL}/movie/{tmdb_id}"
    params = {
        'api_key': TMDB_API_KEY,
        'append_to_response': 'credits,keywords'
    }

    try:
        response = requests.get(url, params=params, timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except Exception as e:
        print(f"Error fetching movie {tmdb_id}: {e}")
        return None

def main():
    # Load MovieLens data
    movies_df = pd.read_csv('data/raw/movies.csv')
    links_df = pd.read_csv('data/raw/links.csv')

    # Merge to get TMDB IDs
    merged = movies_df.merge(links_df, on='movieId')

    # Limit to first 5000 movies for now
    merged = merged.head(5000)

    enriched_movies = []

    print(f"Fetching details for {len(merged)} movies...")

    for idx, row in merged.iterrows():
        if pd.isna(row['tmdbId']):
            continue

        tmdb_id = int(row['tmdbId'])

        # Fetch from TMDB
        details = fetch_movie_details(tmdb_id)

        if details:
            movie_data = {
                'movieId': row['movieId'],
                'tmdb_id': tmdb_id,
                'imdb_id': row['imdbId'],
                'title': details.get('title'),
                'release_year': details.get('release_date', '')[:4] if details.get('release_date') else None,
                'overview': details.get('overview'),
                'genres': [g['name'] for g in details.get('genres', [])],
                'vote_average': details.get('vote_average'),
                'popularity': details.get('popularity'),
                'poster_path': details.get('poster_path'),
            }

            # Extract cast (top 3)
            credits = details.get('credits', {})
            cast = credits.get('cast', [])[:3]
            movie_data['cast'] = [actor['name'] for actor in cast]

            # Extract director
            crew = credits.get('crew', [])
            directors = [c['name'] for c in crew if c['job'] == 'Director']
            movie_data['director'] = directors[0] if directors else None

            # Extract keywords
            keywords = details.get('keywords', {}).get('keywords', [])[:5]
            movie_data['keywords'] = [k['name'] for k in keywords]

            enriched_movies.append(movie_data)

            if (idx + 1) % 10 == 0:
                print(f"Processed {idx + 1}/{len(merged)} movies...")

        # Rate limit: 40 requests per 10 seconds
        time.sleep(0.26)  # ~3.8 requests/second

    # Save to CSV
    result_df = pd.DataFrame(enriched_movies)
    result_df.to_csv('data/raw/movies_enriched.csv', index=False)

    print(f"\n✅ Saved {len(enriched_movies)} movies to data/raw/movies_enriched.csv")

if __name__ == '__main__':
    main()
```

**4. Run Data Fetching**:
```bash
python scripts/fetch_tmdb_data.py

# This will take ~20-30 minutes for 5000 movies
# You can stop and resume by modifying the script
```

**Verification**:
```bash
# Check the output file
wc -l data/raw/movies_enriched.csv
head -n 5 data/raw/movies_enriched.csv

# Should have columns: movieId, tmdb_id, title, overview, genres, cast, director, etc.
```

**Commit**: `git commit -m "Add TMDB data fetching script and enriched movie data"`

---

#### Day 6-7: Database Setup

**Tasks**:
- [ ] Design database schema
- [ ] Create database models (SQLAlchemy)
- [ ] Create database initialization script
- [ ] Load movies into database
- [ ] Verify data integrity

**Step-by-Step**:

**1. Create Database Models**:

Create `app/models/movie.py`:
```python
from app import db
import json

class Movie(db.Model):
    __tablename__ = 'movies'

    movie_id = db.Column(db.Integer, primary_key=True)
    tmdb_id = db.Column(db.Integer, unique=True)
    imdb_id = db.Column(db.String(20))
    title = db.Column(db.String(255), nullable=False)
    release_year = db.Column(db.Integer)
    overview = db.Column(db.Text)
    genres = db.Column(db.Text)  # JSON string
    cast = db.Column(db.Text)    # JSON string
    director = db.Column(db.String(100))
    keywords = db.Column(db.Text)  # JSON string
    vote_average = db.Column(db.Float)
    popularity = db.Column(db.Float)
    poster_path = db.Column(db.String(255))
    genre_vector = db.Column(db.Text)    # JSON string
    feature_vector = db.Column(db.Text)   # JSON string

    def get_genres_list(self):
        return json.loads(self.genres) if self.genres else []

    def get_cast_list(self):
        return json.loads(self.cast) if self.cast else []

    def __repr__(self):
        return f'<Movie {self.title} ({self.release_year})>'
```

Create `app/models/user.py`:
```python
from app import db
from datetime import datetime
from werkzeug.security import generate_password_hash, check_password_hash
import json

class User(db.Model):
    __tablename__ = 'users'

    user_id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(50), unique=True, nullable=False)
    email = db.Column(db.String(100), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    preferred_genres = db.Column(db.Text)      # JSON
    preferred_industry = db.Column(db.String(50))
    preferred_format = db.Column(db.String(20))
    taste_vector = db.Column(db.Text)          # JSON

    # Relationships
    favorites = db.relationship('UserFavorite', backref='user', lazy=True)
    interactions = db.relationship('UserInteraction', backref='user', lazy=True)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)

    def get_taste_vector(self):
        return json.loads(self.taste_vector) if self.taste_vector else {}

    def set_taste_vector(self, vector_dict):
        self.taste_vector = json.dumps(vector_dict)

    def __repr__(self):
        return f'<User {self.username}>'


class UserFavorite(db.Model):
    __tablename__ = 'user_favorites'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False)
    added_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (db.UniqueConstraint('user_id', 'movie_id', name='unique_user_movie'),)

    def __repr__(self):
        return f'<UserFavorite user={self.user_id} movie={self.movie_id}>'


class UserInteraction(db.Model):
    __tablename__ = 'user_interactions'

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.user_id'), nullable=False)
    movie_id = db.Column(db.Integer, db.ForeignKey('movies.movie_id'), nullable=False)
    interaction_type = db.Column(db.String(20))  # 'like', 'view', 'rate'
    rating = db.Column(db.Float)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Interaction {self.interaction_type} user={self.user_id} movie={self.movie_id}>'
```

Create `app/models/__init__.py`:
```python
from app.models.movie import Movie
from app.models.user import User, UserFavorite, UserInteraction

__all__ = ['Movie', 'User', 'UserFavorite', 'UserInteraction']
```

**2. Update Flask App Initialization**:

Update `app/__init__.py`:
```python
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
import os

db = SQLAlchemy()

def create_app():
    app = Flask(__name__)

    # Configuration
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///movies.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Initialize extensions
    db.init_app(app)
    CORS(app)

    # Import models
    from app.models import Movie, User, UserFavorite, UserInteraction

    # Create tables
    with app.app_context():
        db.create_all()

    return app
```

Create `app/config.py`:
```python
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
    SQLALCHEMY_DATABASE_URI = os.getenv('DATABASE_URL', 'sqlite:///movies.db')
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    TMDB_API_KEY = os.getenv('TMDB_API_KEY')
```

**3. Create Database Setup Script**:

Create `scripts/setup_database.py`:
```python
from app import create_app, db
from app.models import Movie, User, UserFavorite, UserInteraction

def setup_database():
    """Initialize database and create tables"""
    app = create_app()

    with app.app_context():
        # Drop all tables (careful in production!)
        db.drop_all()
        print("Dropped existing tables")

        # Create all tables
        db.create_all()
        print("Created all tables")

        print("\n✅ Database setup complete!")
        print(f"Database location: {app.config['SQLALCHEMY_DATABASE_URI']}")

if __name__ == '__main__':
    setup_database()
```

**4. Create Data Loading Script**:

Create `scripts/load_movies.py`:
```python
import pandas as pd
import json
from app import create_app, db
from app.models import Movie

def load_movies():
    """Load movies from CSV into database"""
    app = create_app()

    with app.app_context():
        # Read enriched movies CSV
        df = pd.read_csv('data/raw/movies_enriched.csv')

        print(f"Loading {len(df)} movies into database...")

        movies_added = 0

        for idx, row in df.iterrows():
            # Skip if essential data is missing
            if pd.isna(row['title']) or pd.isna(row['overview']):
                continue

            # Parse JSON fields
            genres = json.loads(row['genres']) if isinstance(row['genres'], str) else row['genres']
            cast = json.loads(row['cast']) if isinstance(row['cast'], str) else row['cast']
            keywords = json.loads(row['keywords']) if isinstance(row['keywords'], str) else row['keywords']

            # Create Movie object
            movie = Movie(
                movie_id=int(row['movieId']),
                tmdb_id=int(row['tmdb_id']) if not pd.isna(row['tmdb_id']) else None,
                imdb_id=str(row['imdb_id']) if not pd.isna(row['imdb_id']) else None,
                title=row['title'],
                release_year=int(row['release_year']) if not pd.isna(row['release_year']) else None,
                overview=row['overview'],
                genres=json.dumps(genres),
                cast=json.dumps(cast),
                director=row['director'] if not pd.isna(row['director']) else None,
                keywords=json.dumps(keywords),
                vote_average=float(row['vote_average']) if not pd.isna(row['vote_average']) else None,
                popularity=float(row['popularity']) if not pd.isna(row['popularity']) else None,
                poster_path=row['poster_path'] if not pd.isna(row['poster_path']) else None
            )

            db.session.add(movie)
            movies_added += 1

            if (idx + 1) % 100 == 0:
                db.session.commit()
                print(f"Committed {idx + 1} movies...")

        # Final commit
        db.session.commit()

        print(f"\n✅ Successfully loaded {movies_added} movies into database")

        # Verify
        total = Movie.query.count()
        print(f"Total movies in database: {total}")

if __name__ == '__main__':
    load_movies()
```

**5. Run Database Setup**:
```bash
# Initialize database
python scripts/setup_database.py

# Load movies
python scripts/load_movies.py
```

**Verification**:
```bash
# Check database file exists
ls -lh movies.db

# Test queries with Python
python -c "
from app import create_app, db
from app.models import Movie

app = create_app()
with app.app_context():
    count = Movie.query.count()
    sample = Movie.query.first()
    print(f'Total movies: {count}')
    print(f'Sample movie: {sample.title}')
"
```

**Commit**: `git commit -m "Add database models and data loading scripts"`

---

### ✅ Phase 1 Complete Checklist

- [ ] Virtual environment created and activated
- [ ] All dependencies installed
- [ ] TMDB API key obtained
- [ ] MovieLens dataset downloaded
- [ ] Movie data enriched with TMDB API
- [ ] Database schema created
- [ ] 3000+ movies loaded into database
- [ ] Data verified with test queries

**Deliverable**: Database with 3000-5000 movies, ready for ML processing

---

## Phase 2: Core ML Engine (Week 2-3)

**Goal**: Build recommendation engine that returns top N movies for a user

### Week 2: Feature Engineering & Similarity

#### Day 8-10: Feature Engineering

**Tasks**:
- [ ] Extract and encode genres
- [ ] Create TF-IDF vectors from plots
- [ ] Build movie feature vectors
- [ ] Save vectors to database

**Step-by-Step**:

**1. Create Vectorizer Module**:

Create `app/ml/vectorizer.py`:
```python
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MultiLabelBinarizer
import json
import pickle

class MovieVectorizer:
    """Convert movie data into numerical vectors"""

    def __init__(self):
        self.genre_encoder = MultiLabelBinarizer()
        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=5000,
            stop_words='english',
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.8
        )
        self.all_genres = []

    def fit_transform_genres(self, movies_df):
        """Convert genres to one-hot vectors"""
        # Parse genre strings
        genres_list = movies_df['genres'].apply(
            lambda x: json.loads(x) if isinstance(x, str) else x
        ).tolist()

        # Fit and transform
        genre_matrix = self.genre_encoder.fit_transform(genres_list)
        self.all_genres = self.genre_encoder.classes_.tolist()

        print(f"Genre encoding: {len(self.all_genres)} unique genres")
        return genre_matrix

    def transform_genres(self, genres_list):
        """Transform a single movie's genres"""
        return self.genre_encoder.transform([genres_list])[0]

    def fit_transform_plots(self, movies_df):
        """Convert plot overviews to TF-IDF vectors"""
        plots = movies_df['overview'].fillna('')

        tfidf_matrix = self.tfidf_vectorizer.fit_transform(plots)

        print(f"TF-IDF: {tfidf_matrix.shape[0]} movies × {tfidf_matrix.shape[1]} features")
        return tfidf_matrix

    def transform_plot(self, plot_text):
        """Transform a single plot"""
        return self.tfidf_vectorizer.transform([plot_text])

    def save(self, filepath):
        """Save vectorizer state"""
        with open(filepath, 'wb') as f:
            pickle.dump({
                'genre_encoder': self.genre_encoder,
                'tfidf_vectorizer': self.tfidf_vectorizer,
                'all_genres': self.all_genres
            }, f)
        print(f"Saved vectorizer to {filepath}")

    def load(self, filepath):
        """Load vectorizer state"""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
            self.genre_encoder = data['genre_encoder']
            self.tfidf_vectorizer = data['tfidf_vectorizer']
            self.all_genres = data['all_genres']
        print(f"Loaded vectorizer from {filepath}")
```

**2. Create Preprocessing Script**:

Create `scripts/preprocess_features.py`:
```python
import pandas as pd
import numpy as np
from app import create_app, db
from app.models import Movie
from app.ml.vectorizer import MovieVectorizer
import json
import pickle

def preprocess_features():
    """Extract features and create vectors for all movies"""
    app = create_app()

    with app.app_context():
        # Load all movies
        movies = Movie.query.all()
        print(f"Processing {len(movies)} movies...")

        # Create DataFrame
        data = []
        for movie in movies:
            data.append({
                'movie_id': movie.movie_id,
                'title': movie.title,
                'genres': movie.genres,
                'overview': movie.overview,
                'cast': movie.cast,
                'director': movie.director
            })

        df = pd.DataFrame(data)

        # Initialize vectorizer
        vectorizer = MovieVectorizer()

        # 1. Genre vectors
        print("\n1. Creating genre vectors...")
        genre_matrix = vectorizer.fit_transform_genres(df)

        # 2. TF-IDF on plots
        print("\n2. Creating TF-IDF vectors from plots...")
        tfidf_matrix = vectorizer.fit_transform_plots(df)

        # 3. Save genre vectors to database
        print("\n3. Saving genre vectors to database...")
        for idx, movie in enumerate(movies):
            genre_vec = genre_matrix[idx].tolist()
            movie.genre_vector = json.dumps(genre_vec)

            if (idx + 1) % 100 == 0:
                db.session.commit()
                print(f"  Saved {idx + 1}/{len(movies)} movies...")

        db.session.commit()
        print("✅ Genre vectors saved to database")

        # 4. Save models
        print("\n4. Saving models to disk...")
        vectorizer.save('data/models/vectorizer.pkl')

        # Save TF-IDF matrix (sparse)
        import scipy.sparse as sp
        sp.save_npz('data/models/tfidf_matrix.npz', tfidf_matrix)

        # Save movie ID mapping
        movie_id_map = {movie_id: idx for idx, movie_id in enumerate(df['movie_id'])}
        with open('data/models/movie_id_map.pkl', 'wb') as f:
            pickle.dump(movie_id_map, f)

        print("\n✅ Feature preprocessing complete!")
        print(f"   - Genre vectors: {genre_matrix.shape}")
        print(f"   - TF-IDF matrix: {tfidf_matrix.shape}")
        print(f"   - Files saved to data/models/")

if __name__ == '__main__':
    preprocess_features()
```

**3. Run Preprocessing**:
```bash
python scripts/preprocess_features.py
```

**Verification**:
```bash
ls -lh data/models/
# Should see:
# - vectorizer.pkl
# - tfidf_matrix.npz
# - movie_id_map.pkl
```

**Commit**: `git commit -m "Add feature engineering and vectorization"`

---

#### Day 11-13: Similarity Computation

**Tasks**:
- [ ] Implement cosine similarity functions
- [ ] Pre-compute similarity matrices
- [ ] Create similarity module
- [ ] Test similarity calculations

**Step-by-Step**:

**1. Create Similarity Module**:

Create `app/ml/similarity.py`:
```python
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import scipy.sparse as sp

class SimilarityEngine:
    """Calculate similarities between movies"""

    def __init__(self):
        self.tfidf_matrix = None
        self.plot_similarity_matrix = None
        self.movie_id_map = None
        self.reverse_movie_map = None

    def load_models(self):
        """Load pre-computed matrices"""
        # Load TF-IDF matrix
        self.tfidf_matrix = sp.load_npz('data/models/tfidf_matrix.npz')
        print(f"Loaded TF-IDF matrix: {self.tfidf_matrix.shape}")

        # Load movie ID mapping
        with open('data/models/movie_id_map.pkl', 'rb') as f:
            self.movie_id_map = pickle.load(f)

        self.reverse_movie_map = {v: k for k, v in self.movie_id_map.items()}
        print(f"Loaded {len(self.movie_id_map)} movie mappings")

        # Try to load pre-computed similarity
        try:
            self.plot_similarity_matrix = np.load('data/models/plot_similarity.npy')
            print(f"Loaded pre-computed plot similarity: {self.plot_similarity_matrix.shape}")
        except FileNotFoundError:
            print("Plot similarity matrix not found (will compute on-demand)")

    def compute_plot_similarity(self, movie_id_1, movie_id_2):
        """Compute plot similarity between two movies"""
        if movie_id_1 not in self.movie_id_map or movie_id_2 not in self.movie_id_map:
            return 0.0

        idx1 = self.movie_id_map[movie_id_1]
        idx2 = self.movie_id_map[movie_id_2]

        # If we have pre-computed matrix, use it
        if self.plot_similarity_matrix is not None:
            return float(self.plot_similarity_matrix[idx1, idx2])

        # Otherwise compute on-demand
        vec1 = self.tfidf_matrix[idx1]
        vec2 = self.tfidf_matrix[idx2]

        similarity = cosine_similarity(vec1, vec2)[0][0]
        return float(similarity)

    def get_similar_movies_by_plot(self, movie_id, n=10):
        """Get top N movies with similar plots"""
        if movie_id not in self.movie_id_map:
            return []

        idx = self.movie_id_map[movie_id]

        # Compute similarity with all movies
        movie_vec = self.tfidf_matrix[idx]
        similarities = cosine_similarity(movie_vec, self.tfidf_matrix)[0]

        # Get top N (excluding itself)
        similar_indices = similarities.argsort()[::-1][1:n+1]

        # Convert back to movie IDs with scores
        results = []
        for sim_idx in similar_indices:
            results.append({
                'movie_id': self.reverse_movie_map[sim_idx],
                'similarity': float(similarities[sim_idx])
            })

        return results

    @staticmethod
    def compute_cosine_similarity(vec1, vec2):
        """Generic cosine similarity"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)

        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return float(dot_product / (norm1 * norm2))
```

**2. Create Similarity Pre-computation Script**:

Create `scripts/precompute_similarities.py`:
```python
import numpy as np
import scipy.sparse as sp
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def precompute_similarities():
    """Pre-compute plot similarity matrix (optional but recommended)"""
    print("Loading TF-IDF matrix...")
    tfidf_matrix = sp.load_npz('data/models/tfidf_matrix.npz')

    n_movies = tfidf_matrix.shape[0]
    print(f"Computing similarities for {n_movies} movies...")
    print("⚠️  This may take a few minutes...")

    # Compute pairwise cosine similarity
    # Note: For large datasets (10k+), this creates a huge matrix
    # Consider computing on-demand or using approximate methods

    if n_movies > 10000:
        print("⚠️  Dataset is large. Consider on-demand computation instead.")
        print("   Skipping pre-computation.")
        return

    similarity_matrix = cosine_similarity(tfidf_matrix)

    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    print(f"Memory size: {similarity_matrix.nbytes / 1e6:.2f} MB")

    # Save to disk
    np.save('data/models/plot_similarity.npy', similarity_matrix)

    print("✅ Plot similarity matrix saved to data/models/plot_similarity.npy")

if __name__ == '__main__':
    precompute_similarities()
```

**3. Run Similarity Pre-computation** (Optional):
```bash
# Only run if you have < 10k movies
python scripts/precompute_similarities.py
```

**4. Test Similarity**:

Create `notebooks/test_similarity.ipynb`:
```python
# Jupyter Notebook for testing

from app import create_app, db
from app.models import Movie
from app.ml.similarity import SimilarityEngine

app = create_app()

with app.app_context():
    # Load similarity engine
    sim_engine = SimilarityEngine()
    sim_engine.load_models()

    # Test with a movie (e.g., "Inception")
    test_movie = Movie.query.filter_by(title='Inception').first()

    if test_movie:
        print(f"Finding movies similar to: {test_movie.title}")

        similar = sim_engine.get_similar_movies_by_plot(test_movie.movie_id, n=10)

        print("\nTop 10 similar movies:")
        for i, result in enumerate(similar, 1):
            movie = Movie.query.get(result['movie_id'])
            print(f"{i}. {movie.title} - Similarity: {result['similarity']:.3f}")
```

**Verification**:
```bash
jupyter notebook notebooks/test_similarity.ipynb
# Run all cells and verify results make sense
```

**Commit**: `git commit -m "Add similarity computation engine"`

---

#### Day 14: Recommendation Engine Core

**Tasks**:
- [ ] Create recommender module
- [ ] Implement hybrid scoring
- [ ] Test with sample users
- [ ] Optimize performance

**Create Main Recommender**:

Create `app/ml/recommender.py`:
```python
import numpy as np
import json
from app import db
from app.models import Movie, User, UserFavorite
from app.ml.similarity import SimilarityEngine

class RecommendationEngine:
    """Core recommendation engine"""

    def __init__(self):
        self.similarity_engine = SimilarityEngine()
        self.similarity_engine.load_models()

    def get_recommendations(self, user_id, mood=None, limit=10):
        """
        Generate personalized recommendations

        Args:
            user_id: User ID
            mood: Optional mood string ('happy', 'sad', etc.)
            limit: Number of recommendations to return

        Returns:
            List of recommended movies with scores
        """
        user = User.query.get(user_id)
        if not user:
            raise ValueError(f"User {user_id} not found")

        # Get user's favorite movies
        favorites = UserFavorite.query.filter_by(user_id=user_id).all()
        favorite_movie_ids = [fav.movie_id for fav in favorites]

        if not favorite_movie_ids:
            # Cold start: return popular movies
            return self._get_popular_movies(limit)

        # Get user's taste vector
        taste_vector = user.get_taste_vector()

        # Get candidate movies (exclude already watched)
        candidates = Movie.query.filter(
            ~Movie.movie_id.in_(favorite_movie_ids)
        ).all()

        # Score each candidate
        scored_movies = []
        for movie in candidates:
            score = self._calculate_movie_score(
                movie,
                favorite_movie_ids,
                taste_vector,
                mood
            )
            scored_movies.append({
                'movie': movie,
                'score': score
            })

        # Sort by score
        scored_movies.sort(key=lambda x: x['score'], reverse=True)

        # Apply diversity filter
        diverse_movies = self._apply_diversity_filter(scored_movies, limit * 2)

        # Return top N
        return diverse_movies[:limit]

    def _calculate_movie_score(self, movie, favorite_ids, taste_vector, mood):
        """Calculate hybrid score for a movie"""

        # 1. Content similarity (40%) - average similarity to favorites
        content_scores = []
        for fav_id in favorite_ids:
            sim = self.similarity_engine.compute_plot_similarity(fav_id, movie.movie_id)
            content_scores.append(sim)
        content_score = np.mean(content_scores) if content_scores else 0.0

        # 2. Plot similarity (30%) - max similarity to any favorite
        plot_score = max(content_scores) if content_scores else 0.0

        # 3. Taste match (20%)
        movie_genres = movie.get_genres_list()
        taste_score = self._compute_taste_match(movie_genres, taste_vector)

        # 4. Base score
        base_score = (
            0.4 * content_score +
            0.3 * plot_score +
            0.2 * taste_score
        )

        # 5. Apply mood boost (multiplicative)
        mood_multiplier = self._get_mood_boost(mood, movie_genres)

        final_score = base_score * mood_multiplier

        return final_score

    def _compute_taste_match(self, movie_genres, taste_vector):
        """Compute how well movie matches user's taste"""
        if not taste_vector or not movie_genres:
            return 0.0

        # Sum of taste weights for movie's genres
        total_taste = sum(taste_vector.get(genre, 0) for genre in movie_genres)

        # Normalize by number of genres
        if movie_genres:
            return total_taste / len(movie_genres)

        return 0.0

    def _get_mood_boost(self, mood, movie_genres):
        """Get mood-based boost factor"""
        if not mood or not movie_genres:
            return 1.0

        MOOD_GENRE_MAP = {
            'happy': {
                'Comedy': 1.5, 'Animation': 1.3, 'Family': 1.2, 'Romance': 1.1
            },
            'sad': {
                'Drama': 1.4, 'Romance': 1.2, 'Music': 1.1
            },
            'stressed': {
                'Comedy': 1.5, 'Animation': 1.3, 'Adventure': 1.2
            },
            'romantic': {
                'Romance': 1.6, 'Drama': 1.2, 'Comedy': 1.1
            },
            'motivated': {
                'Action': 1.4, 'Adventure': 1.3, 'Biography': 1.2, 'Sport': 1.2
            },
            'thriller': {
                'Thriller': 1.5, 'Mystery': 1.4, 'Crime': 1.3, 'Horror': 1.2
            }
        }

        mood = mood.lower()
        if mood not in MOOD_GENRE_MAP:
            return 1.0

        # Get max boost for any matching genre
        boosts = [
            MOOD_GENRE_MAP[mood].get(genre, 1.0)
            for genre in movie_genres
        ]

        return max(boosts) if boosts else 1.0

    def _apply_diversity_filter(self, scored_movies, n):
        """Ensure genre diversity in recommendations"""
        diverse_set = []
        genre_count = {}

        for item in scored_movies:
            movie = item['movie']
            genres = movie.get_genres_list()
            primary_genre = genres[0] if genres else 'Unknown'

            # Limit to 3 movies per genre
            if genre_count.get(primary_genre, 0) < 3:
                diverse_set.append(item)
                genre_count[primary_genre] = genre_count.get(primary_genre, 0) + 1

            if len(diverse_set) >= n:
                break

        return diverse_set

    def _get_popular_movies(self, limit):
        """Fallback for cold start"""
        popular = Movie.query.order_by(
            Movie.vote_average.desc(),
            Movie.popularity.desc()
        ).limit(limit).all()

        return [{'movie': m, 'score': 0.5} for m in popular]
```

**Test the Recommender**:
```python
# Test script
from app import create_app, db
from app.models import User, UserFavorite, Movie
from app.ml.recommender import RecommendationEngine

app = create_app()

with app.app_context():
    # Create test user
    test_user = User(
        username='test_user',
        email='test@example.com',
        preferred_genres='["Action", "Sci-Fi"]',
        taste_vector='{"Action": 0.8, "Sci-Fi": 0.9}'
    )
    test_user.set_password('password123')

    db.session.add(test_user)
    db.session.commit()

    # Add some favorites
    inception = Movie.query.filter_by(title='Inception').first()
    if inception:
        fav = UserFavorite(user_id=test_user.user_id, movie_id=inception.movie_id)
        db.session.add(fav)
        db.session.commit()

    # Get recommendations
    engine = RecommendationEngine()
    recommendations = engine.get_recommendations(test_user.user_id, mood='happy', limit=10)

    print("Recommendations:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['movie'].title} - Score: {rec['score']:.3f}")
```

**Commit**: `git commit -m "Implement core recommendation engine with hybrid scoring"`

---

### ✅ Phase 2 Complete Checklist

- [ ] Genre vectors created and saved
- [ ] TF-IDF matrix computed from plots
- [ ] Similarity engine implemented
- [ ] Recommendation engine working
- [ ] Tested with sample users
- [ ] Performance < 500ms per request

**Deliverable**: Working recommendation function that returns top 10 movies

---

## Phase 3: Backend API (Week 3-4)

**Goal**: Create REST API with all endpoints

### Week 3-4: Flask Routes & API

#### Day 15-17: Flask Setup & Authentication

**Tasks**:
- [ ] Set up Flask blueprints
- [ ] Create authentication routes
- [ ] Implement signup with taste collection
- [ ] Implement login/logout
- [ ] Test with Postman

**See PRD Section 13 for complete API specifications**

**Quick Implementation Checklist**:
- [ ] Create `app/routes/auth.py`
- [ ] Create `app/routes/recommendation.py`
- [ ] Create `app/routes/user.py`
- [ ] Update `app/__init__.py` to register blueprints
- [ ] Create `run.py` entry point
- [ ] Test all endpoints with Postman

**Commit**: `git commit -m "Implement REST API with authentication"`

---

#### Day 18-20: Recommendation API

**Create recommendation route** in `app/routes/recommendation.py`:
- POST `/api/recommend` - Generate recommendations
- Include mood parameter handling
- Call RecommendationEngine
- Generate explanations
- Return JSON response

**Commit**: `git commit -m "Add recommendation API endpoint with mood support"`

---

#### Day 21: User Profile API

**Create user routes** in `app/routes/user.py`:
- GET `/api/user/profile`
- PUT `/api/user/profile`
- POST `/api/user/favorites`
- DELETE `/api/user/favorites/:id`
- GET `/api/user/favorites`

**Commit**: `git commit -m "Add user profile management endpoints"`

---

### ✅ Phase 3 Complete Checklist

- [ ] All API endpoints implemented
- [ ] Authentication working (signup, login, logout)
- [ ] Recommendation endpoint returns results
- [ ] User profile CRUD operations working
- [ ] Tested with Postman
- [ ] Error handling added

**Deliverable**: Functional REST API

---

## Phase 4: Frontend & Integration (Week 4-5)

**Goal**: Build web interface and connect to API

### Week 4-5: HTML, CSS, JavaScript

#### Day 22-24: Core Pages

**Tasks**:
- [ ] Create base template (`templates/base.html`)
- [ ] Landing page (`templates/index.html`)
- [ ] Signup page with multi-step form (`templates/signup.html`)
- [ ] Login page (`templates/login.html`)
- [ ] Dashboard with mood selector (`templates/dashboard.html`)

**Key Features**:
- Use Bootstrap 5 for styling
- Responsive design (mobile-friendly)
- Form validation
- Loading states

**Commit**: `git commit -m "Add core HTML templates and styling"`

---

#### Day 25-27: Recommendations UI

**Tasks**:
- [ ] Recommendations page (`templates/recommendations.html`)
- [ ] Movie card component (poster, title, genres, rating)
- [ ] Explanation display (collapsible)
- [ ] Like button functionality
- [ ] JavaScript for AJAX calls (`static/js/recommendations.js`)

**Commit**: `git commit -m "Add recommendations UI with movie cards"`

---

#### Day 28: Full Integration

**Tasks**:
- [ ] Connect all pages to backend APIs
- [ ] Test full user flow (signup → login → recommend → like)
- [ ] Fix bugs
- [ ] Add loading spinners
- [ ] Handle error states

**Commit**: `git commit -m "Complete frontend-backend integration"`

---

### ✅ Phase 4 Complete Checklist

- [ ] All pages created and styled
- [ ] Frontend connected to backend
- [ ] Full user flow working
- [ ] Responsive on mobile
- [ ] No console errors

**Deliverable**: Fully functional web application

---

## Phase 5: Adaptive Learning (Week 5-6)

**Goal**: Implement taste vector updates when users like movies

### Week 5-6: Adaptive Learning

#### Day 29-31: Taste Vector Updates

**Tasks**:
- [ ] Create `app/ml/taste_updater.py`
- [ ] Implement EMA update algorithm
- [ ] Integrate with like button
- [ ] Test taste evolution

**Algorithm**:
```python
new_taste[genre] = (1 - α) × old_taste[genre] + α × new_movie[genre]
where α = 0.2
```

**Commit**: `git commit -m "Implement adaptive learning with taste vector updates"`

---

#### Day 32-33: Explanation Enhancement

**Tasks**:
- [ ] Create `app/utils/explainer.py`
- [ ] Generate explanations for each recommendation
- [ ] Show taste update notification after like
- [ ] Display improved explanations

**Commit**: `git commit -m "Add explainable AI output with reasons"`

---

### ✅ Phase 5 Complete Checklist

- [ ] Taste vector updates on like
- [ ] Recommendations improve after 5+ likes
- [ ] Users notified of taste changes
- [ ] Explanations clear and helpful

**Deliverable**: System that learns from user interactions

---

## Phase 6: Testing & Polish (Week 6-8)

**Goal**: Ensure quality, fix bugs, polish UI

### Week 6-7: Testing

#### Day 34-37: Unit & Integration Tests

**Tasks**:
- [ ] Write tests for ML functions (`tests/test_recommender.py`)
- [ ] Write tests for API routes (`tests/test_routes.py`)
- [ ] Write tests for taste updater (`tests/test_taste_updater.py`)
- [ ] Run all tests and fix failures

**Run tests**:
```bash
pytest tests/ -v
```

**Commit**: `git commit -m "Add comprehensive test suite"`

---

#### Day 38-40: User Testing & Bug Fixes

**Tasks**:
- [ ] Test with 5+ real users
- [ ] Collect feedback
- [ ] Fix critical bugs
- [ ] Improve UI based on feedback

---

### Week 8: Final Polish

#### Day 41-42: Documentation

**Tasks**:
- [ ] Update README.md with setup instructions
- [ ] Add API documentation
- [ ] Comment code
- [ ] Create user guide

**Commit**: `git commit -m "Add comprehensive documentation"`

---

#### Day 43-44: Demo Preparation

**Tasks**:
- [ ] Record demo video (5-10 minutes)
- [ ] Create presentation slides
- [ ] Prepare talking points
- [ ] Practice demo

---

#### Day 45: Final Review

**Tasks**:
- [ ] Code review
- [ ] Performance optimization
- [ ] Final testing
- [ ] Deployment (optional)

---

### ✅ Phase 6 Complete Checklist

- [ ] All tests passing
- [ ] No critical bugs
- [ ] Documentation complete
- [ ] Demo ready
- [ ] Project ready for submission

**Deliverable**: Production-ready application

---

## Daily Checklist Template

Copy this for each day:

```markdown
### Day X - [Date]

**Focus**: [Main goal for the day]

**Tasks**:
- [ ] Task 1
- [ ] Task 2
- [ ] Task 3

**Progress**:
- Started: [time]
- Completed: [time]
- Blockers: [any issues]

**Learnings**:
- [What I learned today]

**Tomorrow**:
- [What to focus on next]

**Git Commit**:
- [ ] Committed today's work
- Commit message: "[description]"
```

---

## Quick Commands Reference

### Daily Development

```bash
# Activate environment
source venv/bin/activate

# Run Flask app
python run.py

# Run tests
pytest tests/ -v

# Check database
python -c "from app import create_app, db; from app.models import *; app = create_app(); app.app_context().push(); print(f'Movies: {Movie.query.count()}')"

# Git workflow
git status
git add .
git commit -m "Description"
git push origin main
```

### Data Scripts

```bash
# Fetch TMDB data
python scripts/fetch_tmdb_data.py

# Setup database
python scripts/setup_database.py

# Load movies
python scripts/load_movies.py

# Preprocess features
python scripts/preprocess_features.py

# Precompute similarities
python scripts/precompute_similarities.py
```

### Testing

```bash
# Run specific test file
pytest tests/test_recommender.py -v

# Run with coverage
pytest --cov=app tests/

# Test API endpoint
curl -X POST http://localhost:5000/api/auth/login \
  -H "Content-Type: application/json" \
  -d '{"email":"test@example.com","password":"password123"}'
```

### Database Management

```bash
# SQLite CLI
sqlite3 movies.db

# In SQLite:
.tables                     # List tables
SELECT COUNT(*) FROM movies;
SELECT * FROM users LIMIT 5;
.quit
```

---

## Progress Tracking

### Week 1
- [ ] Day 1: Environment setup
- [ ] Day 2: Git initialization
- [ ] Day 3: TMDB API setup
- [ ] Day 4: Data fetching
- [ ] Day 5: Data verification
- [ ] Day 6: Database models
- [ ] Day 7: Data loading

### Week 2
- [ ] Day 8: Feature engineering start
- [ ] Day 9: Genre encoding
- [ ] Day 10: TF-IDF implementation
- [ ] Day 11: Similarity module
- [ ] Day 12: Similarity testing
- [ ] Day 13: Similarity optimization
- [ ] Day 14: Recommender core

### Week 3
- [ ] Day 15: Flask blueprints
- [ ] Day 16: Auth routes
- [ ] Day 17: Auth testing
- [ ] Day 18: Recommendation endpoint
- [ ] Day 19: Mood logic
- [ ] Day 20: Explanation generation
- [ ] Day 21: User profile API

### Week 4
- [ ] Day 22: Base templates
- [ ] Day 23: Signup page
- [ ] Day 24: Dashboard
- [ ] Day 25: Recommendations UI
- [ ] Day 26: Movie cards
- [ ] Day 27: JavaScript integration
- [ ] Day 28: Full integration

### Week 5
- [ ] Day 29: Taste updater
- [ ] Day 30: EMA algorithm
- [ ] Day 31: Like integration
- [ ] Day 32: Explainer module
- [ ] Day 33: UI polish

### Week 6
- [ ] Day 34: Unit tests
- [ ] Day 35: Integration tests
- [ ] Day 36: API tests
- [ ] Day 37: Test fixes

### Week 7
- [ ] Day 38: User testing
- [ ] Day 39: Bug fixes
- [ ] Day 40: UI improvements

### Week 8
- [ ] Day 41: Documentation
- [ ] Day 42: README
- [ ] Day 43: Demo video
- [ ] Day 44: Presentation
- [ ] Day 45: Final review

---

## Key Milestones

| Milestone | Target Date | Status | Notes |
|-----------|-------------|--------|-------|
| Environment Ready | Week 1, Day 2 | ⬜ Pending | |
| Data Loaded | Week 1, Day 7 | ⬜ Pending | 3000+ movies |
| ML Engine Working | Week 3, Day 14 | ⬜ Pending | Returns recommendations |
| API Complete | Week 4, Day 21 | ⬜ Pending | All endpoints tested |
| Frontend Done | Week 5, Day 28 | ⬜ Pending | Full user flow |
| Adaptive Learning | Week 6, Day 33 | ⬜ Pending | Taste updates working |
| Testing Complete | Week 7, Day 40 | ⬜ Pending | All tests pass |
| Project Ready | Week 8, Day 45 | ⬜ Pending | Ready to present |

---

## Tips for Success

### Time Management
- ⏰ **Set daily goals**: Work 2-4 hours per day consistently
- 📅 **Weekly reviews**: Every Sunday, review progress and adjust plan
- 🎯 **Focus on MVP**: Don't add extra features until core is done
- ⚡ **Quick wins**: Start each day with an easy task to build momentum

### Problem-Solving
- 🐛 **Stuck for > 30 min?** Take a break, Google it, or ask for help
- 📝 **Document blockers**: Keep a list of issues and solutions
- 🔍 **Read error messages**: Most errors have clear solutions
- 💡 **Experiment in Jupyter**: Test algorithms before implementing

### Code Quality
- ✅ **Test as you build**: Don't wait until the end to test
- 💬 **Comment complex code**: Future you will thank present you
- 🔄 **Refactor gradually**: Clean up code as you go
- 📦 **Commit frequently**: Small, logical commits are better

### Staying Motivated
- 🎉 **Celebrate milestones**: Reward yourself after each phase
- 👥 **Share progress**: Show friends/classmates your work
- 📊 **Track visually**: Use checkboxes, charts, or task boards
- 🎬 **Remember the goal**: You're building something cool!

---

## When Things Go Wrong

### Common Issues & Solutions

**Problem**: TMDB API not responding
- **Solution**: Check API key, check rate limits, add delays between requests

**Problem**: Database file locked
- **Solution**: Close all connections, restart Flask app, check for open SQLite sessions

**Problem**: Recommendations all the same
- **Solution**: Check diversity filter, verify similarity calculations, test with different users

**Problem**: Frontend not connecting to backend
- **Solution**: Check CORS settings, verify API endpoint URLs, check browser console for errors

**Problem**: Tests failing
- **Solution**: Run tests individually to isolate issue, check test data setup, verify database state

**Problem**: Out of memory
- **Solution**: Reduce dataset size, use sparse matrices, compute on-demand instead of pre-computing

---

## Resources

### Documentation
- Flask: https://flask.palletsprojects.com/
- Scikit-learn: https://scikit-learn.org/
- Bootstrap: https://getbootstrap.com/
- SQLAlchemy: https://www.sqlalchemy.org/

### Tutorials
- TF-IDF: https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction
- Cosine Similarity: https://scikit-learn.org/stable/modules/metrics.html#cosine-similarity
- Flask REST API: https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world

### Datasets
- MovieLens: https://grouplens.org/datasets/movielens/
- TMDB API: https://developers.themoviedb.org/

---

## Notes & Ideas

Use this space to jot down ideas, questions, or notes as you work:

```
[Space for your notes]
```

---

**Good luck with your project! Remember: progress over perfection. Build incrementally, test continuously, and celebrate small wins! 🚀**

---

**Last Updated**: February 25, 2026
**Current Phase**: Planning
**Next Action**: Day 1 - Environment Setup

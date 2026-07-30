# MovieMind - AI-Powered Movie Recommendation System

An intelligent movie recommendation system that learns from your preferences and adapts to your mood.

## Features

- **Personalized Recommendations**: AI-powered suggestions based on your unique taste
- **Mood-Based Filtering**: Get recommendations matching your current mood (happy, excited, romantic, etc.)
- **Adaptive Learning**: The system learns and improves from your likes and interactions
- **Explainable AI**: Every recommendation comes with a clear explanation of why it was suggested
- **Rich Movie Database**: 5,000+ movies with detailed information from TMDB

## Tech Stack

- **Backend**: Python, Flask, SQLAlchemy
- **Database**: SQLite
- **ML**: scikit-learn (TF-IDF, Cosine Similarity)
- **Frontend**: HTML5, CSS3, JavaScript, Bootstrap 5
- **Data Sources**: TMDB API, MovieLens Dataset

## Quick Start

### Prerequisites

- Python 3.10+
- pip

### Installation

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd PDL
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your TMDB API key
   ```

5. **Initialize database and load data**
   ```bash
   python scripts/setup_database.py
   python scripts/load_movies.py
   python scripts/preprocess_features.py
   ```

6. **Run the application**
   ```bash
   python run.py
   ```

7. **Open in browser**
   ```
   http://localhost:5000
   ```

## Project Structure

```
PDL/
├── app/
│   ├── __init__.py          # Flask app factory
│   ├── config.py            # Configuration
│   ├── models/              # Database models
│   │   ├── movie.py         # Movie model
│   │   └── user.py          # User, Favorites, Interactions
│   ├── routes/              # API endpoints
│   │   ├── auth.py          # Authentication
│   │   ├── recommendation.py # Recommendations API
│   │   ├── user.py          # User profile API
│   │   └── pages.py         # HTML page routes
│   ├── ml/                  # Machine learning
│   │   ├── vectorizer.py    # TF-IDF & genre encoding
│   │   ├── similarity.py    # Similarity calculations
│   │   ├── recommender.py   # Recommendation engine
│   │   ├── taste_updater.py # Adaptive learning
│   │   └── explainer.py     # Explanation generation
│   └── utils/               # Utilities
├── data/
│   ├── raw/                 # Raw data files
│   └── models/              # Trained ML models
├── scripts/                 # Setup scripts
├── templates/               # HTML templates
├── static/                  # CSS, JS, images
├── tests/                   # Test suite
├── requirements.txt
├── run.py                   # Entry point
└── README.md
```

## API Endpoints

### Authentication
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/auth/signup` | POST | Register new user |
| `/api/auth/login` | POST | Login |
| `/api/auth/logout` | POST | Logout |
| `/api/auth/check` | GET | Check auth status |

### Recommendations
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/recommend` | GET | Get personalized recommendations |
| `/api/similar/<id>` | GET | Get similar movies |
| `/api/movies` | GET | Browse/search movies |
| `/api/movies/<id>` | GET | Get movie details |
| `/api/genres` | GET | List all genres |
| `/api/moods` | GET | List mood options |

### User
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/user/profile` | GET/PUT | Get/update profile |
| `/api/user/favorites` | GET/POST | List/add favorites |
| `/api/user/favorites/<id>` | DELETE | Remove favorite |
| `/api/user/taste` | GET | Get taste vector |

## How the Recommendation Engine Works

### 1. Content-Based Filtering
- Movies are converted to TF-IDF vectors based on plots, genres, cast, and keywords
- Cosine similarity finds movies with similar content

### 2. Genre Matching
- User's preferred genres are tracked in a "taste vector"
- Movies matching preferred genres get higher scores

### 3. Mood Boosting
- Each mood maps to specific genres (e.g., "excited" → Action, Thriller)
- Movies in mood-matching genres receive score boosts

### 4. Adaptive Learning
- When users like movies, their taste vector updates using EMA
- Formula: `new_taste[genre] = (1 - α) × old_taste[genre] + α × 1.0`
- Learning rate α = 0.2 (20% weight to new data)

### 5. Diversity Filter
- Limits movies per genre to ensure varied recommendations

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=app

# Run specific test file
pytest tests/test_api.py -v
```

## Configuration

Environment variables (`.env`):

```
SECRET_KEY=your-secret-key
TMDB_API_KEY=your-tmdb-api-key
DATABASE_URL=sqlite:///movies.db
FLASK_ENV=development
```

## Mood Options

| Mood | Boosted Genres |
|------|----------------|
| Happy | Comedy, Animation, Family |
| Sad | Drama, Romance |
| Excited | Action, Adventure, Thriller |
| Relaxed | Comedy, Documentary |
| Romantic | Romance, Drama |
| Thrilled | Thriller, Horror, Mystery |
| Nostalgic | Drama, Family, Animation |
| Adventurous | Adventure, Action, Fantasy |
| Thoughtful | Documentary, Drama, Mystery |
| Scared | Horror, Thriller |

## Data Sources

- **MovieLens**: https://grouplens.org/datasets/movielens/
- **TMDB API**: https://www.themoviedb.org/

## License

This project is for educational purposes.

## Acknowledgments

- MovieLens for the movie ratings dataset
- TMDB for the movie metadata API
- scikit-learn for ML algorithms

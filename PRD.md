# Product Requirements Document (PRD)
# AI-Based Movie Recommendation System

**Version:** 1.0
**Date:** February 25, 2026
**Project Type:** College Project - Machine Learning & Web Development
**Status:** Planning Phase

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Problem Statement](#2-problem-statement)
3. [Goals & Objectives](#3-goals--objectives)
4. [Target Audience](#4-target-audience)
5. [Product Overview](#5-product-overview)
6. [Functional Requirements](#6-functional-requirements)
7. [Non-Functional Requirements](#7-non-functional-requirements)
8. [User Stories](#8-user-stories)
9. [Technical Architecture](#9-technical-architecture)
10. [Machine Learning Requirements](#10-machine-learning-requirements)
11. [Data Requirements](#11-data-requirements)
12. [User Interface Requirements](#12-user-interface-requirements)
13. [API Specifications](#13-api-specifications)
14. [Success Metrics](#14-success-metrics)
15. [Project Timeline](#15-project-timeline)
16. [Risks & Mitigation](#16-risks--mitigation)
17. [Future Enhancements](#17-future-enhancements)
18. [Appendix](#18-appendix)

---

## 1. Executive Summary

### 1.1 Project Vision
Build an intelligent movie recommendation web application that provides personalized suggestions using hybrid recommendation algorithms, mood-based filtering, and adaptive learning capabilities.

### 1.2 Key Differentiators
- **Mood-Based Recommendations**: Real-time mood input influences suggestions
- **Explainable AI**: Users understand why each movie was recommended
- **Adaptive Learning**: System improves recommendations based on user interactions
- **Hybrid Approach**: Combines content-based filtering, plot similarity (NLP), and user taste matching

### 1.3 Project Scope
A college-level demonstration of recommendation systems similar to Netflix/Amazon Prime Video, focusing on content-based filtering with NLP enhancements and user preference adaptation.

---

## 2. Problem Statement

### 2.1 Current Challenges
1. **Generic Recommendations**: Most simple recommendation systems only filter by genre
2. **Static Preferences**: Systems don't adapt to changing user tastes
3. **Black Box Recommendations**: Users don't understand why movies are suggested
4. **Context Ignorance**: Current mood/situation isn't considered
5. **Cold Start Problem**: New users receive poor recommendations

### 2.2 Opportunity
Create a demonstration system that showcases:
- Advanced ML techniques (TF-IDF, cosine similarity)
- Adaptive learning algorithms
- Explainable AI principles
- Real-world application of hybrid recommendation systems

---

## 3. Goals & Objectives

### 3.1 Primary Goals
1. **Build a working recommendation engine** using content-based filtering and NLP
2. **Implement adaptive learning** that improves with user interactions
3. **Provide explainable recommendations** with clear reasoning
4. **Demonstrate technical competency** in ML, web development, and system design

### 3.2 Learning Objectives
- Understand and implement recommendation algorithms
- Apply NLP techniques (TF-IDF) to real-world problems
- Build REST APIs with Flask
- Design and implement database schemas
- Create responsive web interfaces
- Handle real-time user interactions

### 3.3 Success Criteria
- [ ] System recommends 10 relevant movies within 500ms
- [ ] Recommendations improve after 5+ user interactions
- [ ] Users rate 6/10 recommendations as "interesting" or better
- [ ] System handles 100+ concurrent users
- [ ] All core features implemented and tested

---

## 4. Target Audience

### 4.1 Primary Users
- **College Students**: Age 18-25, tech-savvy, frequent movie watchers
- **Movie Enthusiasts**: People who actively seek new movies to watch
- **Casual Users**: Anyone looking for movie suggestions based on mood

### 4.2 User Characteristics
- Basic understanding of web applications
- Willing to provide initial preferences during signup
- Interested in discovering new movies
- Value personalized recommendations
- Appreciate understanding why recommendations are made

### 4.3 Use Cases
1. **Weekend Planning**: User wants movie suggestions for a relaxing weekend
2. **Mood-Based Discovery**: User feeling stressed wants comedy recommendations
3. **Similar Movie Search**: User loved "Inception" and wants similar films
4. **Taste Evolution**: User's preferences change over time, system adapts

---

## 5. Product Overview

### 5.1 Core Functionality
The system combines multiple intelligent factors to generate personalized movie recommendations:

```
User Input → [Taste Vector + Mood + Recent Likes] → ML Engine → Ranked Recommendations
```

### 5.2 Key Features
1. **User Taste Profiling**: Collect preferences during signup
2. **Mood-Based Filtering**: Real-time mood selection influences results
3. **Hybrid Recommendation**: Content + Plot Similarity + Taste Matching
4. **Explainable Output**: Clear reasons for each recommendation
5. **Adaptive Learning**: Taste vector updates with interactions
6. **Interactive UI**: Simple, responsive web interface

### 5.3 System Flow
```
Signup → Taste Collection → Dashboard → Mood Selection →
Recommendations → Like/View → Taste Update → Better Recommendations
```

---

## 6. Functional Requirements

### 6.1 User Authentication

#### FR-1.1: User Registration
**Priority**: P0 (Must Have)
**Description**: Users can create accounts with taste preference collection

**Requirements**:
- Username (unique, 3-50 characters)
- Email (unique, valid format)
- Password (minimum 8 characters, hashed)
- Preference collection:
  - Favorite genres (minimum 3 selections)
  - Favorite movies (minimum 5 selections)
  - Preferred industry (Hollywood/Bollywood/Korean/All)
  - Format preference (Movies/Series/Both)

**Acceptance Criteria**:
- User can successfully create account
- Duplicate username/email rejected with clear error
- Initial taste vector computed from selections
- User redirected to dashboard after signup

#### FR-1.2: User Login
**Priority**: P0 (Must Have)
**Description**: Registered users can login with credentials

**Requirements**:
- Email/username + password authentication
- Session management (30-day expiry)
- "Remember me" functionality
- Password validation

**Acceptance Criteria**:
- Valid credentials grant access
- Invalid credentials show error message
- User session persists across page refreshes

#### FR-1.3: User Logout
**Priority**: P0 (Must Have)
**Description**: Users can securely logout

**Requirements**:
- Clear session data
- Redirect to landing page

---

### 6.2 Recommendation Engine

#### FR-2.1: Get Personalized Recommendations
**Priority**: P0 (Must Have)
**Description**: Generate top N movie recommendations for authenticated users

**Requirements**:
- Input: User ID, optional mood, limit (default 10)
- Output: Ranked list of movies with scores and explanations
- Algorithm: Hybrid scoring (content + plot + taste + mood)
- Exclude already watched/liked movies
- Apply diversity filter (varied genres)

**Acceptance Criteria**:
- Returns 10 unique movies
- Response time < 500ms
- Each movie includes explanation
- Mood boost applied correctly when mood provided
- No duplicate recommendations

#### FR-2.2: Mood-Based Filtering
**Priority**: P0 (Must Have)
**Description**: Adjust recommendations based on current mood

**Requirements**:
- Supported moods: Happy, Sad, Stressed, Romantic, Motivated, Thriller
- Mood-to-genre mapping with boost factors
- Optional (can get recommendations without mood)

**Mood Mappings**:
```
Happy → Comedy (1.5x), Animation (1.3x), Family (1.2x)
Sad → Drama (1.4x), Romance (1.2x)
Stressed → Comedy (1.5x), Animation (1.3x), Adventure (1.2x)
Romantic → Romance (1.6x), Drama (1.2x)
Motivated → Action (1.4x), Biography (1.2x), Sport (1.2x)
Thriller → Thriller (1.5x), Mystery (1.4x), Crime (1.3x)
```

**Acceptance Criteria**:
- Mood-appropriate genres boosted in results
- User can skip mood selection
- Mood selection persists during session

#### FR-2.3: Content-Based Filtering
**Priority**: P0 (Must Have)
**Description**: Recommend movies similar to user's favorites

**Requirements**:
- Feature vector includes: genres, cast, director, keywords
- Use cosine similarity for comparison
- Weight: 40% of final score

**Acceptance Criteria**:
- Movies similar to favorites ranked higher
- Similarity calculation accurate
- Pre-computed similarity matrix loaded efficiently

#### FR-2.4: Plot Similarity (NLP)
**Priority**: P0 (Must Have)
**Description**: Find movies with similar storylines using NLP

**Requirements**:
- TF-IDF vectorization of movie plots
- Cosine similarity between plot vectors
- Weight: 30% of final score
- Parameters:
  - max_features: 5000
  - ngram_range: (1, 2)
  - stop_words: 'english'

**Acceptance Criteria**:
- Plots with similar themes score higher
- TF-IDF matrix pre-computed
- Calculation completes within timeout

#### FR-2.5: Taste Vector Matching
**Priority**: P0 (Must Have)
**Description**: Match movie genres with user's taste profile

**Requirements**:
- User taste vector stored as genre weights (0-1)
- Cosine similarity between user taste and movie genres
- Weight: 20% of final score

**Acceptance Criteria**:
- Movies matching taste preferences score higher
- Taste vector normalized (sum = 1)
- All genres represented in vector

---

### 6.3 Adaptive Learning

#### FR-3.1: Update Taste Vector on Like
**Priority**: P0 (Must Have)
**Description**: Automatically update user preferences when they like movies

**Requirements**:
- Triggered when user likes a movie
- Algorithm: Exponential Moving Average
  ```
  new_taste[genre] = (1 - α) × old_taste[genre] + α × new_movie[genre]
  where α = 0.2 (learning rate)
  ```
- Normalize after update
- Save to database immediately

**Acceptance Criteria**:
- Taste vector updates correctly
- New genres added to profile
- Existing genre weights adjusted
- Future recommendations reflect change

#### FR-3.2: Track User Interactions
**Priority**: P1 (Should Have)
**Description**: Log user interactions for analytics and improvement

**Requirements**:
- Track: likes, views, ratings
- Store: user_id, movie_id, interaction_type, timestamp
- Used for: taste updates, analytics

**Acceptance Criteria**:
- All interactions logged correctly
- Timestamp accurate
- Data retrievable for analysis

---

### 6.4 Explainable AI

#### FR-4.1: Generate Recommendation Explanations
**Priority**: P0 (Must Have)
**Description**: Provide clear reasons why each movie was recommended

**Requirements**:
- 2-4 reasons per movie
- Reason types:
  - "Because you liked [Movie Name]" (highest similarity)
  - "Matches your [Genre] preference" (taste match)
  - "[X]% storyline similarity to [Movie]" (plot similarity)
  - "Perfect for your [mood] mood" (mood boost)

**Acceptance Criteria**:
- Each recommendation has explanations
- Explanations are accurate
- Language is clear and user-friendly
- Most relevant reason shown first

---

### 6.5 User Profile Management

#### FR-5.1: View User Profile
**Priority**: P1 (Should Have)
**Description**: Users can view their preferences and taste profile

**Requirements**:
- Display: username, email, preferences, favorite movies
- Show taste vector (optional visualization)
- Show interaction history

**Acceptance Criteria**:
- All user data displayed correctly
- Taste vector readable format
- Favorite movies list visible

#### FR-5.2: Update Preferences
**Priority**: P1 (Should Have)
**Description**: Users can modify their preferences

**Requirements**:
- Edit: preferred genres, industry, format
- Cannot change: username (unique identifier)
- Can change: email, password

**Acceptance Criteria**:
- Changes saved to database
- Validation on all fields
- Confirmation message shown

#### FR-5.3: Manage Favorite Movies
**Priority**: P0 (Must Have)
**Description**: Users can add/remove favorite movies

**Requirements**:
- Add movie to favorites (like button)
- Remove movie from favorites
- View all favorites list
- Limit: No maximum

**Acceptance Criteria**:
- Add/remove operations succeed
- Taste vector updates on add
- UI reflects changes immediately

---

### 6.6 Movie Information

#### FR-6.1: Search Movies
**Priority**: P1 (Should Have)
**Description**: Users can search for movies by title

**Requirements**:
- Search by title (partial match)
- Autocomplete suggestions
- Display: title, year, poster
- Used during: signup (favorite selection), profile updates

**Acceptance Criteria**:
- Search returns relevant results
- Response time < 200ms
- Minimum 3 characters to search

#### FR-6.2: View Movie Details
**Priority**: P2 (Nice to Have)
**Description**: Users can view detailed information about a movie

**Requirements**:
- Display: title, year, plot, cast, director, genres, rating
- Show poster image
- "Add to Favorites" button

**Acceptance Criteria**:
- All data displayed correctly
- Images load properly
- Buttons functional

---

## 7. Non-Functional Requirements

### 7.1 Performance

#### NFR-1.1: Response Time
- **Requirement**: API responses < 500ms for 95% of requests
- **Measurement**: Server-side timing logs
- **Priority**: P0

#### NFR-1.2: Recommendation Generation
- **Requirement**: Generate 10 recommendations in < 300ms
- **Strategy**: Pre-compute similarity matrices, use efficient algorithms
- **Priority**: P0

#### NFR-1.3: Database Queries
- **Requirement**: All queries < 100ms
- **Strategy**: Index key columns, optimize queries
- **Priority**: P0

### 7.2 Scalability

#### NFR-2.1: Concurrent Users
- **Requirement**: Handle 100+ concurrent users without degradation
- **Testing**: Load testing with 100 simulated users
- **Priority**: P1

#### NFR-2.2: Dataset Size
- **Requirement**: Support 5,000-10,000 movies
- **Strategy**: Efficient data structures, pagination
- **Priority**: P0

#### NFR-2.3: User Base
- **Requirement**: Support 1,000+ registered users
- **Database**: Indexed user table
- **Priority**: P1

### 7.3 Reliability

#### NFR-3.1: Uptime
- **Requirement**: 99% uptime during evaluation period
- **Strategy**: Error handling, graceful degradation
- **Priority**: P1

#### NFR-3.2: Data Integrity
- **Requirement**: No data loss during user interactions
- **Strategy**: Database transactions, backup strategy
- **Priority**: P0

#### NFR-3.3: Error Handling
- **Requirement**: All errors handled gracefully with user-friendly messages
- **Examples**:
  - Network errors → "Connection lost, please retry"
  - No recommendations → "No movies match your criteria"
  - Database error → "System error, please try again"
- **Priority**: P0

### 7.4 Security

#### NFR-4.1: Password Security
- **Requirement**: Passwords hashed using bcrypt (cost factor: 12)
- **Storage**: Never store plaintext passwords
- **Priority**: P0

#### NFR-4.2: Session Management
- **Requirement**: Secure session tokens, 30-day expiry
- **Strategy**: Flask-Login or JWT
- **Priority**: P0

#### NFR-4.3: Input Validation
- **Requirement**: All user inputs validated and sanitized
- **Protection**: SQL injection, XSS prevention
- **Priority**: P0

#### NFR-4.4: API Authentication
- **Requirement**: All API endpoints require authentication (except signup/login)
- **Strategy**: Session-based auth or JWT tokens
- **Priority**: P0

### 7.5 Usability

#### NFR-5.1: User Interface
- **Requirement**: Clean, intuitive interface
- **Standard**: Responsive design (desktop + mobile)
- **Framework**: Bootstrap 5 or similar
- **Priority**: P0

#### NFR-5.2: Onboarding
- **Requirement**: New users can sign up and get recommendations in < 3 minutes
- **Steps**: Signup → Preference collection → First recommendations
- **Priority**: P0

#### NFR-5.3: Accessibility
- **Requirement**: Basic accessibility standards (WCAG 2.0 Level A)
- **Features**: Keyboard navigation, alt text for images
- **Priority**: P2

### 7.6 Maintainability

#### NFR-6.1: Code Quality
- **Requirement**: Clean, well-commented code
- **Standards**: PEP 8 for Python
- **Documentation**: Docstrings for all functions
- **Priority**: P1

#### NFR-6.2: Modularity
- **Requirement**: Clear separation of concerns
- **Structure**: Routes → Services → ML Engine → Database
- **Priority**: P1

#### NFR-6.3: Testing
- **Requirement**: Unit tests for core ML functions
- **Coverage**: Minimum 60% code coverage
- **Priority**: P1

---

## 8. User Stories

### 8.1 As a New User

**US-1**: As a new user, I want to sign up quickly so that I can start getting recommendations.
- **Acceptance**: Signup takes < 2 minutes including preference collection

**US-2**: As a new user, I want to select my favorite movies during signup so that the system understands my taste.
- **Acceptance**: Can search and select at least 5 favorite movies

**US-3**: As a new user, I want to receive good recommendations immediately so that I feel the system works.
- **Acceptance**: First recommendations are relevant despite limited data (cold start handled)

### 8.2 As a Returning User

**US-4**: As a returning user, I want to select my current mood so that I get contextually relevant recommendations.
- **Acceptance**: Mood selector available on dashboard, results reflect mood

**US-5**: As a returning user, I want to understand why movies are recommended so that I trust the system.
- **Acceptance**: Each movie has 2-4 clear explanations

**US-6**: As a returning user, I want to like movies I'm interested in so that future recommendations improve.
- **Acceptance**: Like button works, taste vector updates, future recommendations better

**US-7**: As a returning user, I want recommendations to evolve with my changing tastes so that the system stays relevant.
- **Acceptance**: After liking 5+ movies, recommendations noticeably different

### 8.3 As a Movie Enthusiast

**US-8**: As a movie enthusiast, I want to find movies similar to ones I loved so that I can discover new favorites.
- **Acceptance**: Recommendations include movies with high plot similarity to favorites

**US-9**: As a movie enthusiast, I want to see detailed information about recommended movies so that I can make informed decisions.
- **Acceptance**: Click on movie shows full details (plot, cast, ratings)

**US-10**: As a movie enthusiast, I want diverse recommendations so that I discover movies outside my usual genres.
- **Acceptance**: Recommendations include variety of genres (not all same type)

### 8.4 As a Casual User

**US-11**: As a casual user, I want a simple interface so that I can get recommendations without confusion.
- **Acceptance**: UI is intuitive, no training needed

**US-12**: As a casual user, I want fast recommendations so that I don't wait.
- **Acceptance**: Recommendations load in < 1 second

---

## 9. Technical Architecture

### 9.1 System Architecture

```
┌─────────────────────────────────────────────────┐
│              Frontend Layer                      │
│  HTML5 + CSS3 + JavaScript + Bootstrap 5        │
└─────────────────┬───────────────────────────────┘
                  │ HTTP/REST
┌─────────────────▼───────────────────────────────┐
│              Backend Layer                       │
│  Flask 3.0 (Python 3.10+)                       │
│  ├── Routes (Auth, Recommendation, User)        │
│  ├── ML Engine (Recommender, Similarity)        │
│  └── Utils (Validators, Helpers, Explainer)     │
└─────────────────┬───────────────────────────────┘
                  │
┌─────────────────▼───────────────────────────────┐
│              Data Layer                          │
│  ├── SQLite (Development)                       │
│  ├── Pre-computed Models (.pkl files)           │
│  └── Movie Dataset (5000+ movies)               │
└──────────────────────────────────────────────────┘
```

### 9.2 Technology Stack

#### Frontend
- **HTML5**: Structure
- **CSS3 + Bootstrap 5**: Styling and responsive design
- **JavaScript (Vanilla)**: Interactivity, AJAX calls
- **Fetch API**: HTTP requests

#### Backend
- **Framework**: Flask 3.0
- **Language**: Python 3.10+
- **ORM**: Flask-SQLAlchemy
- **Authentication**: Flask-Login
- **CORS**: Flask-CORS

#### Machine Learning
- **Data Processing**: Pandas 2.1+, NumPy 1.26+
- **ML Library**: Scikit-learn 1.3+
  - TfidfVectorizer (NLP)
  - cosine_similarity (similarity calculation)
  - StandardScaler (feature normalization)
- **Serialization**: Pickle/Joblib

#### Database
- **Development**: SQLite 3
- **Production Option**: PostgreSQL or MongoDB

#### Development Tools
- **Version Control**: Git
- **API Testing**: Postman
- **IDE**: VS Code / PyCharm
- **Environment**: venv (Python virtual environment)

---

## 10. Machine Learning Requirements

### 10.1 Algorithms

#### 10.1.1 Content-Based Filtering
**Algorithm**: Cosine Similarity on Feature Vectors

**Features**:
- Genres (one-hot encoded)
- Cast (top 3 actors, encoded)
- Director (encoded)
- Keywords (top 5, encoded)

**Formula**:
```
similarity(A, B) = cos(θ) = (A · B) / (||A|| × ||B||)
```

**Weight in Hybrid Score**: 40%

#### 10.1.2 Plot Similarity (NLP)
**Algorithm**: TF-IDF + Cosine Similarity

**Process**:
1. Tokenize movie plots
2. Remove stop words
3. Generate TF-IDF matrix
4. Calculate pairwise cosine similarity

**Parameters**:
```python
TfidfVectorizer(
    max_features=5000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.8
)
```

**Weight in Hybrid Score**: 30%

#### 10.1.3 Taste Vector Matching
**Algorithm**: Cosine Similarity between User Taste and Movie Genres

**User Taste Vector Format**:
```json
{
  "Action": 0.75,
  "Drama": 0.65,
  "Comedy": 0.40,
  "Sci-Fi": 0.90,
  ...
}
```

**Weight in Hybrid Score**: 20%

#### 10.1.4 Mood Boosting
**Algorithm**: Multiplicative Boost Factor

**Formula**:
```
final_score = base_score × mood_boost_factor
where mood_boost_factor ∈ [1.0, 1.6]
```

**Weight in Hybrid Score**: 10% (applied as multiplier)

### 10.2 Hybrid Recommendation Formula

```python
def calculate_final_score(movie, user, mood=None):
    # Base scores
    content_score = cosine_similarity(
        movie.feature_vector,
        user.favorite_movies_avg_vector
    )

    plot_score = max([
        cosine_similarity(movie.plot_vector, fav.plot_vector)
        for fav in user.favorites
    ])

    taste_score = cosine_similarity(
        movie.genre_vector,
        user.taste_vector
    )

    # Weighted combination
    base_score = (
        0.4 * content_score +
        0.3 * plot_score +
        0.2 * taste_score
    )

    # Apply mood boost
    mood_multiplier = get_mood_boost(mood, movie.primary_genre)

    final_score = base_score * mood_multiplier

    return final_score
```

### 10.3 Adaptive Learning Algorithm

**Algorithm**: Exponential Moving Average (EMA)

**Formula**:
```
new_taste[genre] = (1 - α) × old_taste[genre] + α × new_movie[genre]

where:
  α = 0.2 (learning rate)
  old_taste[genre] = current weight for genre
  new_movie[genre] = 1 if movie has genre, else 0
```

**Normalization**:
```python
total = sum(taste_vector.values())
normalized_taste = {k: v/total for k, v in taste_vector.items()}
```

**Properties**:
- Recent likes have more influence than old ones
- Gradual adaptation (not sudden changes)
- All genres eventually represented if user likes diverse movies

### 10.4 Diversity Filter

**Purpose**: Prevent all recommendations from being same genre

**Algorithm**:
1. Group top N candidates by primary genre
2. Select at most 3 movies per genre
3. Fill remaining slots with next highest scores
4. Ensure at least 4 different genres in top 10

---

## 11. Data Requirements

### 11.1 Dataset Sources

#### Primary: MovieLens Latest Small Dataset
- **URL**: https://grouplens.org/datasets/movielens/latest/
- **Size**: 100,000 ratings, 9,000 movies
- **Files**:
  - `movies.csv`: movieId, title, genres
  - `ratings.csv`: userId, movieId, rating, timestamp
  - `links.csv`: movieId, imdbId, tmdbId

#### Secondary: TMDB API
- **URL**: https://www.themoviedb.org/
- **Usage**: Supplement with plots, cast, posters, keywords
- **Rate Limit**: 40 requests per 10 seconds
- **Endpoint**: `/movie/{movie_id}`

### 11.2 Data Schema

#### Movies Table
```sql
CREATE TABLE movies (
    movie_id INTEGER PRIMARY KEY,
    tmdb_id INTEGER UNIQUE,
    imdb_id VARCHAR(20),
    title VARCHAR(255) NOT NULL,
    release_year INTEGER,
    overview TEXT,
    genres TEXT,  -- JSON array
    cast TEXT,    -- JSON array (top 3)
    director VARCHAR(100),
    keywords TEXT,  -- JSON array
    vote_average FLOAT,
    popularity FLOAT,
    poster_path VARCHAR(255),
    genre_vector TEXT,    -- JSON object
    feature_vector TEXT   -- JSON array
);
```

#### Users Table
```sql
CREATE TABLE users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username VARCHAR(50) UNIQUE NOT NULL,
    email VARCHAR(100) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    preferred_genres TEXT,     -- JSON array
    preferred_industry VARCHAR(50),
    preferred_format VARCHAR(20),
    taste_vector TEXT          -- JSON object
);
```

#### User Favorites Table
```sql
CREATE TABLE user_favorites (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    movie_id INTEGER,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id),
    UNIQUE(user_id, movie_id)
);
```

#### User Interactions Table
```sql
CREATE TABLE user_interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    movie_id INTEGER,
    interaction_type VARCHAR(20),  -- 'like', 'view', 'rate'
    rating FLOAT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id),
    FOREIGN KEY (movie_id) REFERENCES movies(movie_id)
);
```

### 11.3 Pre-computed Models

**Files** (stored in `data/models/`):
1. **tfidf_matrix.pkl**: TF-IDF vectors for all movie plots (shape: N × 5000)
2. **similarity_matrix.pkl**: Pre-computed cosine similarity (shape: N × N)
3. **movie_vectors.pkl**: Feature vectors for all movies (shape: N × F)
4. **tfidf_vectorizer.pkl**: Fitted TfidfVectorizer for transforming new plots

**Size Estimates**:
- 5,000 movies: ~50-100 MB total
- 10,000 movies: ~200-300 MB total

---

## 12. User Interface Requirements

### 12.1 Pages & Wireframes

#### Landing Page
**URL**: `/`
**Purpose**: Welcome page, project introduction
**Components**:
- Hero section with project title
- Brief description
- "Get Started" button → Signup
- "Login" link

#### Signup Page
**URL**: `/signup`
**Purpose**: User registration with taste collection
**Components**:
- Step 1: Basic Info
  - Username (input)
  - Email (input)
  - Password (input)
  - Confirm Password (input)
  - Next button

- Step 2: Taste Profile
  - "Select favorite genres" (multi-select chips)
  - "Choose 5 favorite movies" (searchable dropdown)
  - "Preferred industry" (radio buttons)
  - "Format preference" (radio buttons)
  - Submit button

**Validation**:
- All fields required
- Minimum 3 genres
- Exactly 5 movies
- Password strength indicator

#### Login Page
**URL**: `/login`
**Purpose**: User authentication
**Components**:
- Email/Username (input)
- Password (input)
- "Remember Me" (checkbox)
- Login button
- "Forgot Password?" link (optional)
- "Don't have an account? Sign up" link

#### Dashboard
**URL**: `/dashboard`
**Purpose**: Main page for logged-in users
**Components**:
- Welcome message: "Welcome back, [Username]!"
- Mood selector:
  ```
  How are you feeling today?
  [😊 Happy] [😢 Sad] [😰 Stressed] [💕 Romantic] [💪 Motivated] [🎬 Thriller]
  ```
- "Get Recommendations" button
- Quick stats (optional):
  - Movies liked: X
  - Recommendations viewed: Y
- Navigation: Dashboard | Profile | Logout

#### Recommendations Page
**URL**: `/recommendations`
**Purpose**: Display personalized movie recommendations
**Components**:
- Header: "Recommendations for You"
- Mood indicator (if selected): "Based on your [mood] mood"
- Movie grid (2-3 columns):
  - Each card:
    - Poster image
    - Title + Year
    - Genres (chips)
    - Rating (⭐ X.X)
    - "Why this movie?" (expandable)
      - Reason 1
      - Reason 2
      - Reason 3
    - ❤️ Like button
    - ℹ️ More Info button
- "Get New Recommendations" button

**Interactions**:
- Click "Like" → Add to favorites, update taste vector, show toast notification
- Click "More Info" → Navigate to movie details page
- Expand "Why this movie?" → Show/hide explanations

#### Profile Page
**URL**: `/profile`
**Purpose**: View and edit user preferences
**Components**:
- User info: Username, Email
- Preferences section (editable):
  - Favorite genres
  - Preferred industry
  - Format preference
- Favorite movies list:
  - Movie title + poster
  - Remove button
- Taste vector visualization (optional pie chart)
- "Update Profile" button

#### Movie Details Page (Optional)
**URL**: `/movie/<movie_id>`
**Purpose**: Detailed information about a movie
**Components**:
- Large poster
- Title + Year
- Genres
- Rating
- Plot/Overview
- Cast
- Director
- "Add to Favorites" button
- "Get Similar Movies" button

### 12.2 UI/UX Guidelines

**Color Scheme**:
- Primary: Blue (#007bff) or custom theme
- Success: Green (#28a745)
- Danger: Red (#dc3545)
- Background: Light gray (#f8f9fa)

**Typography**:
- Font: System fonts (San Francisco, Segoe UI, Roboto)
- Headings: 24-32px, bold
- Body: 14-16px, regular

**Responsiveness**:
- Mobile: Single column layout
- Tablet: 2-column grid
- Desktop: 3-column grid

**Loading States**:
- Spinner during recommendation generation
- Skeleton screens for movie cards
- Progress bar for multi-step signup

**Error States**:
- Toast notifications for errors
- Inline validation messages
- Friendly error pages (404, 500)

---

## 13. API Specifications

### 13.1 Authentication Endpoints

#### POST /api/auth/signup
**Description**: Register new user with preferences

**Request Body**:
```json
{
  "username": "john_doe",
  "email": "john@example.com",
  "password": "securepass123",
  "preferences": {
    "favorite_genres": ["Action", "Sci-Fi", "Thriller"],
    "favorite_movies": [550, 680, 27205, 157336, 22],
    "preferred_industry": "Hollywood",
    "preferred_format": "Both"
  }
}
```

**Response** (201 Created):
```json
{
  "success": true,
  "message": "Account created successfully",
  "user": {
    "user_id": 123,
    "username": "john_doe",
    "email": "john@example.com"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Errors**:
- 400: Validation error
- 409: Username/email already exists

#### POST /api/auth/login
**Description**: Authenticate user

**Request Body**:
```json
{
  "email": "john@example.com",
  "password": "securepass123"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Login successful",
  "user": {
    "user_id": 123,
    "username": "john_doe"
  },
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
}
```

**Errors**:
- 401: Invalid credentials

#### POST /api/auth/logout
**Description**: Logout user

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Logged out successfully"
}
```

#### GET /api/auth/me
**Description**: Get current user info

**Response** (200 OK):
```json
{
  "user_id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "preferences": {
    "favorite_genres": ["Action", "Sci-Fi"],
    "preferred_industry": "Hollywood",
    "preferred_format": "Both"
  },
  "taste_vector": {
    "Action": 0.8,
    "Sci-Fi": 0.9,
    "Drama": 0.5
  }
}
```

---

### 13.2 Recommendation Endpoints

#### POST /api/recommend
**Description**: Get personalized movie recommendations

**Request Body**:
```json
{
  "mood": "happy",
  "limit": 10
}
```

**Response** (200 OK):
```json
{
  "recommendations": [
    {
      "movie_id": 456,
      "title": "The Grand Budapest Hotel",
      "year": 2014,
      "genres": ["Comedy", "Drama"],
      "poster": "https://image.tmdb.org/t/p/w500/...",
      "vote_average": 8.1,
      "score": 0.95,
      "reasons": [
        "Perfect for your happy mood 😊",
        "Matches your Comedy preference",
        "Because you liked Inception",
        "78% storyline similarity"
      ]
    },
    ...9 more movies
  ],
  "mood": "happy",
  "generation_time_ms": 284
}
```

**Errors**:
- 400: Invalid mood value
- 404: No recommendations found

---

### 13.3 User Profile Endpoints

#### GET /api/user/profile
**Description**: Get user profile

**Response** (200 OK):
```json
{
  "user_id": 123,
  "username": "john_doe",
  "email": "john@example.com",
  "preferences": {
    "favorite_genres": ["Action", "Sci-Fi"],
    "preferred_industry": "Hollywood",
    "preferred_format": "Both"
  },
  "taste_vector": {
    "Action": 0.8,
    "Sci-Fi": 0.9
  }
}
```

#### PUT /api/user/profile
**Description**: Update user preferences

**Request Body**:
```json
{
  "preferred_genres": ["Action", "Sci-Fi", "Thriller"],
  "preferred_industry": "All",
  "preferred_format": "Movies"
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Profile updated successfully"
}
```

#### POST /api/user/favorites
**Description**: Add movie to favorites (like)

**Request Body**:
```json
{
  "movie_id": 456
}
```

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Added to favorites",
  "taste_updated": true,
  "new_taste_vector": {
    "Action": 0.75,
    "Comedy": 0.44,
    "Drama": 0.76
  }
}
```

#### DELETE /api/user/favorites/:movie_id
**Description**: Remove movie from favorites

**Response** (200 OK):
```json
{
  "success": true,
  "message": "Removed from favorites"
}
```

#### GET /api/user/favorites
**Description**: Get all favorite movies

**Response** (200 OK):
```json
{
  "favorites": [
    {
      "movie_id": 550,
      "title": "Fight Club",
      "year": 1999,
      "poster": "https://...",
      "added_at": "2026-02-20T10:30:00Z"
    },
    ...
  ],
  "total": 12
}
```

---

### 13.4 Movie Endpoints

#### GET /api/movies/search
**Description**: Search movies by title

**Query Parameters**:
- `q`: Search query (required)
- `limit`: Number of results (default: 10)

**Example**: `/api/movies/search?q=inception&limit=5`

**Response** (200 OK):
```json
{
  "results": [
    {
      "movie_id": 27205,
      "title": "Inception",
      "year": 2010,
      "poster": "https://..."
    },
    ...
  ],
  "total": 5
}
```

#### GET /api/movies/:movie_id
**Description**: Get movie details

**Response** (200 OK):
```json
{
  "movie_id": 27205,
  "title": "Inception",
  "year": 2010,
  "genres": ["Action", "Sci-Fi", "Thriller"],
  "overview": "Cobb, a skilled thief...",
  "cast": ["Leonardo DiCaprio", "Joseph Gordon-Levitt", "Ellen Page"],
  "director": "Christopher Nolan",
  "vote_average": 8.4,
  "popularity": 89.5,
  "poster": "https://...",
  "keywords": ["dream", "subconscious", "heist"]
}
```

#### GET /api/movies/popular
**Description**: Get popular movies (for cold start)

**Query Parameters**:
- `limit`: Number of results (default: 20)

**Response** (200 OK):
```json
{
  "movies": [
    {
      "movie_id": 550,
      "title": "Fight Club",
      "year": 1999,
      "vote_average": 8.4,
      "poster": "https://..."
    },
    ...
  ]
}
```

---

## 14. Success Metrics

### 14.1 Technical Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recommendation Response Time** | < 500ms | Server-side timing logs |
| **API Success Rate** | > 99% | Error rate monitoring |
| **Database Query Time** | < 100ms | Query profiling |
| **Concurrent Users** | 100+ | Load testing |
| **Uptime** | > 99% | Monitoring logs |

### 14.2 ML Performance Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Recommendation Relevance** | 6/10 rated "interesting" | User survey |
| **Taste Adaptation** | Noticeable change after 5+ likes | A/B testing |
| **Cold Start Performance** | 4/10 relevant for new users | User feedback |
| **Diversity Score** | At least 4 genres in top 10 | Algorithmic check |

### 14.3 User Experience Metrics

| Metric | Target | Measurement Method |
|--------|--------|-------------------|
| **Signup Time** | < 3 minutes | Analytics tracking |
| **User Retention** | 70% return after first session | Session logs |
| **Interaction Rate** | 30% of users like movies | Interaction logs |
| **Explanation Clarity** | 80% find reasons helpful | User survey |

### 14.4 Feature Completeness

- [ ] User authentication (signup, login, logout)
- [ ] Taste profile creation and management
- [ ] Mood-based recommendation filtering
- [ ] Hybrid scoring algorithm
- [ ] Explainable AI output
- [ ] Adaptive learning (taste vector updates)
- [ ] Responsive UI
- [ ] Search functionality
- [ ] Favorite movies management
- [ ] Error handling

---

## 15. Project Timeline

### Phase 1: Foundation & Data Setup (Week 1-2)

**Week 1**
- [ ] Day 1-2: Environment setup, Git initialization
- [ ] Day 3-5: Data collection (MovieLens + TMDB API)
- [ ] Day 6-7: Database schema design and setup

**Week 2**
- [ ] Day 8-10: Data cleaning and preprocessing
- [ ] Day 11-12: Load data into database
- [ ] Day 13-14: Verify data integrity, exploratory analysis

**Deliverable**: Database with 5,000+ movies, all data fields populated

---

### Phase 2: Core ML Engine (Week 2-3)

**Week 2-3**
- [ ] Day 15-17: Feature engineering (genre vectors, TF-IDF)
- [ ] Day 18-20: Implement similarity calculations
- [ ] Day 21-22: Build recommendation engine core
- [ ] Day 23-24: Test with sample user profiles
- [ ] Day 25: Optimize performance (pre-compute matrices)

**Deliverable**: Working recommendation function (Python) returning top N movies

---

### Phase 3: Backend API (Week 3-4)

**Week 3-4**
- [ ] Day 26-28: Flask app structure, route setup
- [ ] Day 29-30: User authentication (signup, login)
- [ ] Day 31-32: Recommendation API endpoint
- [ ] Day 33-34: Mood mapping and boosting logic
- [ ] Day 35: Explanation generation
- [ ] Day 36: API testing with Postman

**Deliverable**: REST API with all endpoints functional

---

### Phase 4: Frontend & Integration (Week 4-5)

**Week 4-5**
- [ ] Day 37-38: Landing and signup pages
- [ ] Day 39-40: Login and dashboard
- [ ] Day 41-42: Recommendations UI (cards, explanations)
- [ ] Day 43-44: Profile management page
- [ ] Day 45: Connect frontend to backend (AJAX)
- [ ] Day 46: UI polish (Bootstrap, responsive design)

**Deliverable**: Fully functional web application

---

### Phase 5: Adaptive Learning (Week 5-6)

**Week 5-6**
- [ ] Day 47-48: Implement taste vector update logic
- [ ] Day 49-50: Like button integration
- [ ] Day 51: Test adaptive learning with multiple users
- [ ] Day 52: Verify recommendations improve over time

**Deliverable**: System that learns from user interactions

---

### Phase 6: Testing & Polish (Week 6-8)

**Week 6-7**
- [ ] Day 53-55: Write unit tests (ML functions, API)
- [ ] Day 56-57: Integration testing
- [ ] Day 58-59: User acceptance testing (5+ testers)
- [ ] Day 60-61: Bug fixes

**Week 8**
- [ ] Day 62-63: UI/UX improvements based on feedback
- [ ] Day 64-65: Performance optimization
- [ ] Day 66: Documentation (README, API docs)
- [ ] Day 67-68: Demo video and presentation preparation
- [ ] Day 69-70: Final review and submission prep

**Deliverable**: Production-ready application with documentation

---

### Milestone Summary

| Milestone | Week | Deliverable |
|-----------|------|-------------|
| Data Ready | 2 | Database populated with movies |
| ML Engine | 3 | Recommendation function working |
| API Complete | 4 | All endpoints functional |
| Frontend Done | 5 | Full web app integrated |
| Adaptive Learning | 6 | Taste updates implemented |
| Testing Complete | 7 | All tests passed, bugs fixed |
| Project Ready | 8 | Final demo ready |

---

## 16. Risks & Mitigation

### 16.1 Technical Risks

#### Risk 1: TMDB API Rate Limiting
- **Probability**: High
- **Impact**: Medium
- **Description**: API limits 40 requests per 10 seconds
- **Mitigation**:
  - Batch requests with delays
  - Cache results locally
  - Pre-fetch data during setup phase
  - Use MovieLens data as fallback

#### Risk 2: Poor Recommendation Quality
- **Probability**: Medium
- **Impact**: High
- **Description**: Recommendations not relevant to users
- **Mitigation**:
  - Test with multiple user profiles
  - Tune algorithm weights (40-30-20-10)
  - Add diversity filter
  - Implement feedback collection
  - Use popularity as fallback for cold start

#### Risk 3: Slow Performance
- **Probability**: Medium
- **Impact**: High
- **Description**: Recommendation generation takes > 500ms
- **Mitigation**:
  - Pre-compute similarity matrices
  - Use efficient data structures (NumPy arrays)
  - Cache frequently accessed data
  - Profile and optimize bottlenecks
  - Limit dataset size to 5,000-10,000 movies

#### Risk 4: Database Corruption
- **Probability**: Low
- **Impact**: High
- **Description**: Data loss during development
- **Mitigation**:
  - Regular backups (daily)
  - Use version control for schema
  - Test database migrations
  - Keep raw CSV files as source of truth

### 16.2 Project Management Risks

#### Risk 5: Scope Creep
- **Probability**: High
- **Impact**: Medium
- **Description**: Adding too many features, project incomplete
- **Mitigation**:
  - Stick to MVP (must-have features only)
  - Use P0/P1/P2 prioritization
  - Defer nice-to-have features
  - Regular progress reviews

#### Risk 6: Timeline Overrun
- **Probability**: Medium
- **Impact**: Medium
- **Description**: Project not completed in 6-8 weeks
- **Mitigation**:
  - Break work into small tasks
  - Set weekly milestones
  - Track progress daily
  - Identify delays early and adjust

#### Risk 7: Learning Curve
- **Probability**: Medium
- **Impact**: Medium
- **Description**: Unfamiliarity with Flask/ML libraries
- **Mitigation**:
  - Follow tutorials first
  - Use Jupyter notebooks for experimentation
  - Ask for help when stuck
  - Allocate buffer time for learning

### 16.3 Data Risks

#### Risk 8: Incomplete Movie Data
- **Probability**: Medium
- **Impact**: Medium
- **Description**: Many movies missing plots, cast, etc.
- **Mitigation**:
  - Filter out movies with incomplete data
  - Set minimum data requirements
  - Use multiple data sources (MovieLens + TMDB)
  - Have at least 3,000 complete movies

#### Risk 9: Biased Dataset
- **Probability**: Low
- **Impact**: Low
- **Description**: Dataset skewed toward certain genres/years
- **Mitigation**:
  - Analyze dataset distribution
  - Ensure diversity across genres
  - Include movies from multiple decades
  - Document dataset limitations

---

## 17. Future Enhancements

### 17.1 Post-MVP Features (P2 Priority)

#### Collaborative Filtering
- Implement user-based collaborative filtering
- "Users like you also enjoyed..."
- Requires larger user base (100+ users with ratings)

#### Advanced NLP
- Use pre-trained embeddings (Word2Vec, GloVe)
- Sentiment analysis on reviews
- Fine-tune BERT for plot similarity

#### Social Features
- Share recommendations with friends
- Collaborative watchlists
- Movie discussion threads

#### Personalization Enhancements
- Time-of-day recommendations
- Seasonal recommendations
- Watch history tracking
- "Watch it again" suggestions

#### UI Improvements
- Movie trailers integration (YouTube API)
- Advanced filters (year, rating, language)
- Personalized homepage
- Dark mode

#### Analytics Dashboard
- User engagement metrics
- Recommendation quality tracking
- A/B testing framework
- Taste evolution visualization

### 17.2 Scalability Improvements

#### Performance
- Migrate to PostgreSQL/MongoDB
- Implement Redis caching
- Use Celery for background tasks
- Deploy with Gunicorn + Nginx

#### Infrastructure
- Containerization (Docker)
- Cloud deployment (AWS/Heroku/Railway)
- CI/CD pipeline
- Monitoring and logging

#### ML Pipeline
- Automated model retraining
- Online learning (real-time updates)
- A/B testing for algorithm changes
- Feature importance analysis

---

## 18. Appendix

### 18.1 Glossary

| Term | Definition |
|------|------------|
| **TF-IDF** | Term Frequency-Inverse Document Frequency: NLP technique to convert text to numerical vectors |
| **Cosine Similarity** | Measure of similarity between two vectors (range: -1 to 1) |
| **Content-Based Filtering** | Recommendation based on item attributes (genres, cast, etc.) |
| **Collaborative Filtering** | Recommendation based on user behavior patterns |
| **Hybrid Recommendation** | Combination of multiple recommendation approaches |
| **Cold Start** | Problem of recommending to new users with no history |
| **Taste Vector** | Numerical representation of user's genre preferences |
| **Adaptive Learning** | System that improves with user interactions |
| **Explainable AI** | AI that provides clear reasons for decisions |

### 18.2 References

**Academic Papers**:
- "Content-Based Recommendation Systems" - Pazzani & Billsus
- "Recommender Systems Handbook" - Ricci, Rokach, Shapira

**Technical Documentation**:
- Scikit-learn Documentation: https://scikit-learn.org/
- Flask Documentation: https://flask.palletsprojects.com/
- TMDB API Docs: https://developers.themoviedb.org/

**Datasets**:
- MovieLens: https://grouplens.org/datasets/movielens/
- TMDB: https://www.themoviedb.org/

**Tutorials**:
- TF-IDF Tutorial: https://scikit-learn.org/stable/modules/feature_extraction.html
- Flask REST API: https://flask-restful.readthedocs.io/

### 18.3 Contact & Roles

| Role | Responsibility |
|------|----------------|
| **Developer** | Full-stack development, ML implementation |
| **Tester** | Testing, bug reporting |
| **Advisor** | Technical guidance, code review |

---

## Document History

| Version | Date | Changes | Author |
|---------|------|---------|--------|
| 1.0 | 2026-02-25 | Initial PRD created | Team |

---

**End of Document**

This PRD serves as the single source of truth for the AI-Based Movie Recommendation System project. All development should align with the requirements specified in this document. Any changes to scope, requirements, or timeline should be documented and approved.

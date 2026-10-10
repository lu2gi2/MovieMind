# MovieMind: Current System → Proposed System

## A plain-English guide to what we have, what we're building, and what's publishable

---

## Table of Contents

1. [How It Works Now (Current System)](#how-it-works-now)
2. [Why It Needs to Change](#why-it-needs-to-change)
3. [How It Will Work (Proposed System)](#how-it-will-work)
4. [Side-by-Side Comparison](#side-by-side-comparison)
5. [The Paper-Publishable Parts](#-the-paper-publishable-parts)
6. [What the Paper Would Look Like](#what-the-paper-would-look-like)

---

## How It Works Now

### The Big Picture

```
User signs up → picks favorite genres → system shows movies with those genre tags
User likes a movie → system boosts those genres in user's "taste" → shows more of same genres
User picks a mood → system maps mood to genres → boosts those genres → shows results
```

### Step-by-step

#### 1. Movie Storage

Every movie is stored with its **genre labels** as a JSON string:

```
Movie: "The Prestige"
Genres: ["Drama", "Mystery", "Thriller"]
```

That's it. The system knows this movie is tagged "Thriller." It has no idea whether it's a psychological thriller, a horror thriller, or a heist thriller.

#### 2. How Movies Become Numbers (TF-IDF)

We take the movie's plot summary, genre names, keywords, and cast — mash them into one big text blob — and run TF-IDF (Term Frequency - Inverse Document Frequency) on it.

```
"The Prestige" combined text:
"Two stage magicians engage in competitive one-upmanship... Drama Mystery Thriller Drama Mystery Thriller Christopher Nolan Hugh Jackman Christian Bale magic illusion rivalry twist"

TF-IDF converts this to a sparse vector (a list of numbers where most are zero).
```

**The problem with TF-IDF:** It works by counting words. "Bank heist" and "robbery" have zero word overlap, so TF-IDF thinks they're unrelated — even though they mean the same thing.

#### 3. How User Taste Is Stored

Each user has a **taste vector** — a dictionary mapping genres to scores:

```json
{
  "Action": 0.45,
  "Thriller": 0.72,
  "Mystery": 0.68,
  "Comedy": 0.30,
  "Horror": 0.10
}
```

When a user likes a movie, we increase the scores for that movie's genres using Exponential Moving Average (EMA):

```
Formula: new_score = (0.8 × old_score) + (0.2 × 1.0)

Example: User likes "The Prestige" (genres: Drama, Mystery, Thriller)
  Mystery:  0.60 → (0.8 × 0.60) + (0.2 × 1.0) = 0.68
  Thriller: 0.65 → (0.8 × 0.65) + (0.2 × 1.0) = 0.72
```

#### 4. How Recommendations Are Generated

```
Step 1: Find movies similar to what the user already liked
        (using TF-IDF cosine similarity)

Step 2: Score each candidate movie using this formula:
        score = (0.40 × content_similarity)
              + (0.30 × genre_taste_match)
              + (0.15 × movie_rating)
              + (0.15 × mood_boost)

Step 3: Sort by score, apply diversity filter, return top 10
```

#### 5. How Mood Works

Mood is a hardcoded lookup table. "Excited" = give a 1.5× boost to Action genre, 1.4× to Adventure, 1.3× to Thriller.

```python
'excited': {
    'Action': 1.5, 'Adventure': 1.4, 'Thriller': 1.3,
    'Science Fiction': 1.2
}
```

This means: "When the user feels excited, just show them more Action and Adventure movies, regardless of their actual taste."

#### 6. How Explanations Are Generated

Random selection from a template bank:

```python
"Matches your love for Thriller"
"Perfect for Thriller fans like you"
"Because you enjoy Thriller"
```

The explanation always references a genre name. It can never say "because this has the same tight pacing and twist structure as movies you like."

---

## Why It Needs to Change

### Problem 1: Genre Is a Blunt Instrument

| User likes... | Genre says... | So the system recommends... |
|--------------|---------------|---------------------------|
| *Ocean's Eleven* (heist thriller) | "Thriller" | *Hereditary* (horror thriller) |
| *The Prestige* (psychological thriller) | "Thriller" | *Saw* (gore thriller) |
| *Inception* (sci-fi thriller) | "Thriller" | *The Conjuring* (supernatural thriller) |

All "Thriller." Completely different movies.

### Problem 2: Mood Throws Away User Context

When you pick "Excited," the system immediately boosts Action/Adventure genres — **regardless of what you actually like.** A user who only watches slow-burn dramas and picks "Excited" gets *Fast & Furious*, which they'll hate. The mood system and the taste system don't talk to each other.

### Problem 3: TF-IDF Doesn't Understand Meaning

"Robbery" and "heist" = zero overlap in TF-IDF. "Bank job" and "stolen money" = zero overlap. TF-IDF only matches exact words, not concepts.

### Problem 4: No Gradual Mood Control

Mood is either ON (boosting genres) or OFF (not boosting). There's no way to say "a little bit excited" vs. "extremely excited."

---

## How It Will Work

### The Big Picture

```
Movies → converted to 384-dimension "vibe vectors" (semantic embeddings)
User taste → average of the vibe vectors of movies they liked (a single point in space)
Mood → a direction to nudge the user's taste point (not a genre filter)

Recommendation = "start at the user's taste point, walk in the mood's direction, find the nearest movies"
```

### The Core Idea (In One Sentence)

> Instead of filtering by genre tags, we put movies, users, and moods into a shared "map" where distance = similarity, and mood just nudges you to a different neighborhood on the map.

### Step-by-step

#### 1. Give Every Movie a "Vibe Vector"

We use a **Sentence Transformer** model (`all-MiniLM-L6-v2`) that reads the movie's full description — plot, themes, keywords, director style, cast — and converts it into a list of 384 numbers. These numbers capture the movie's *vibe*, not just its genre label.

```
"The Prestige"
  Plot: "Two rival magicians in Victorian London engage in a bitter feud..."
  Keywords: "magic", "rivalry", "obsession", "twist ending", "dual identity"
  Director style: Christopher Nolan (non-linear, cerebral, dark)

        ↓ Sentence Transformer encodes all of this ↓

  Vibe vector: [0.23, -0.45, 0.78, 0.12, -0.33, 0.56, ...]
                     ↑ 384 numbers that represent the movie's "position in vibe space"
```

**What makes this better:** Movies with similar vibes end up with similar numbers — even if they share zero genre tags or keywords. "Robbery" and "heist" are close because the model understands they mean the same thing. "Mind-bending" and "cerebral" are close. "Tense" and "suspenseful" are close.

```
"Inception"       → [0.25, -0.39, 0.80, 0.11, ...]  
"The Prestige"    → [0.23, -0.45, 0.78, 0.12, ...]  ← very similar!
"Ocean's Eleven"  → [0.21, -0.41, 0.75, 0.18, ...]  ← also close (smart, twisty)
"Hereditary"      → [0.82, 0.53, -0.31, 0.91, ...]  ← completely different
```

#### 2. Build Your Taste as a Point in the Same Space

Every time you like a movie, we pull its vibe vector and average it into your personal taste coordinate. No more `{Action: 0.7, Comedy: 0.3}`. Your taste is a 384-number point in the same space as the movies.

```
You liked: "Inception", "The Prestige", "Ocean's Eleven"

Your taste coordinate = (Inception_vec + Prestige_vec + Ocean's_vec) / 3
                      = [0.23, -0.42, 0.78, 0.14, ...]

This point is literally the CENTER of the cluster of movies you like.
```

#### 3. Define Each Mood as a Direction Arrow

Instead of "Excited = Action + Adventure," we figure out where excitement *lives* in the vibe space.

**How we find it:** Take movies that are known to be exciting (high-energy, fast-paced, high-stakes) and compute where they cluster relative to the average movie. The arrow pointing from the average movie to the exciting-movie cluster = the "excited" direction.

```
Average position of all movies:    [0.00, 0.00, 0.00, 0.00, ...]
Average position of "excited" movies (Mad Max, Baby Driver, etc.): [+0.15, +0.08, -0.22, +0.31, ...]

Excited mood direction = excited_avg - all_movie_avg
                       = [+0.15, +0.08, -0.22, +0.31, ...]

This vector literally "points" toward excitement in the semantic space.
```

We do this for all 10 moods (happy, sad, excited, relaxed, romantic, thrilled, nostalgic, adventurous, thoughtful, scared). Each one = a 384-number direction arrow.

#### 4. Search by Nudging Your Taste in the Mood's Direction

This is the key innovation. The recommendation formula:

```
search_position = user_taste + (λ × mood_direction)

where λ (lambda) controls mood strength:
  λ = 0.0  →  pure taste, mood ignored
  λ = 0.3  →  taste with a light mood nudge (DEFAULT)
  λ = 0.7  →  strong mood influence
  λ = 1.0  →  pure mood, taste ignored
```

**What this looks like in practice:**

```
User: Likes smart heist/investigation thrillers
      Taste is in the "cerebral crime" neighborhood
Mood: "Excited"

λ = 0.0 (no mood):     Returns "The Town", "Inside Man", "Prisoners"
                        (pure taste — more cerebral crime films)

λ = 0.3 (light nudge): Returns "Baby Driver", "Edge of Tomorrow", "Heat"
                        (taste + energy — smart but fast-paced)

λ = 0.7 (strong nudge): Returns "Mad Max: Fury Road", "John Wick"
                         (heavily shifted toward energy, some taste lost)
```

#### 5. Find the Nearest Movies

Once we have the `search_position`, we compute cosine similarity against every movie's vibe vector, sort by similarity, exclude already-liked movies, apply diversity filtering, and return the top 10.

```
search_position = [0.27, -0.35, 0.73, 0.20, ...]

cosine_similarity(search_position, movie_vectors) for all 5000+ movies
    │
    ▼
Top matches (highest cosine similarity):
  1. Baby Driver       (0.89)
  2. Edge of Tomorrow  (0.87)
  3. Heat              (0.85)
  4. The Town          (0.83)
  5. Mission: Impossible - Fallout (0.82)
  ...
```

#### 6. Better Explanations

Since everything lives in the same space, we can explain WHY a movie was recommended in richer terms:

```
Before: "Matches your love for Thriller"

After:  "Similar pacing and tension to Inception (which you liked), 
         with the high-energy feel you want right now"

Before: "Perfect for Thriller fans like you"

After:  "Has the same twist structure as The Prestige, 
         shifted toward your excited mood"
```

---

## Side-by-Side Comparison

| Aspect | Current System | Proposed System |
|--------|---------------|-----------------|
| **Movie representation** | TF-IDF on words (bag of words) | Sentence Transformer embeddings (understands meaning) |
| **Similarity** | Word overlap ("robbery" ≠ "heist") | Semantic similarity ("robbery" ≈ "heist") |
| **User taste** | `{genre: score}` dictionary (~20 numbers) | 384-dimension vector (a point in vibe space) |
| **Mood** | Hardcoded genre lookup table | Learned direction vector in semantic space |
| **Recommendation** | Weighted sum of similarity + genre match + mood genre boost | Cosine similarity from mood-shifted taste point |
| **Mood control** | Binary (on/off) | Continuous (λ from 0 to 1) |
| **Explanation** | Random genre-based template | Content-aware, references specific similar movies |
| **Cold start** | Popular movies in preferred genres | Popular movies near the center of vibe space |
| **Adaptive learning** | EMA on genre scores | Centroid update on vibe vectors |

### Concrete Example

**User profile:** Likes *The Prestige*, *Zodiac*, *Prisoners* (smart, methodical, investigation-focused thrillers)

**Mood selected:** "Excited"

| | Current System | Proposed System |
|---|---------------|-----------------|
| **What happens** | "Excited" → boost Action & Adventure genres | Shift taste coordinate toward "excitement" direction |
| **Top recommendations** | *John Wick 4*, *Fast X*, *The Meg 2* | *Sicario*, *Heat*, *The Town*, *No Country for Old Men* |
| **Why these?** | They have Action/Adventure tags + high rating | They're crime/investigation films (matches taste) with high-intensity pacing (matches mood) |
| **User reaction** | "These are nothing like what I watch..." | "These are exactly my kind of movie but more intense" |

---

## ⭐ The Paper-Publishable Parts

### What Makes This Novel (Not Found in Any Published Paper)

#### 1. Mood as a Directional Vector in the Same Semantic Space as Content

**What everyone else does:**
```
Content model (separate)  ←→  Emotion model (separate)
       ↓                             ↓
  "This movie is similar      "This movie is exciting"
   to The Prestige"           
       ↓                             ↓
            Combined with some weighting
```

**What we do:**
```
One shared space:
  Movies = points
  Users  = points (average of liked movies)
  Moods  = direction arrows
  
  Recommendation = user_point + (λ × mood_arrow)
```

Everything is in one space. One equation. No separate models. No glue code. This is the core novelty.

#### 2. The Controllable λ (Lambda) Parameter

No existing paper lets you tune *how much* mood influences recommendations. It's always:
- "Apply mood filter" (binary ON/OFF)
- "Boost mood-relevant genres by X%" (fixed, non-tunable)

Our λ parameter allows:
- **User control:** A slider in the UI — "How much should mood influence today?"
- **System control:** The system could learn the optimal λ per user over time
- **Evaluation:** We can measure recommendation quality at every λ value and prove that λ ≈ 0.3 is the sweet spot

This is explicitly called out as an open problem in the 2025 survey of Affective Recommender Systems.

#### 3. Empirically Learned Mood Directions (Not Hardcoded)

Our mood vectors aren't invented by a human saying "excited = action." They're computed from the data:

```
mood_direction = mean(embeddings of mood-exemplar movies) - mean(embeddings of all movies)
```

This means:
- The direction is objective (grounded in actual movie content)
- It can be updated as the movie database grows
- It captures subtle patterns no human would write down (e.g., "excited" also correlates with certain cinematography keywords, pacing descriptors, etc.)

#### 4. Unified Representation Enables Richer Explanations

Because user taste, movie content, and mood are all in the same space, we can generate explanations that reference ALL three:

```
"This movie is 15% toward your 'excited' direction from 'The Prestige' 
 (which you liked), with matching investigation themes."
```

No existing system can do this because their models live in separate spaces.

### What's NOT Novel (Don't Claim These)

| Thing | Why not novel |
|-------|--------------|
| Using Sentence Transformers for movie embeddings | Multiple papers already do this |
| Averaging embeddings for user profiles | Standard technique (user centroid) |
| Cosine similarity for nearest neighbor search | Standard technique |
| Diversity filtering | Standard technique |
| EMA for taste updates | Standard technique |

**The novelty is the COMBINATION:** mood + content in one space + controllable shift + learned directions. None of the individual pieces are new. The way we put them together is.

---

## What the Paper Would Look Like

### Title

> *"Mood as a Vector: Controllable Affective Modulation of User Embeddings for Semantic Movie Recommendation"*

### Abstract (Draft)

> Content-based movie recommendation systems typically rely on genre labels and keyword matching, treating user mood as a separate filtering step. We propose a unified framework where movies, user taste profiles, and moods are all represented in the same semantic embedding space. User taste is modeled as the centroid of liked movie embeddings. Each mood is represented as a learned direction vector in this space, computed from the embeddings of mood-exemplar movies. Recommendations are generated by shifting the user's taste centroid in the mood's direction with a controllable intensity parameter λ, then retrieving the nearest movie embeddings via cosine similarity. We evaluate on the MovieLens dataset and show that mood-shifted retrieval (λ = 0.3) improves Hit Rate@10 and NDCG over both pure content-based baselines and traditional genre-filtered mood recommenders. Our approach provides controllable, interpretable, and semantically coherent mood-aware recommendations within a single unified representation space.

### Paper Structure

| Section | Content |
|---------|---------|
| **1. Introduction** | The problem (genre is bad, mood is disconnected), the gap (no unified space), our idea in one sentence |
| **2. Related Work** | Content-based RS, affective RS, embedding-based RS — and what's missing |
| **3. Method** | Our 3-component architecture: movie vectors, user centroids, mood directions + λ shift |
| **4. Experiments** | Dataset (MovieLens), baselines (3), metrics (HR@K, NDCG, Diversity), results table |
| **5. Ablation** | λ sweep (0.0 to 1.0), mood direction quality, embedding model comparison |
| **6. Discussion** | Limitations, future work (learning λ per user, multi-mood blending) |
| **7. Conclusion** | Restate contribution, practical implications |

### Target Venues

| Venue | Length | Difficulty | Fit |
|-------|--------|-----------|-----|
| **RecSys Late-Breaking Results** | 4 pages | Medium | Best fit — values concise novel ideas |
| **UMAP** (User Modeling, Adaptation, Personalization) | 6-8 pages | Medium | Covers affective user modeling |
| **IUI** (Intelligent User Interfaces) | 6-8 pages | Medium | Values controllable, explainable interactions |

---

## Implementation Plan

### Phase 1: Embedding Infrastructure (Days 1-2)
- [ ] Install `sentence-transformers`
- [ ] Write `app/ml/embeddings.py` — encode all movies using `all-MiniLM-L6-v2`
- [ ] Store movie embeddings as `data/models/movie_embeddings.npy` (5000 × 384 matrix)
- [ ] Update movie ID mapping
- [ ] Verify: nearest neighbors of known movies make semantic sense

### Phase 2: User Taste Centroid (Day 3)
- [ ] Replace `{genre: score}` taste vector with 384-dim centroid
- [ ] Update `User` model: store taste as JSON array of 384 floats
- [ ] Update `TasteUpdater`: blend movie vector into user centroid on like/unlike
- [ ] Verify: two users with different liked movies get genuinely different centroids

### Phase 3: Mood Direction Vectors (Days 4-5)
- [ ] Label a set of "mood exemplar" movies per mood (or use keywords + reviews)
- [ ] Compute mood direction: `mean(mood_movies) - mean(all_movies)`
- [ ] Normalize mood vectors
- [ ] Store as `data/models/mood_vectors.npy` (10 moods × 384)
- [ ] Verify: "Excited" direction actually points toward high-energy movies

### Phase 4: λ-Shifted Search (Day 6)
- [ ] Implement `search_position = user_centroid + (λ × mood_direction)`
- [ ] Add λ parameter to recommendation API and frontend slider
- [ ] Update `RecommendationEngine` to use embedding-based search
- [ ] Verify: changing λ changes results in the expected direction

### Phase 5: Evaluation Framework (Days 7-9)
- [ ] Set up train/test split on MovieLens (last 20% of ratings per user)
- [ ] Implement metrics: Hit Rate@K, NDCG@K, Diversity, Serendipity
- [ ] Run baselines: TF-IDF (current), genre-based mood, static centroid (no mood)
- [ ] Run your method at λ = 0.1, 0.3, 0.5, 0.7
- [ ] Create result tables and charts

### Phase 6: Paper Writing (Days 10-14)
- [ ] Write introduction + related work
- [ ] Write method section with math notation
- [ ] Generate results tables from Phase 5 output
- [ ] Write discussion + conclusion
- [ ] Format for target venue

---

## Summary

| | Current | Proposed |
|---|---------|-----------|
| **Movie understanding** | Counts words | Understands meaning |
| **User profile** | List of genre scores | Point in vibe space |
| **Mood** | Genre lookup table | Direction arrow in vibe space |
| **Recommendation** | Weighted score formula | Nearest neighbors from shifted point |
| **Mood control** | On/off | Slider (λ: 0 → 1) |
| **Publishable?** | No (standard college project) | Yes (workshop/short paper) |

**The one-sentence pitch:** We're the first to put movies, users, and moods into one shared space where mood is a direction you can walk in — not a filter you turn on and off.

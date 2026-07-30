#!/usr/bin/env python3
"""
Fetch movie details from TMDB API to enrich MovieLens data
"""
import requests
import pandas as pd
import time
import os
import sys
import json

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
        response = requests.get(url, params=params, timeout=10)
        if response.status_code == 200:
            return response.json()
        elif response.status_code == 404:
            return None  # Movie not found
        else:
            print(f"  Error {response.status_code} for movie {tmdb_id}")
            return None
    except Exception as e:
        print(f"  Exception fetching movie {tmdb_id}: {e}")
        return None


def main():
    print("=" * 60)
    print("TMDB Data Fetcher")
    print("=" * 60)

    # Check API key
    if not TMDB_API_KEY or TMDB_API_KEY == 'your-tmdb-api-key-here':
        print("ERROR: TMDB_API_KEY not set in .env file")
        sys.exit(1)

    print(f"API Key: {TMDB_API_KEY[:8]}...")

    # Load MovieLens data
    print("\nLoading MovieLens data...")
    movies_df = pd.read_csv('data/raw/movies.csv')
    links_df = pd.read_csv('data/raw/links.csv')

    # Merge to get TMDB IDs
    merged = movies_df.merge(links_df, on='movieId')

    # Remove rows without TMDB ID
    merged = merged.dropna(subset=['tmdbId'])
    merged['tmdbId'] = merged['tmdbId'].astype(int)

    print(f"Total movies with TMDB IDs: {len(merged)}")

    # Limit to first N movies (adjust as needed)
    MAX_MOVIES = 5000
    merged = merged.head(MAX_MOVIES)
    print(f"Processing first {len(merged)} movies...")

    # Check if we have a partial file to resume from
    output_file = 'data/raw/movies_enriched.csv'
    already_fetched = set()

    if os.path.exists(output_file):
        existing_df = pd.read_csv(output_file)
        already_fetched = set(existing_df['tmdb_id'].tolist())
        print(f"Found existing file with {len(already_fetched)} movies. Resuming...")

    enriched_movies = []
    errors = 0

    print("\nFetching movie details from TMDB...")
    print("-" * 60)

    for idx, row in merged.iterrows():
        tmdb_id = int(row['tmdbId'])

        # Skip if already fetched
        if tmdb_id in already_fetched:
            continue

        # Fetch from TMDB
        details = fetch_movie_details(tmdb_id)

        if details:
            # Extract genres
            genres = [g['name'] for g in details.get('genres', [])]

            # Extract cast (top 5)
            credits = details.get('credits', {})
            cast = credits.get('cast', [])[:5]
            cast_names = [actor['name'] for actor in cast]

            # Extract director
            crew = credits.get('crew', [])
            directors = [c['name'] for c in crew if c.get('job') == 'Director']
            director = directors[0] if directors else None

            # Extract keywords
            keywords = details.get('keywords', {}).get('keywords', [])[:10]
            keyword_names = [k['name'] for k in keywords]

            # Extract release year
            release_date = details.get('release_date', '')
            release_year = int(release_date[:4]) if release_date and len(release_date) >= 4 else None

            movie_data = {
                'movieId': row['movieId'],
                'tmdb_id': tmdb_id,
                'imdb_id': row['imdbId'],
                'title': details.get('title', row['title']),
                'original_title': details.get('original_title'),
                'release_year': release_year,
                'overview': details.get('overview', ''),
                'genres': json.dumps(genres),
                'cast': json.dumps(cast_names),
                'director': director,
                'keywords': json.dumps(keyword_names),
                'vote_average': details.get('vote_average', 0),
                'vote_count': details.get('vote_count', 0),
                'popularity': details.get('popularity', 0),
                'runtime': details.get('runtime'),
                'poster_path': details.get('poster_path'),
                'backdrop_path': details.get('backdrop_path'),
                'original_language': details.get('original_language'),
                'tagline': details.get('tagline', '')
            }

            enriched_movies.append(movie_data)

            # Progress update
            total_done = len(already_fetched) + len(enriched_movies)
            if total_done % 50 == 0:
                print(f"  Processed {total_done}/{MAX_MOVIES} movies...")

                # Save intermediate results every 200 movies
                if total_done % 200 == 0 and enriched_movies:
                    save_progress(enriched_movies, already_fetched, output_file)
                    print(f"  [Checkpoint saved]")
        else:
            errors += 1

        # Rate limit: ~3 requests per second (TMDB allows 40/10sec)
        time.sleep(0.35)

    # Final save
    save_progress(enriched_movies, already_fetched, output_file)

    print("-" * 60)
    print(f"\nDone!")
    print(f"  Successfully fetched: {len(enriched_movies)} movies")
    print(f"  Errors/Not found: {errors}")
    print(f"  Total in file: {len(already_fetched) + len(enriched_movies)}")
    print(f"\nOutput saved to: {output_file}")


def save_progress(new_movies, already_fetched_ids, output_file):
    """Save progress to CSV file"""
    if not new_movies:
        return

    new_df = pd.DataFrame(new_movies)

    if os.path.exists(output_file) and already_fetched_ids:
        existing_df = pd.read_csv(output_file)
        combined_df = pd.concat([existing_df, new_df], ignore_index=True)
        combined_df.to_csv(output_file, index=False)
    else:
        new_df.to_csv(output_file, index=False)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""
Load enriched movie data from CSV into the database
"""
import os
import sys
import pandas as pd
import json

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Movie


def load_movies():
    """Load movies from CSV into database"""
    print("=" * 50)
    print("Movie Data Loader")
    print("=" * 50)

    app = create_app()

    with app.app_context():
        # Check if movies already exist
        existing_count = Movie.query.count()
        if existing_count > 0:
            print(f"\nDatabase already has {existing_count} movies.")
            response = input("Do you want to reload? This will delete existing movies. (yes/no): ")
            if response.lower() != 'yes':
                print("Aborted.")
                return

            # Delete existing movies
            print("Deleting existing movies...")
            Movie.query.delete()
            db.session.commit()

        # Read enriched CSV
        csv_path = 'data/raw/movies_enriched.csv'
        if not os.path.exists(csv_path):
            print(f"ERROR: {csv_path} not found!")
            print("Run fetch_tmdb_data.py first.")
            sys.exit(1)

        print(f"\nReading {csv_path}...")
        df = pd.read_csv(csv_path)
        print(f"Found {len(df)} movies in CSV")

        # Load movies
        print("\nLoading movies into database...")
        movies_added = 0
        movies_skipped = 0

        for idx, row in df.iterrows():
            # Skip if essential data is missing
            if pd.isna(row.get('title')) or pd.isna(row.get('overview')):
                movies_skipped += 1
                continue

            try:
                movie = Movie(
                    movie_id=int(row['movieId']),
                    tmdb_id=int(row['tmdb_id']) if pd.notna(row.get('tmdb_id')) else None,
                    imdb_id=str(row['imdb_id']) if pd.notna(row.get('imdb_id')) else None,
                    title=str(row['title']),
                    original_title=str(row['original_title']) if pd.notna(row.get('original_title')) else None,
                    release_year=int(row['release_year']) if pd.notna(row.get('release_year')) else None,
                    overview=str(row['overview']) if pd.notna(row.get('overview')) else None,
                    tagline=str(row['tagline']) if pd.notna(row.get('tagline')) else None,
                    genres=str(row['genres']) if pd.notna(row.get('genres')) else '[]',
                    cast=str(row['cast']) if pd.notna(row.get('cast')) else '[]',
                    keywords=str(row['keywords']) if pd.notna(row.get('keywords')) else '[]',
                    director=str(row['director']) if pd.notna(row.get('director')) else None,
                    runtime=int(row['runtime']) if pd.notna(row.get('runtime')) else None,
                    original_language=str(row['original_language']) if pd.notna(row.get('original_language')) else None,
                    vote_average=float(row['vote_average']) if pd.notna(row.get('vote_average')) else 0,
                    vote_count=int(row['vote_count']) if pd.notna(row.get('vote_count')) else 0,
                    popularity=float(row['popularity']) if pd.notna(row.get('popularity')) else 0,
                    poster_path=str(row['poster_path']) if pd.notna(row.get('poster_path')) else None,
                    backdrop_path=str(row['backdrop_path']) if pd.notna(row.get('backdrop_path')) else None,
                )

                db.session.add(movie)
                movies_added += 1

                # Commit in batches
                if movies_added % 500 == 0:
                    db.session.commit()
                    print(f"  Loaded {movies_added} movies...")

            except Exception as e:
                print(f"  Error loading movie {row.get('title')}: {e}")
                movies_skipped += 1
                continue

        # Final commit
        db.session.commit()

        # Verify
        total_in_db = Movie.query.count()

        print("\n" + "=" * 50)
        print("Loading complete!")
        print(f"  Movies added: {movies_added}")
        print(f"  Movies skipped: {movies_skipped}")
        print(f"  Total in database: {total_in_db}")
        print("=" * 50)

        # Show sample
        print("\nSample movies:")
        samples = Movie.query.limit(5).all()
        for m in samples:
            print(f"  - {m.title} ({m.release_year}) - {m.get_genres_list()}")


if __name__ == '__main__':
    load_movies()

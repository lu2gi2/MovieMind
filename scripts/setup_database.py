#!/usr/bin/env python3
"""
Initialize the database and create all tables
"""
import os
import sys

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import create_app, db
from app.models import Movie, User, UserFavorite, UserInteraction


def setup_database(drop_existing=False):
    """Initialize database and create tables"""
    print("=" * 50)
    print("Database Setup")
    print("=" * 50)

    app = create_app()

    with app.app_context():
        if drop_existing:
            print("\nDropping existing tables...")
            db.drop_all()
            print("  Tables dropped.")

        print("\nCreating tables...")
        db.create_all()
        print("  Tables created:")
        print("    - movies")
        print("    - users")
        print("    - user_favorites")
        print("    - user_interactions")

        # Verify tables exist
        from sqlalchemy import inspect
        inspector = inspect(db.engine)
        tables = inspector.get_table_names()
        print(f"\nVerification: {len(tables)} tables in database")

        print("\n" + "=" * 50)
        print("Database setup complete!")
        print(f"Location: {app.config['SQLALCHEMY_DATABASE_URI']}")
        print("=" * 50)


if __name__ == '__main__':
    # Pass --drop flag to drop existing tables
    drop = '--drop' in sys.argv
    if drop:
        print("WARNING: This will drop all existing tables!")
        confirm = input("Are you sure? (yes/no): ")
        if confirm.lower() != 'yes':
            print("Aborted.")
            sys.exit(0)

    setup_database(drop_existing=drop)

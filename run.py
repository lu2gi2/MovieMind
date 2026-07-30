#!/usr/bin/env python3
"""
Main entry point for the Movie Recommendation System
"""
from app import create_app

app = create_app()

if __name__ == '__main__':
    print("=" * 50)
    print("🎬 Movie Recommendation System")
    print("=" * 50)
    print("Starting Flask server...")
    print("Open http://localhost:5000 in your browser")
    print("=" * 50)

    app.run(
        host='0.0.0.0',
        port=5000,
        debug=True
    )

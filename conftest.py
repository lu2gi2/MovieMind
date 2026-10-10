"""
Pytest configuration and fixtures
"""
import pytest
import sys
import os

# Tests must never touch the real database (fixtures call db.drop_all()).
# Set before the app is imported; python-dotenv does not override existing env vars.
os.environ['DATABASE_URL'] = 'sqlite:///:memory:'

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

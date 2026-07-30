"""
Authentication routes - Signup, Login, Logout
"""
from flask import Blueprint, request, jsonify, session
from flask_login import login_user, logout_user, login_required, current_user
from app import db
from app.models import User
import json

auth_bp = Blueprint('auth', __name__)


@auth_bp.route('/signup', methods=['POST'])
def signup():
    """
    Register a new user with preferences

    Expected JSON:
    {
        "username": "john_doe",
        "email": "john@example.com",
        "password": "securepassword",
        "preferred_genres": ["Action", "Comedy", "Sci-Fi"],
        "preferred_industry": "Hollywood",  // optional
        "preferred_format": "movies"  // optional: movies, series, both
    }
    """
    data = request.get_json()

    # Validate required fields
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    required_fields = ['username', 'email', 'password']
    for field in required_fields:
        if not data.get(field):
            return jsonify({'error': f'{field} is required'}), 400

    # Check if user already exists
    if User.query.filter_by(username=data['username']).first():
        return jsonify({'error': 'Username already taken'}), 409

    if User.query.filter_by(email=data['email']).first():
        return jsonify({'error': 'Email already registered'}), 409

    # Validate password length
    if len(data['password']) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    # Create new user
    user = User(
        username=data['username'],
        email=data['email'],
        preferred_industry=data.get('preferred_industry'),
        preferred_format=data.get('preferred_format', 'movies')
    )
    user.set_password(data['password'])

    # Set preferred genres
    preferred_genres = data.get('preferred_genres', [])
    if preferred_genres:
        user.set_preferred_genres(preferred_genres)

        # Initialize taste vector based on preferred genres
        initial_taste = {genre: 0.7 for genre in preferred_genres}
        user.set_taste_vector(initial_taste)

    try:
        db.session.add(user)
        db.session.commit()

        # Auto-login after signup
        login_user(user)

        return jsonify({
            'message': 'User registered successfully',
            'user': user.to_dict()
        }), 201

    except Exception as e:
        db.session.rollback()
        return jsonify({'error': f'Registration failed: {str(e)}'}), 500


@auth_bp.route('/login', methods=['POST'])
def login():
    """
    Login user

    Expected JSON:
    {
        "email": "john@example.com",
        "password": "securepassword"
    }
    """
    data = request.get_json()

    if not data:
        return jsonify({'error': 'No data provided'}), 400

    email = data.get('email')
    password = data.get('password')

    if not email or not password:
        return jsonify({'error': 'Email and password are required'}), 400

    # Find user by email
    user = User.query.filter_by(email=email).first()

    if not user or not user.check_password(password):
        return jsonify({'error': 'Invalid email or password'}), 401

    # Login user
    login_user(user, remember=True)

    return jsonify({
        'message': 'Login successful',
        'user': user.to_dict()
    }), 200


@auth_bp.route('/logout', methods=['POST'])
@login_required
def logout():
    """Logout current user"""
    logout_user()
    return jsonify({'message': 'Logged out successfully'}), 200


@auth_bp.route('/me', methods=['GET'])
@login_required
def get_current_user():
    """Get current logged-in user info"""
    return jsonify({
        'user': current_user.to_dict()
    }), 200


@auth_bp.route('/check', methods=['GET'])
def check_auth():
    """Check if user is authenticated"""
    if current_user.is_authenticated:
        return jsonify({
            'authenticated': True,
            'user': current_user.to_dict()
        }), 200
    else:
        return jsonify({
            'authenticated': False
        }), 200

"""
Page routes - Serve HTML templates
"""
from flask import Blueprint, render_template, redirect, url_for
from flask_login import login_required, current_user

pages_bp = Blueprint('pages', __name__)


@pages_bp.route('/')
def index():
    """Landing page"""
    if current_user.is_authenticated:
        return redirect(url_for('pages.dashboard'))
    return render_template('index.html')


@pages_bp.route('/login')
def login():
    """Login page"""
    if current_user.is_authenticated:
        return redirect(url_for('pages.dashboard'))
    return render_template('login.html')


@pages_bp.route('/signup')
def signup():
    """Signup page"""
    if current_user.is_authenticated:
        return redirect(url_for('pages.dashboard'))
    return render_template('signup.html')


@pages_bp.route('/dashboard')
@login_required
def dashboard():
    """Main dashboard with mood selector and recommendations"""
    return render_template('dashboard.html')


@pages_bp.route('/browse')
@login_required
def browse():
    """Browse all movies"""
    return render_template('browse.html')


@pages_bp.route('/favorites')
@login_required
def favorites():
    """User's favorite movies"""
    return render_template('favorites.html')


@pages_bp.route('/profile')
@login_required
def profile():
    """User profile and settings"""
    return render_template('profile.html')


@pages_bp.route('/discover')
@login_required
def discover():
    """Discover (alias of the dashboard)"""
    return render_template('dashboard.html')


@pages_bp.route('/watchlist')
@login_required
def watchlist():
    """Titles saved for later"""
    return render_template('watchlist.html')


@pages_bp.route('/taste')
@login_required
def taste():
    """What MovieMind has learned about the user's taste"""
    return render_template('taste.html')


@pages_bp.route('/movie/<int:movie_id>')
@login_required
def movie_detail(movie_id):
    """Movie details page"""
    return render_template('detail.html', movie_id=movie_id)

/**
 * MovieMind - Global JavaScript
 */

// Check authentication status and update navbar
async function checkAuth() {
    try {
        const response = await fetch('/api/auth/check');
        const data = await response.json();

        const navLogin = document.getElementById('nav-login');
        const navSignup = document.getElementById('nav-signup');
        const navUser = document.getElementById('nav-user');
        const navDashboard = document.getElementById('nav-dashboard');
        const navBrowse = document.getElementById('nav-browse');
        const navFavorites = document.getElementById('nav-favorites');

        if (data.authenticated) {
            // Show logged-in state
            if (navLogin) navLogin.style.display = 'none';
            if (navSignup) navSignup.style.display = 'none';
            if (navUser) navUser.style.display = 'block';
            if (navDashboard) navDashboard.style.display = 'block';
            if (navBrowse) navBrowse.style.display = 'block';
            if (navFavorites) navFavorites.style.display = 'block';

            // Update username
            const usernameSpan = document.getElementById('nav-username');
            if (usernameSpan) {
                usernameSpan.textContent = data.user.username;
            }
        } else {
            // Show logged-out state
            if (navLogin) navLogin.style.display = 'block';
            if (navSignup) navSignup.style.display = 'block';
            if (navUser) navUser.style.display = 'none';
            if (navDashboard) navDashboard.style.display = 'none';
            if (navBrowse) navBrowse.style.display = 'none';
            if (navFavorites) navFavorites.style.display = 'none';
        }

        return data.authenticated;
    } catch (error) {
        console.error('Auth check failed:', error);
        return false;
    }
}

// Logout handler
document.addEventListener('DOMContentLoaded', function() {
    const logoutBtn = document.getElementById('logout-btn');
    if (logoutBtn) {
        logoutBtn.addEventListener('click', async function(e) {
            e.preventDefault();

            try {
                const response = await fetch('/api/auth/logout', { method: 'POST' });
                if (response.ok) {
                    window.location.href = '/';
                }
            } catch (error) {
                console.error('Logout failed:', error);
            }
        });
    }

    // Check auth on page load
    checkAuth();
});

// Utility: Format date
function formatDate(dateString) {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
        year: 'numeric',
        month: 'short',
        day: 'numeric'
    });
}

// Utility: Truncate text
function truncateText(text, maxLength = 150) {
    if (!text) return '';
    if (text.length <= maxLength) return text;
    return text.substring(0, maxLength) + '...';
}

// Utility: Show loading spinner
function showLoading(container) {
    container.innerHTML = `
        <div class="text-center py-5">
            <div class="spinner-border text-primary" role="status">
                <span class="visually-hidden">Loading...</span>
            </div>
        </div>
    `;
}

// Utility: Show error message
function showError(container, message) {
    container.innerHTML = `
        <div class="text-center py-5">
            <i class="bi bi-exclamation-circle" style="font-size: 3rem; color: var(--danger-color);"></i>
            <p class="mt-3 text-muted">${message}</p>
            <button class="btn btn-outline-light" onclick="location.reload()">
                <i class="bi bi-arrow-clockwise me-2"></i>Try Again
            </button>
        </div>
    `;
}

// Utility: Create movie card HTML
function createMovieCard(movie, showExplanation = false) {
    return `
        <div class="col-6 col-md-4 col-lg-3">
            <div class="movie-card" onclick="showMovieDetail(${movie.movie_id})">
                <img src="${movie.poster_url || 'https://via.placeholder.com/300x450?text=No+Poster'}"
                     class="poster" alt="${movie.title}" loading="lazy">
                <div class="rating">
                    <i class="bi bi-star-fill"></i> ${movie.vote_average ? movie.vote_average.toFixed(1) : 'N/A'}
                </div>
                <button class="favorite-btn" onclick="event.stopPropagation(); toggleFavorite(${movie.movie_id}, this)">
                    <i class="bi bi-heart"></i>
                </button>
                <div class="overlay">
                    <div class="title">${movie.title}</div>
                    <div class="year">${movie.release_year || 'N/A'}</div>
                    ${showExplanation && movie.explanation ? `<div class="explanation-badge">${movie.explanation}</div>` : ''}
                </div>
            </div>
        </div>
    `;
}

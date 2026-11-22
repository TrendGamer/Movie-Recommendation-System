"""Convenient public functions for the CineMatch app and notebooks."""
from .data_preprocessing import load_movies, load_ratings, load_tags
from .content_based import build_similarity_matrix, recommend_similar_movies
from .popularity_based import get_popular_movies


def get_movie_details(movie_id, movies):
    """Return the row for a movie ID, or None when it is unknown."""
    match = movies.loc[movies["movieId"] == movie_id]
    return None if match.empty else match.iloc[0].to_dict()

"""Bayesian weighted-rating popularity ranking."""
import pandas as pd


def get_popular_movies(movies: pd.DataFrame, n: int = 10, min_votes: int = 50, genre: str | None = None) -> pd.DataFrame:
    """Rank by IMDb-style weighted mean, using the dataset-wide mean as prior."""
    if n < 1 or movies.empty:
        return movies.iloc[0:0].copy()
    data = movies.copy()
    data = data.dropna(subset=["average_rating"])
    if genre and genre != "All genres":
        data = data[data["genres"].str.contains(genre, regex=False, na=False)]
    if data.empty:
        return data
    mean_rating = float(data["average_rating"].mean())
    threshold = max(int(min_votes), 1)
    count = data["rating_count"]
    data["weighted_rating"] = (count / (count + threshold)) * data["average_rating"] + (threshold / (count + threshold)) * mean_rating
    return data.sort_values(["weighted_rating", "rating_count", "title"], ascending=[False, False, True]).head(n).reset_index(drop=True)

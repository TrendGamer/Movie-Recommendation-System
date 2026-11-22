"""A small user-based collaborative filtering baseline using Pearson similarity."""
import pandas as pd


def recommend_for_user(user_id: int, ratings: pd.DataFrame, movies: pd.DataFrame, n: int = 10, min_common: int = 3) -> pd.DataFrame:
    """Predict unseen scores from similar users' ratings; requires a known user."""
    matrix = ratings.pivot_table(index="userId", columns="movieId", values="rating")
    if user_id not in matrix.index:
        raise ValueError(f"User {user_id} is not present in the ratings data.")
    target = matrix.loc[user_id]
    overlap = matrix.notna().mul(target.notna(), axis=1).sum(axis=1)
    similarities = matrix.T.corr(method="pearson", min_periods=min_common)[user_id].drop(user_id, errors="ignore")
    similarities = similarities[overlap.reindex(similarities.index).fillna(0) >= min_common]
    similarities = similarities.dropna()
    if similarities.empty:
        return movies.iloc[0:0].copy()
    candidates = matrix.columns[target.isna()]
    predictions = {}
    for movie_id in candidates:
        observed = matrix.loc[similarities.index, movie_id].dropna()
        weights = similarities.reindex(observed.index)
        denom = weights.abs().sum()
        if denom > 0:
            predictions[movie_id] = float((observed * weights).sum() / denom)
    if not predictions:
        return movies.iloc[0:0].copy()
    scored = pd.DataFrame({"movieId": list(predictions), "predicted_rating": list(predictions.values())})
    return scored.merge(movies, on="movieId", how="inner").sort_values("predicted_rating", ascending=False).head(n).reset_index(drop=True)
